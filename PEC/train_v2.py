import jax
import jax.numpy as jnp
import optax
import tensorflow as tf
import os
from os.path import exists, join
from typing import Union, List, NamedTuple, Callable, Tuple
import numpy as np
import haiku as hk
import multiprocessing
import mlflow
from urllib.parse import unquote, urlparse
import time
import glob
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
import functools
import graphviz
import json
import copy


import dataset_v2
import visualization
from simpleCNN import SimpleCNN2D
import utils
import modified_resnet


class expState(NamedTuple):
    """
    diff_model_state = {params: hk.Params}
    non_diff_state = {state: hk.State
                      optState: optax.OptState}
    """

    diff: dict
    non_diff: dict


def define_forward(hparams: dict, experiment: int) -> Callable:
    multi_head_instruction = hparams["multi_head_instruction"][experiment]
    n_last_second_logit = hparams["n_last_second_logit"][experiment]
    model_name = hparams["model_name"][experiment]

    model_book = {
        "resnet18": modified_resnet.ResNet18,
        "resnet34": modified_resnet.ResNet34,
        "resnet50": modified_resnet.ResNet50,
        "resnet101": modified_resnet.ResNet101,
        "simpleCNN": SimpleCNN2D,
        "mlp": hk.nets.MLP,
    }

    def one_head(
        multi_head_outputs: dict, x: np.ndarray, n_logit: int, head_type: str
    ) -> None:
        classifier = hk.Linear(n_logit, name=head_type.replace("-", ""))
        multi_head_outputs[head_type + "Pred"] = classifier(x)

    def multi_head(x_in: np.ndarray) -> dict:
        x = jax.nn.relu(x_in)
        multi_head_outputs = {}
        for head_key in multi_head_instruction.keys():
            one_head(multi_head_outputs, x, multi_head_instruction[head_key], head_key)
        return multi_head_outputs

    def _forward(x: np.ndarray, is_training: bool) -> jnp.ndarray:
        # Define forward-pass.
        if "resnet" in model_name:
            module = model_book[model_name](
                num_classes=n_last_second_logit,
                resnet_v2=hparams["resnet_v2"][experiment],
            )
            x = module(x, is_training)
            x = multi_head(x)
            return x
        elif "simpleCNN" in model_name:
            module = model_book[model_name](
                output_size=n_last_second_logit,
                output_channels_list=hparams["simpleCNN_list"][experiment],
                kernel_shape=(3, 1),
                stride=hparams["simpleCNN_stride"][experiment],
                bn_decay_rate=0.9,
                activation_fn=jax.nn.relu,
                bn_flag=True,
                dropoutRate=hparams["dropout_rate"][experiment],
            )
            x = module(x, is_training)
            x = multi_head(x)
            return x
        elif "mlp" in model_name:
            flat = hk.Flatten()
            module = model_book[model_name](
                hparams["mlp_list"][experiment] + [n_last_second_logit]
            )
            x = flat(x)
            x = module(x)
            x = multi_head(x)
            return x
        else:
            print(f"model_name should be in {model_book.keys()}, but get {model_name}")
            assert False

    return _forward


def define_loss_fn(
    forward: hk.Transformed,
    is_training: bool,
    optax_loss: Callable,
    hparams: dict,
    experiment: int,
) -> Callable:
    @jax.jit
    def loss_fn(
        params: hk.Params, state: hk.State, data_dict: dict
    ) -> Tuple[jnp.ndarray, Tuple[hk.State, jnp.ndarray]]:
        # Forward-pass.
        if is_training:
            # Update state.
            y_pred, state = forward.apply(params, state, data_dict["data"], is_training)
        else:
            # Do not update state.
            y_pred, _ = forward.apply(params, state, data_dict["data"], is_training)

        # Calculate loss. loss = target_loss + a * weight_decay.
        loss = 0.0
        n_head = len(hparams["multi_head_instruction"][experiment])
        for head_name in hparams["multi_head_instruction"][experiment].keys():
            pred_head_key = head_name + "Pred"
            # Calculate mean loss.
            if "regression" in hparams["problem"][experiment]:
                # If regression, true label is y_pred[pred_head_key].
                y_pred[pred_head_key] = y_pred[pred_head_key].reshape(-1)
                loss += optax_loss(y_pred[pred_head_key], data_dict[head_name]).mean()
            elif "classification" in hparams["problem"][experiment]:
                # If classification, true label is data_dict[head_name + "Label"]
                loss += optax_loss(
                    y_pred[pred_head_key], data_dict[head_name + "Label"]
                ).mean()

        # Average multi-heads loss.
        loss = loss / n_head

        # Add weight decay.
        if hparams["weight_decay"][experiment] is not None:
            decayLoss = hparams["weight_decay"][experiment] * utils.weightDecay(params)
            loss += decayLoss
        return loss, (state, y_pred)

    return loss_fn


def define_train_step(
    loss_fn: Callable,
    optimizer: optax.GradientTransformation,
    optimizer_schedule: Callable,
) -> Callable:
    @jax.jit
    def train_step(
        train_exp_state: expState, data_dict: dict
    ) -> Tuple[expState, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        # Forward-pass and backward-pass.
        (
            (loss, (train_exp_state.non_diff["state"], y_pred)),
            grads_dict,
        ) = jax.value_and_grad(loss_fn, has_aux=True)(
            train_exp_state.diff["params"], train_exp_state.non_diff["state"], data_dict
        )

        # Record inner learning rate.
        record_lr = optimizer_schedule(train_exp_state.non_diff["opt_state"][0].count)

        # Calculate gradient_update and update opt_state.
        updates, train_exp_state.non_diff["opt_state"] = optimizer.update(
            grads_dict, train_exp_state.non_diff["opt_state"]
        )

        # update params.
        train_exp_state.diff["params"] = optax.apply_updates(
            train_exp_state.diff["params"], updates
        )
        return train_exp_state, loss, y_pred, record_lr, grads_dict

    return train_step


def discard_str_keys(record, data_dict):
    """
    Discard str keys in data_dict and save fileName as identifier.
    """
    for data_key in list(data_dict.keys()):
        if data_dict[data_key].dtype == object:
            if data_key == "fileName":
                if data_key not in record.keys():
                    record[data_key] = []
                temp_data = data_dict[data_key].copy().tolist()
                record[data_key].append([d.decode() for d in temp_data])
            del data_dict[data_key]


def define_train(train_step: Callable, hparams: dict, experiment: int) -> Callable:
    def train(
        train_exp_state: expState,
        dataset: tf.data.Dataset,
    ) -> Tuple[expState, dict]:
        record = {"loss": [], "lr": [], "grads_norm": [], "y_pred": []}
        for head_name in hparams["multi_head_instruction"][experiment].keys():
            if "classification" == hparams["problem"][experiment]:
                record_name = head_name + "Label"
            elif "regression" == hparams["problem"][experiment]:
                record_name = head_name
            record[record_name] = []

        for data_dict in dataset.as_numpy_iterator():
            discard_str_keys(record, data_dict)
            # Update train_exp_state.
            train_exp_state, loss, y_pred, lr, grads_dict = train_step(
                train_exp_state, data_dict
            )
            record["loss"].append(loss.tolist())
            record["y_pred"].append({k: y_pred[k].tolist() for k in y_pred.keys()})
            record["lr"].append(lr.tolist())
            record["grads_norm"].append(utils.calculate_norm(grads_dict).tolist())
            for head_name in hparams["multi_head_instruction"][experiment].keys():
                if "classification" == hparams["problem"][experiment]:
                    record_name = head_name + "Label"
                elif "regression" == hparams["problem"][experiment]:
                    record_name = head_name
                record[record_name].append(data_dict[record_name].tolist())

        record["loss"] = jnp.mean(jnp.array(record["loss"])).tolist()
        return train_exp_state, record

    return train


def define_test(lossFn: Callable, hparams: dict, experiment: int) -> Callable:
    def test(text_exp_state: expState, dataset: tf.data.Dataset) -> dict:
        record = {"loss": [], "y_pred": []}
        for head_name in hparams["multi_head_instruction"][experiment].keys():
            if "classification" == hparams["problem"][experiment]:
                record_name = head_name + "Label"
            elif "regression" == hparams["problem"][experiment]:
                record_name = head_name
            record[record_name] = []

        for data_dict in dataset.as_numpy_iterator():
            discard_str_keys(record, data_dict)
            # Do not update state.
            loss, (_, y_pred) = lossFn(
                text_exp_state.diff["params"],
                text_exp_state.non_diff["state"],
                data_dict,
            )
            record["loss"].append(loss.tolist())
            record["y_pred"].append({k: y_pred[k].tolist() for k in y_pred.keys()})
            for head_name in hparams["multi_head_instruction"][experiment].keys():
                if "classification" == hparams["problem"][experiment]:
                    record_name = head_name + "Label"
                elif "regression" == hparams["problem"][experiment]:
                    record_name = head_name
                record[record_name].append(data_dict[record_name].tolist())

        record["loss"] = jnp.mean(jnp.array(record["loss"])).tolist()
        return record

    return test


def define_forward_and_optimizer(
    hparams: dict, experiment: int, data_name: str, DATA_SIZE: Tuple
):
    # Define _forward.
    _forward = define_forward(hparams, experiment)

    # Define optimizer and learning rate schedule.
    nData = len(glob.glob(join(DATA_ROOT, data_name, "*")))

    optimizerSchedule = utils.lr_schedule(
        hparams["lr"][experiment],
        hparams["lr_schedule_flag"][experiment],
        int(
            nData
            * hparams["split_ratio"][experiment]
            * hparams["epoch"][experiment]
            / hparams["batch_size"][experiment]
        ),
    )
    optimizer = utils.optimizerSelector(hparams["optimizer"][experiment])(
        learning_rate=optimizerSchedule
    )

    def summary_model():
        """
        Summary model.
        """

        def temp_forward(x):
            _forward(x, True)

        # Summary model.
        dummy_x = np.random.uniform(
            size=(
                hparams["batch_size"][experiment],
                DATA_SIZE[0],
                DATA_SIZE[1],
                DATA_SIZE[2],
            )
        ).astype(np.float32)
        summary_message = f"{hk.experimental.tabulate(temp_forward)(dummy_x)}"
        return summary_message

    summary_message = summary_model()

    return (_forward, optimizer, optimizerSchedule, summary_message)


def initialize_train_exp_state(DATA_SIZE, forward, optimizer, mlflow_artifact_path):
    # Initialize the parameters and states of the network and return them.
    dummy_x = np.random.uniform(
        size=(1, DATA_SIZE[0], DATA_SIZE[1], DATA_SIZE[2])
    ).astype(np.float32)
    params, state = forward.init(
        rng=jax.random.PRNGKey(42), x=dummy_x, is_training=True
    )

    # Visualize model.
    dot = hk.experimental.to_dot(forward.apply)(params, state, dummy_x, True)
    dot_plot = graphviz.Source(dot)
    dot_plot.source.replace("rankdir = TD", "rankdir = TB")
    dot_plot_save_path = join(mlflow_artifact_path, "summary")
    dot_plot.render(filename="model_plot", directory=dot_plot_save_path)

    # Initialize model and optimiser.
    opt_state = optimizer.init(params)

    # Initialize train state.
    train_exp_state = expState(
        {"params": params}, {"state": state, "opt_state": opt_state}
    )
    return train_exp_state


def save_exp_state(exp_state, epoch, mlflow_artifact_path):
    saving_history = {
        int(os.path.normpath(p).split(os.sep)[-1][5:]): p
        for p in glob.glob(join(mlflow_artifact_path, "Epoch*"))
    }
    # Only keep the latest N models.
    if len(saving_history) >= 5:
        shutil.rmtree(saving_history[min(saving_history.keys())])
        print("Delete save", min(saving_history.keys()))

    save_ckpt_dir = join(mlflow_artifact_path, "Epoch" + str(epoch))
    utils.save_data(save_ckpt_dir, exp_state.diff["params"], "params")
    utils.save_data(save_ckpt_dir, exp_state.non_diff["state"], "state")
    utils.save_data(save_ckpt_dir, exp_state.non_diff["opt_state"], "opt_state")


def restore_exp_state(starting_epoch, mlflow_artifact_path):
    restore_ckpt_dir = join(mlflow_artifact_path, "Epoch" + str(starting_epoch))
    print(f"Restore from {restore_ckpt_dir}")

    params = utils.restore(restore_ckpt_dir, "params")
    state = utils.restore(restore_ckpt_dir, "state")
    opt_state = utils.restore(restore_ckpt_dir, "opt_state")

    exp_state = expState({"params": params}, {"state": state, "opt_state": opt_state})
    return exp_state


def plot_confusion_matrix(
    train_result: dict,
    val_result: dict,
    test_result: dict,
    train_dataset_dict: dict,
    hparams: dict,
    experiment: int,
    starting_epoch: int,
    mlflowArtifactPath: str,
    transparent: bool = False,
):
    def total_mean_acc(confusionMatrix):
        cmShape = confusionMatrix.shape
        cnt = 0
        for ii in range(cmShape[0]):
            cnt += confusionMatrix[ii][ii]
        return cnt / np.sum(confusionMatrix)

    labelCodeBook = {}
    for head_name in hparams["multi_head_instruction"][experiment].keys():
        temp_list = list(
            set(
                train_dataset_dict[data_key]["metaDataAndLabel"][head_name]
                for data_key in train_dataset_dict.keys()
            )
        )
        temp_list.sort()
        labelCodeBook[head_name] = temp_list

    fig = plt.figure(
        figsize=(4 + 4 * len(hparams["multi_head_instruction"][experiment]), 12)
    )
    gs = plt.GridSpec(
        3, 1 + len(hparams["multi_head_instruction"][experiment]), figure=fig
    )

    for nR, (name, result) in enumerate(
        zip(["Train", "validation", "Test"], [train_result, val_result, test_result])
    ):
        text_info = f"Total Mean Acc: \n"
        for nH, head_name in enumerate(
            hparams["multi_head_instruction"][experiment].keys()
        ):

            dataTrue = np.argmax(np.concatenate(result[head_name + "Label"]), axis=1)
            dataPred = np.argmax(
                np.concatenate([re[head_name + "Pred"] for re in result["y_pred"]]),
                axis=1,
            )
            dataCm = tf.math.confusion_matrix(
                dataTrue,
                dataPred,
                num_classes=np.max(dataTrue) + 1,
            ).numpy()
            # Plot train confusion matrix.
            dataAxes = fig.add_subplot(gs[nR, nH])
            sns.heatmap(dataCm, annot=True, fmt="d", ax=dataAxes, cmap="YlGnBu")

            dataAxes.set_title(name + " " + head_name, wrap=True)
            dataAxes.set_xlabel("Pred Labels")
            dataAxes.set_ylabel("True Labels")
            dataAxes.set_xticklabels(
                labelCodeBook[head_name], fontsize=6.5, rotation=np.pi / 4
            )
            dataAxes.set_yticklabels(
                labelCodeBook[head_name], fontsize=6.5, rotation=np.pi / 4
            )

            text_info += f"{head_name}: {total_mean_acc(dataCm):.3f}\n"

        textAxes = fig.add_subplot(gs[nR, nH + 1])
        textAxes.axis("off")
        textAxes.text(0, 0.3, text_info, wrap=True)

    # Set fig property.
    plt.tight_layout()

    figSavePath = join(
        mlflowArtifactPath, "confusion_matrix", "Epoch" + str(starting_epoch)
    )
    # Save fig.
    if not exists(figSavePath):
        os.makedirs(figSavePath)
        print(f"Create {figSavePath} to store image.png")
    fig.savefig(join(figSavePath, "confusion matrix.png"), transparent=transparent)

    # Close fig to release memory.
    # RuntimeWarning: More than 20 figures have been opened.
    # Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained
    # until explicitly closed and may consume too much memory.
    # (To control this warning, see the rcParam `figure.max_open_warning`).
    plt.close(fig)


def plot_residual_fig(
    trainResult: dict,
    val_result: dict,
    testResult: dict,
    hparams: dict,
    experiment: int,
    starting_epoch: int,
    mlflowArtifactPath: str,
    transparent=False,
):
    def createTable(result, headType):

        headClass = sorted(
            list(set(np.concatenate(result[headType]).astype(np.int32).tolist()))
        )
        classStatistic = {hC: {"Name": str(hC) + " mm"} for hC in headClass}
        for classKey in classStatistic.keys():
            dataTrue = np.concatenate(result[headType])
            dataPred = np.concatenate(
                [re[headType + "Pred"] for re in result["y_pred"]]
            )
            dataIndex = np.where(dataTrue == classKey)
            classStatistic[classKey]["True"] = dataTrue[dataIndex]
            classStatistic[classKey]["Pred"] = dataPred[dataIndex]
            classStatistic[classKey]["Abs Error Mean"] = np.mean(
                np.abs(dataTrue[dataIndex] - dataPred[dataIndex])
            )
            classStatistic[classKey]["Abs Error Std"] = np.std(
                np.abs(dataTrue[dataIndex] - dataPred[dataIndex])
            )
            classStatistic[classKey]["Amount"] = len(dataTrue[dataIndex])
        tableReturn = {}
        tableReturn["colLabels"] = [
            classStatistic[k]["Name"] for k in classStatistic.keys()
        ]
        tableReturn["rowLabels"] = ["Amount", "Abs Error Mean", "Abs Error Std"]
        tableReturn["cellText"] = [
            ["%.3f" % classStatistic[k][row] for k in classStatistic.keys()]
            for row in ["Amount", "Abs Error Mean", "Abs Error Std"]
        ]
        return tableReturn, classStatistic

    fig = plt.figure(
        figsize=(5 + 5 * len(hparams["multi_head_instruction"][experiment]), 12)
    )
    gs = plt.GridSpec(
        3, 3 * len(hparams["multi_head_instruction"][experiment]), figure=fig
    )

    for nR, (name, result) in enumerate(
        zip(["Train", "validation", "Test"], [trainResult, val_result, testResult])
    ):
        for nH, headType in enumerate(
            hparams["multi_head_instruction"][experiment].keys()
        ):
            dataTrue = np.concatenate(result[headType])
            dataPred = np.concatenate(
                [re[headType + "Pred"] for re in result["y_pred"]]
            )

            tableReturn, classStatistic = createTable(result, headType)

            # Residual plots.
            predictedAxes = fig.add_subplot(gs[nR, 3 * nH])
            residualAxes = fig.add_subplot(gs[nR, 3 * nH + 1])
            tableAxes = fig.add_subplot(gs[nR, 3 * nH + 2])

            predictedAxes.plot(dataPred, dataTrue, "o", markersize=0.5)
            predictedAxes.set_title(
                name + " " + headType + " Regression Plot", wrap=True
            )
            predictedAxes.set_xlabel("Pred " + headType + " (mm)")
            predictedAxes.set_ylabel("True " + headType + " (mm)")

            # residualAxes.plot(thicknessTrue,
            #                   thicknessPred - thicknessTrue, 'o')
            residualAxes.violinplot(
                [classStatistic[k]["Pred"] for k in classStatistic.keys()],
                list(classStatistic.keys()),
                widths=(list(classStatistic.keys())[1] - list(classStatistic.keys())[0])
                / 2,
                showmeans=True,
                showmedians=True,
                showextrema=True,
            )
            residualAxes.set_title(name + " " + headType + " Residual Plot", wrap=True)
            residualAxes.set_xlabel("True " + headType + " (mm)")
            residualAxes.set_ylabel("Residual (mm)")

            tableAxes.table(
                cellText=tableReturn["cellText"],
                rowLabels=tableReturn["rowLabels"],
                colLabels=tableReturn["colLabels"],
                loc="center",
            )
            tableAxes.axis("off")
            tableAxes.set_title(name + " " + headType + " Summary", wrap=True)

    figSavePath = join(
        mlflowArtifactPath, "residual_plot", "Epoch" + str(starting_epoch)
    )
    # Set fig property.
    plt.tight_layout()

    # Save fig.
    if not exists(figSavePath):
        os.makedirs(figSavePath)
        print(f"Create {figSavePath} to store image.png")
    fig.savefig(join(figSavePath, "residual plot.png"), transparent=transparent)

    # Close fig to release memory.
    # RuntimeWarning: More than 20 figures have been opened.
    # Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained
    # until explicitly closed and may consume too much memory.
    # (To control this warning, see the rcParam `figure.max_open_warning`).
    plt.close(fig)


def convert_to_tflite(hparams, experiment, forward, params, state, tflite_save_path):
    INPUT_SHAPE = hparams["n_samples"][experiment]
    simpleCNN_list = hparams["simpleCNN_list"][experiment]

    def tf_simple_cnn(input_layer):
        # First ConvBnActivation.
        x = tf.keras.layers.Conv2D(
            filters=simpleCNN_list[0],
            kernel_size=(3, 1),
            strides=(2, 1),
            padding="same",
            name="simple_cnn2_d/conv_batch_activation/conv2_d",
        )(input_layer)
        x = tf.keras.layers.BatchNormalization(
            momentum=0.9,
            epsilon=1e-5,
            name="simple_cnn2_d/conv_batch_activation/batch_norm",
        )(x)
        x = tf.keras.layers.ReLU()(x)

        # Second ConvBnActivation.
        x = tf.keras.layers.Conv2D(
            filters=simpleCNN_list[1],
            kernel_size=(3, 1),
            strides=(2, 1),
            padding="same",
            name="simple_cnn2_d/conv_batch_activation_1/conv2_d",
        )(x)
        x = tf.keras.layers.BatchNormalization(
            momentum=0.9,
            epsilon=1e-5,
            name="simple_cnn2_d/conv_batch_activation_1/batch_norm",
        )(x)
        x = tf.keras.layers.ReLU()(x)

        # Third ConvBnActivation.
        x = tf.keras.layers.Conv2D(
            filters=simpleCNN_list[2],
            kernel_size=(3, 1),
            strides=(2, 1),
            padding="same",
            name="simple_cnn2_d/conv_batch_activation_2/conv2_d",
        )(x)
        x = tf.keras.layers.BatchNormalization(
            momentum=0.9,
            epsilon=1e-5,
            name="simple_cnn2_d/conv_batch_activation_2/batch_norm",
        )(x)
        x = tf.keras.layers.ReLU()(x)

        # Last Dense.
        x = tf.keras.layers.GlobalMaxPool2D()(x)
        x = tf.keras.layers.Dense(units=64, name="simple_cnn2_d/linear")(x)
        return x

    def tf_multihead_model():
        # Define model layers.
        input_layer = tf.keras.Input(shape=(INPUT_SHAPE, 1, 1))
        x = tf_simple_cnn(input_layer)

        x = tf.keras.layers.ReLU()(x)

        y1_output = tf.keras.layers.Dense(
            units=hparams["multi_head_instruction"][experiment]["Thickness"],
            activation=None,
            name="Thickness",
        )(x)

        # y2_output = tf.keras.layers.Dense(
        #     units=hparams["nLift-offLogit"][experiment],
        #     activation=None,
        #     name="Lift-offLabel",
        # )(x)

        # Define the model with the input layer and a list of output layers
        model = tf.keras.Model(inputs=input_layer, outputs=y1_output)
        return model

    def hk_simple_cnn(input_layer, is_training):
        conv1 = hk.Conv2D(
            output_channels=simpleCNN_list[0],
            kernel_shape=(3, 1),
            stride=(2, 1),
            padding="SAME",
        )(input_layer)
        bn1 = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.9)(
            conv1, is_training
        )
        relu1 = jax.nn.relu(bn1)

        conv2 = hk.Conv2D(
            output_channels=simpleCNN_list[1],
            kernel_shape=(3, 1),
            stride=(2, 1),
            padding="SAME",
        )(relu1)
        bn2 = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.9)(
            conv2, is_training
        )
        relu2 = jax.nn.relu(bn2)

        conv3 = hk.Conv2D(
            output_channels=simpleCNN_list[2],
            kernel_shape=(3, 1),
            stride=(2, 1),
            padding="SAME",
        )(relu2)
        bn3 = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.9)(
            conv3, is_training
        )
        relu3 = jax.nn.relu(bn3)

        maxpool = hk.MaxPool(
            window_shape=(1, relu3.shape[1], 1, 1),
            strides=(1, 1, 1, 1),
            padding="VALID",
        )(relu3)
        maxpool = hk.Flatten()(maxpool)
        linear = hk.Linear(output_size=64)(maxpool)

        return {
            "conv1": conv1,
            "bn1": bn1,
            "relu1": relu1,
            "conv2": conv2,
            "bn2": bn2,
            "relu2": relu2,
            "conv3": conv3,
            "bn3": bn3,
            "relu3": relu3,
            "maxpool": maxpool,
            "linear": linear,
        }

    def hk_multihead_model(x: np.ndarray, is_training: bool):
        simple_cnn_out = hk_simple_cnn(x, is_training)

        x = jax.nn.relu(simple_cnn_out["linear"])

        thickness_outputs = hk.Linear(
            hparams["multi_head_instruction"][experiment]["Thickness"], name="Thickness"
        )(x)
        # liftoff_outputs = hk.Linear(6)(x)
        simple_cnn_out["ThicknessPred"] = thickness_outputs
        # simple_cnn_out["Lift-offLabel"] = liftoff_outputs
        return simple_cnn_out

    # Restore JAX model to TF model.
    tf_model = tf_multihead_model()

    # Transform forward-pass into pure functions.
    forward_2 = hk.without_apply_rng(hk.transform_with_state(hk_multihead_model))
    # Initialize the parameters and states of the network and return them.
    params_2, state_2 = forward_2.init(
        rng=jax.random.PRNGKey(42),
        x=np.random.uniform(size=(16, INPUT_SHAPE, 1, 1)).astype(np.float32),
        is_training=True,
    )

    print(f"TF model layer")
    for layer in tf_model.layers:
        print(f"Layer name: {layer.name}, parameters are:")
        for l_w in layer.weights:
            print(f"Weight name: {l_w.name}, weight shape: {l_w.shape}")
        print()
    print()

    def print_hk_model(params, state):
        for p in params.keys():
            print(f"Layer name: {p}, parameters are:")
            for l_w in params[p].keys():
                print(f"Weight name: {l_w}, weight shape: {params[p][l_w].shape}")
            print()
        for s in state.keys():
            print(f"Layer name: {s}, state are:")
            for l_w in state[s].keys():
                print(f"Weight name: {l_w}, weight shape: {state[s][l_w].shape}")
            print()
        print()

    print(f"Hk model layer")
    print_hk_model(params, state)

    print(f"Hk model2 layer")
    print_hk_model(params_2, state_2)

    def load_hk_to_tf(tf_model, params, state):
        for layer in tf_model.layers:
            print(f"Loading tf layer {layer.name} ...")
            if layer.name == "simple_cnn2_d/conv_batch_activation/conv2_d":
                layer.set_weights(
                    [
                        params["simple_cnn2_d/~/conv_batch_activation/~/conv2_d"]["w"],
                        params["simple_cnn2_d/~/conv_batch_activation/~/conv2_d"]["b"],
                    ]
                )
            elif layer.name == "simple_cnn2_d/conv_batch_activation/batch_norm":
                layer.set_weights(
                    [
                        params["simple_cnn2_d/~/conv_batch_activation/~/batch_norm"][
                            "scale"
                        ].reshape(-1),
                        params["simple_cnn2_d/~/conv_batch_activation/~/batch_norm"][
                            "offset"
                        ].reshape(-1),
                        state[
                            "simple_cnn2_d/~/conv_batch_activation/~/batch_norm/~/mean_ema"
                        ]["hidden"].reshape(-1),
                        state[
                            "simple_cnn2_d/~/conv_batch_activation/~/batch_norm/~/var_ema"
                        ]["hidden"].reshape(-1),
                    ]
                )
            elif layer.name == "simple_cnn2_d/conv_batch_activation_1/conv2_d":
                layer.set_weights(
                    [
                        params["simple_cnn2_d/~/conv_batch_activation_1/~/conv2_d"][
                            "w"
                        ],
                        params["simple_cnn2_d/~/conv_batch_activation_1/~/conv2_d"][
                            "b"
                        ],
                    ]
                )
            elif layer.name == "simple_cnn2_d/conv_batch_activation_1/batch_norm":
                layer.set_weights(
                    [
                        params["simple_cnn2_d/~/conv_batch_activation_1/~/batch_norm"][
                            "scale"
                        ].reshape(-1),
                        params["simple_cnn2_d/~/conv_batch_activation_1/~/batch_norm"][
                            "offset"
                        ].reshape(-1),
                        state[
                            "simple_cnn2_d/~/conv_batch_activation_1/~/batch_norm/~/mean_ema"
                        ]["hidden"].reshape(-1),
                        state[
                            "simple_cnn2_d/~/conv_batch_activation_1/~/batch_norm/~/var_ema"
                        ]["hidden"].reshape(-1),
                    ]
                )
            elif layer.name == "simple_cnn2_d/conv_batch_activation_2/conv2_d":
                layer.set_weights(
                    [
                        params["simple_cnn2_d/~/conv_batch_activation_2/~/conv2_d"][
                            "w"
                        ],
                        params["simple_cnn2_d/~/conv_batch_activation_2/~/conv2_d"][
                            "b"
                        ],
                    ]
                )
            elif layer.name == "simple_cnn2_d/conv_batch_activation_2/batch_norm":
                layer.set_weights(
                    [
                        params["simple_cnn2_d/~/conv_batch_activation_2/~/batch_norm"][
                            "scale"
                        ].reshape(-1),
                        params["simple_cnn2_d/~/conv_batch_activation_2/~/batch_norm"][
                            "offset"
                        ].reshape(-1),
                        state[
                            "simple_cnn2_d/~/conv_batch_activation_2/~/batch_norm/~/mean_ema"
                        ]["hidden"].reshape(-1),
                        state[
                            "simple_cnn2_d/~/conv_batch_activation_2/~/batch_norm/~/var_ema"
                        ]["hidden"].reshape(-1),
                    ]
                )

            elif layer.name == "simple_cnn2_d/linear":
                layer.set_weights(
                    [
                        params["simple_cnn2_d/~/linear"]["w"],
                        params["simple_cnn2_d/~/linear"]["b"],
                    ]
                )

            elif layer.name == "Thickness":
                layer.set_weights(
                    [
                        params["Thickness"]["w"],
                        params["Thickness"]["b"],
                    ]
                )

            else:
                if len(layer.weights) != 0:
                    raise Exception("weights name incorrect")
                else:
                    continue
        return tf_model

    def load_hk_to_hk2(params, state, params_2, state_2):
        for layer in params_2.keys():
            if layer == "conv2_d":
                params_2[layer]["w"] = params[
                    "simple_cnn2_d/~/conv_batch_activation/~/conv2_d"
                ]["w"]
                params_2[layer]["b"] = params[
                    "simple_cnn2_d/~/conv_batch_activation/~/conv2_d"
                ]["b"]
            elif layer == "batch_norm":
                params_2[layer]["scale"] = params[
                    "simple_cnn2_d/~/conv_batch_activation/~/batch_norm"
                ]["scale"]
                params_2[layer]["offset"] = params[
                    "simple_cnn2_d/~/conv_batch_activation/~/batch_norm"
                ]["offset"]

            elif layer == "conv2_d_1":
                params_2[layer]["w"] = params[
                    "simple_cnn2_d/~/conv_batch_activation_1/~/conv2_d"
                ]["w"]
                params_2[layer]["b"] = params[
                    "simple_cnn2_d/~/conv_batch_activation_1/~/conv2_d"
                ]["b"]
            elif layer == "batch_norm_1":
                params_2[layer]["scale"] = params[
                    "simple_cnn2_d/~/conv_batch_activation_1/~/batch_norm"
                ]["scale"]
                params_2[layer]["offset"] = params[
                    "simple_cnn2_d/~/conv_batch_activation_1/~/batch_norm"
                ]["offset"]

            elif layer == "conv2_d_2":
                params_2[layer]["w"] = params[
                    "simple_cnn2_d/~/conv_batch_activation_2/~/conv2_d"
                ]["w"]
                params_2[layer]["b"] = params[
                    "simple_cnn2_d/~/conv_batch_activation_2/~/conv2_d"
                ]["b"]
            elif layer == "batch_norm_2":
                params_2[layer]["scale"] = params[
                    "simple_cnn2_d/~/conv_batch_activation_2/~/batch_norm"
                ]["scale"]
                params_2[layer]["offset"] = params[
                    "simple_cnn2_d/~/conv_batch_activation_2/~/batch_norm"
                ]["offset"]

            elif layer == "linear":
                params_2[layer]["w"] = params["simple_cnn2_d/~/linear"]["w"]
                params_2[layer]["b"] = params["simple_cnn2_d/~/linear"]["b"]

            elif layer == "Thickness":
                params_2[layer]["w"] = params["Thickness"]["w"]
                params_2[layer]["b"] = params["Thickness"]["b"]

            else:
                raise Exception("params name incorrect")
        for layer in state_2.keys():
            if layer == "batch_norm/~/mean_ema":
                state_2[layer]["hidden"] = state[
                    "simple_cnn2_d/~/conv_batch_activation/~/batch_norm/~/mean_ema"
                ]["hidden"]
                state_2[layer]["average"] = state[
                    "simple_cnn2_d/~/conv_batch_activation/~/batch_norm/~/mean_ema"
                ]["average"]
            elif layer == "batch_norm/~/var_ema":
                state_2[layer]["hidden"] = state[
                    "simple_cnn2_d/~/conv_batch_activation/~/batch_norm/~/var_ema"
                ]["hidden"]
                state_2[layer]["average"] = state[
                    "simple_cnn2_d/~/conv_batch_activation/~/batch_norm/~/var_ema"
                ]["average"]

            elif layer == "batch_norm_1/~/mean_ema":
                state_2[layer]["hidden"] = state[
                    "simple_cnn2_d/~/conv_batch_activation_1/~/batch_norm/~/mean_ema"
                ]["hidden"]
                state_2[layer]["average"] = state[
                    "simple_cnn2_d/~/conv_batch_activation_1/~/batch_norm/~/mean_ema"
                ]["average"]
            elif layer == "batch_norm_1/~/var_ema":
                state_2[layer]["hidden"] = state[
                    "simple_cnn2_d/~/conv_batch_activation_1/~/batch_norm/~/var_ema"
                ]["hidden"]
                state_2[layer]["average"] = state[
                    "simple_cnn2_d/~/conv_batch_activation_1/~/batch_norm/~/var_ema"
                ]["average"]

            elif layer == "batch_norm_2/~/mean_ema":
                state_2[layer]["hidden"] = state[
                    "simple_cnn2_d/~/conv_batch_activation_2/~/batch_norm/~/mean_ema"
                ]["hidden"]
                state_2[layer]["average"] = state[
                    "simple_cnn2_d/~/conv_batch_activation_2/~/batch_norm/~/mean_ema"
                ]["average"]
            elif layer == "batch_norm_2/~/var_ema":
                state_2[layer]["hidden"] = state[
                    "simple_cnn2_d/~/conv_batch_activation_2/~/batch_norm/~/var_ema"
                ]["hidden"]
                state_2[layer]["average"] = state[
                    "simple_cnn2_d/~/conv_batch_activation_2/~/batch_norm/~/var_ema"
                ]["average"]
            else:
                raise Exception("state name incorrect")
        return params_2, state_2

    tf_model = load_hk_to_tf(tf_model, params, state)
    params_2, state_2 = load_hk_to_hk2(params, state, params_2, state_2)

    for layer in tf_model.layers:
        layer.trainable = False

    def test_acc(tf_model, forward, params, state):
        # Check TF model accracy.
        # Load dataset.
        data_saved_path = join(DATA_ROOT, "formatted_v2")
        dataset_name = hparams["dataset_name"][experiment]
        json_path = join(data_saved_path, dataset_name + ".json")
        train_dataset, val_dataset, test_dataset = dataset_v2.dataPipeline(
            json_path,
            splitRate=hparams["split_ratio"][experiment],
            batchSize=hparams["batch_size"][experiment],
            start_index=hparams["data_start_index"][experiment],
            n_samples=hparams["n_samples"][experiment],
            z_norm_flag=hparams["z_norm_flag"][experiment],
        )

        n_data = 0
        if "classification" == hparams["problem"][experiment]:
            thickness_corrent = 0
        elif "regression" == hparams["problem"][experiment]:
            abs_error = 0.0

        for data_dict in test_dataset.as_numpy_iterator():
            x = data_dict["data"]

            if "classification" == hparams["problem"][experiment]:
                label_name = "ThicknessLabel"
            elif "regression" == hparams["problem"][experiment]:
                label_name = "Thickness"

            true_thickness = data_dict[label_name]
            if "classification" == hparams["problem"][experiment]:
                true_thickness_one_hot_position = tf.math.argmax(true_thickness, axis=1)

            if tf_model is not None:
                pred_thickness = tf_model(x)
            elif forward is not None and params is not None and state is not None:
                yPred, _ = forward.apply(params, state, x, False)
                pred_thickness = yPred[label_name + "Pred"]
            else:
                raise Exception(
                    f"Both network_model and (forward, params, state) are None."
                )
            if "classification" == hparams["problem"][experiment]:
                pred_thickness_one_hot_position = tf.math.argmax(pred_thickness, axis=1)
                thickness_corrent += np.sum(
                    true_thickness_one_hot_position == pred_thickness_one_hot_position
                )
            elif "regression" == hparams["problem"][experiment]:
                if type(tf.constant(0)) == type(pred_thickness):
                    pred_thickness = pred_thickness.numpy()
                pred_thickness = pred_thickness.reshape(-1)
                abs_error += np.sum(np.absolute(pred_thickness - true_thickness))

            n_data += x.shape[0]

        if "classification" == hparams["problem"][experiment]:
            thickness_acc = thickness_corrent / n_data
            print(f"thickness_acc, {thickness_acc}")
        elif "regression" == hparams["problem"][experiment]:
            mean_abs_error = abs_error / n_data
            print(f"mean_abs_error, {mean_abs_error}")

    test_acc(None, forward, params, state)
    test_acc(None, forward_2, params_2, state_2)
    test_acc(tf_model, None, None, None)

    def check_out_per_layer(tf_model, forward, params, state):
        dataSavedPath = join(DATA_ROOT, "formatted")
        jsonPath = join(dataSavedPath, "data3.json")
        trainDataset, validationDataset, testDataset = dataset.dataPipeline(
            jsonPath,
            hparams["splitRatio"][experiment],
            hparams["batchSize"][experiment],
            hparams["separateRandomFlag"][experiment],
            start_index=hparams["data_start_index"][experiment],
            n_samples=hparams["n_samples"][experiment],
            flag_step_sampling=True,
        )

        for dataDict in testDataset.as_numpy_iterator():
            x = dataDict["data"]
            yPred, _ = forward.apply(params, state, x, False)
            for i, layer in enumerate(tf_model.layers):
                print(layer.name)
                middle_layer_model = tf.keras.Model(
                    tf_model.input, tf_model.layers[i].output
                )
                # for ll in middle_layer_model.layers:
                #     ll.trainable = False
                middle_layer_output = middle_layer_model(x, training=False)
                middle_layer_output = middle_layer_output.numpy()
                if layer.name == "simple_cnn2_d/conv_batch_activation/batch_norm":
                    print(
                        np.sum(yPred["bn1"] == middle_layer_output)
                        / np.sum(yPred["bn1"] == yPred["bn1"])
                    )

    # check_out_per_layer(tf_model, forward_2, params_2, state_2)

    # Convert the model.
    converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
    tflite_model = converter.convert()

    # Save the model.
    if not exists(tflite_save_path):
        os.makedirs(tflite_save_path)
        print(f"Create {tflite_save_path} directory.")
    tf_model_saving_name = hparams["dataset_name"][experiment] + "_model.tflite"
    with open(join(tflite_save_path, tf_model_saving_name), "wb") as f:
        f.write(tflite_model)

    # Load dataset.
    data_saved_path = join(DATA_ROOT, "formatted_v2")
    dataset_name = hparams["dataset_name"][experiment]
    json_path = join(data_saved_path, dataset_name + ".json")
    train_dataset, val_dataset, test_dataset = dataset_v2.dataPipeline(
        json_path,
        splitRate=hparams["split_ratio"][experiment],
        batchSize=1000,
        start_index=hparams["data_start_index"][experiment],
        n_samples=hparams["n_samples"][experiment],
        z_norm_flag=hparams["z_norm_flag"][experiment],
    )

    for dataDict in train_dataset.as_numpy_iterator():
        x_train = dataDict["data"]
        break

    # Convert the model to the TensorFlow Lite format with quantization
    def representative_dataset():
        for i in range(1000):
            yield ([x_train[i].reshape(1, INPUT_SHAPE, 1, 1)])

    converter_2 = tf.lite.TFLiteConverter.from_keras_model(tf_model)
    # Set the optimization flag.
    converter_2.optimizations = [tf.lite.Optimize.DEFAULT]
    # Enforce integer only quantization
    converter_2.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter_2.inference_input_type = tf.int8
    converter_2.inference_output_type = tf.int8
    # Provide a representative dataset to ensure we quantize correctly.
    converter_2.representative_dataset = representative_dataset
    model_tflite_quantization = converter_2.convert()
    # Save the model to disk
    tf_model_saving_name = (
        hparams["dataset_name"][experiment] + "_quantization_model.tflite"
    )
    with open(join(tflite_save_path, tf_model_saving_name), "wb") as f:
        f.write(model_tflite_quantization)

    # Run tflite inference.
    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(
        model_path=join(tflite_save_path, tf_model_saving_name)
    )
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on random input data.
    input_shape = input_details[0]["shape"]

    train_dataset, _, test_dataset = dataset_v2.dataPipeline(
        json_path,
        splitRate=hparams["split_ratio"][experiment],
        batchSize=1,
        start_index=hparams["data_start_index"][experiment],
        n_samples=hparams["n_samples"][experiment],
        z_norm_flag=hparams["z_norm_flag"][experiment],
    )

    print(f"input_shape: {input_shape}")

    def test_quantization_model_acc(dataset):
        n_correct = 0
        n_data = 0
        for dataDict in dataset.as_numpy_iterator():
            x = dataDict["data"].reshape(1, INPUT_SHAPE, 1, 1)
            y_thickness = dataDict["ThicknessLabel"]

            x_quantization = (
                x / input_details[0]["quantization_parameters"]["scales"]
                + input_details[0]["quantization_parameters"]["zero_points"]
            ).astype(np.int8)

            interpreter.set_tensor(input_details[0]["index"], x_quantization)

            interpreter.invoke()

            # The function `get_tensor()` returns a copy of the tensor data.
            # Use `tensor()` in order to get a pointer to the tensor.
            # lift_off_output_data = interpreter.get_tensor(output_details[0]["index"])
            thickness_output_data_int = interpreter.get_tensor(
                output_details[0]["index"]
            )

            thickness_output_data_float = (
                thickness_output_data_int
                - output_details[0]["quantization_parameters"]["zero_points"]
            ) / output_details[0]["quantization_parameters"]["scales"]

            if np.argmax(thickness_output_data_float) == np.argmax(y_thickness):
                n_correct += 1
            n_data += 1
        print(
            f"Number of data testing: {n_data}, Quantization model accuracy {n_correct / n_data}."
        )

    # test_quantization_model_acc(train_dataset)
    # test_quantization_model_acc(test_dataset)

    print()


def main(training_flag, DEVICE, data_root, convert_flag):
    # Get number of avaliable CPU cores.
    local_cpu_count = multiprocessing.cpu_count()
    # Get number of avaliable GPU.
    local_gpu_count = jax.local_device_count()
    print(f"-----Avaliable CPU cores: {local_cpu_count}, GPU: {local_gpu_count}-----")

    # Define log epoch.
    LOG_EPOCH = 10
    # Set mlflow parameters.
    mlflow.set_registry_uri(join(".", DEVICE))
    mlflow.set_tracking_uri(join(".", DEVICE))

    if "PECRuns" in DEVICE:
        hparams = {
            # -----------Experiment setting-----------
            "number_of_experiment": 1,
            "description": [
                "This model is trained on Q345 dataset, train without 0 aligenment."
            ],
            "run_name": ["Tflite CNN"],
            # -----------Model setting-----------
            "model_name": ["simpleCNN"],
            "n_last_second_logit": [64],
            "multi_head_instruction": [
                {"Thickness": 1}
            ],  # {"Thickness": 9, "Lift-off": 6}, {"Thickness": 4}
            "dropout_rate": [None],
            "resnet_v2": [True],
            "simpleCNN_list": [[128, 128, 128]],
            "simpleCNN_stride": [2],
            "mlp_list": [[1024, 1164, 2048]],
            # -----------Dataset setting-----------
            "dataset_name": [
                "Q345_15112022"
            ],  # aluminum, Q345_data, circle_Q345_07102022, Q345_15112022
            "n_samples": [256],
            "data_start_index": [0],
            "split_ratio": [0.8],
            "batch_size": [32],  # Smaller better i.e 8, but training is slow.
            "z_norm_flag": [True],
            # -----------optimizer setting-----------
            "optimizer": ["adam"],
            "lr": [0.001],
            "lr_schedule_flag": [True],
            # -----------Training setting-----------
            "epoch": [401],
            # None, 0.0001
            "weight_decay": [None],
            # huber_loss, l2_loss, softmax_cross_entropy
            "loss_name": ["huber_loss"],
            # regression, classification
            "problem": ["regression"],
        }
    DATA_SIZE = (hparams["n_samples"][0], 1, 1)

    # Define PRNGKey.
    rng = jax.random.PRNGKey(666)

    if training_flag:
        # Loop for hyperparameters.
        for experiment in range(hparams["number_of_experiment"]):
            # Check if the experiment has already run.
            already_ran_flag, previous_run_id, starting_epoch = utils._already_ran(
                {
                    k: hparams[k][experiment]
                    for k in hparams.keys()
                    if k != "number_of_experiment"
                }
            )
            # If already ran, skip this experiment.
            if already_ran_flag:
                print(f"Experiment is skiped.")
                continue
            # Run experiment.
            with mlflow.start_run(
                run_id=previous_run_id,
                run_name=hparams["run_name"][experiment],
                description=hparams["description"][experiment],
            ) as active_run:
                # log hyper parameters.
                utils.mf_loghyperparams(hparams, experiment)

                # Get mlflow artifact saving path.
                mlflow_artifact_path = unquote(
                    urlparse(active_run.info.artifact_uri).path
                )
                if "t50851tm" in mlflow_artifact_path:
                    mlflow_artifact_path = os.path.relpath(
                        mlflow_artifact_path, "/net/scratch2/t50851tm/momaml_jax"
                    )

                # Load dataset.
                data_saved_path = join(data_root, "formatted_v2")
                dataset_name = hparams["dataset_name"][experiment]
                json_path = join(data_saved_path, dataset_name + ".json")
                train_dataset, val_dataset, test_dataset = dataset_v2.dataPipeline(
                    json_path,
                    splitRate=hparams["split_ratio"][experiment],
                    batchSize=hparams["batch_size"][experiment],
                    start_index=hparams["data_start_index"][experiment],
                    n_samples=hparams["n_samples"][experiment],
                    z_norm_flag=hparams["z_norm_flag"][experiment],
                )
                train_dataset_dict = visualization.convert_tfpipeline_to_dict(
                    train_dataset
                )

                # Define _forward and optimizer.
                (
                    _forward,
                    optimizer,
                    optimizerSchedule,
                    summary_message,
                ) = define_forward_and_optimizer(
                    hparams, experiment, dataset_name, DATA_SIZE
                )

                # Log model architecture.
                mlflow.log_text(
                    summary_message, join("summary", "model_architecture.txt")
                )

                # Transform forward-pass into pure functions.
                forward = hk.without_apply_rng(hk.transform_with_state(_forward))

                # Define training loss function.
                loss_fn = define_loss_fn(
                    forward,
                    is_training=True,
                    optax_loss=utils.lossSelector(hparams["loss_name"][experiment]),
                    hparams=hparams,
                    experiment=experiment,
                )

                # Define train_step.
                train_step = define_train_step(loss_fn, optimizer, optimizerSchedule)

                # Define train.
                train = define_train(train_step, hparams, experiment)

                # Define test loss function.
                loss_fn_test = define_loss_fn(
                    forward,
                    is_training=False,
                    optax_loss=utils.lossSelector(hparams["loss_name"][experiment]),
                    hparams=hparams,
                    experiment=experiment,
                )

                # Define test.
                test = define_test(loss_fn_test, hparams, experiment)

                # Initialize train_exp_state.
                train_exp_state = initialize_train_exp_state(
                    DATA_SIZE, forward, optimizer, mlflow_artifact_path
                )

                # Check if restore checkpoint is needed.
                if starting_epoch != 0 and starting_epoch is not None:
                    train_exp_state = restore_exp_state(
                        starting_epoch, mlflow_artifact_path
                    )
                    if "best_test_loss" in active_run.data.metrics.keys():
                        best_test_loss = active_run.data.metrics["best_test_loss"]
                    else:
                        best_test_loss = 99999.9
                    print("Restored from Epoch", starting_epoch)
                else:
                    best_test_loss = 99999.9

                # Training loop.
                for epoch in range(starting_epoch, hparams["epoch"][experiment]):
                    start_time = time.time()
                    # Update trainState.
                    train_exp_state, train_result = train(
                        train_exp_state, train_dataset
                    )
                    val_result = test(train_exp_state, val_dataset)
                    test_result = test(train_exp_state, test_dataset)

                    if ((epoch % LOG_EPOCH == 0) and (epoch != 0)) or (
                        epoch == hparams["epoch"][experiment] - 1
                    ):
                        if "classification" in hparams["problem"][experiment]:
                            plot_confusion_matrix(
                                train_result,
                                val_result,
                                test_result,
                                train_dataset_dict,
                                hparams,
                                experiment,
                                epoch,
                                mlflow_artifact_path,
                            )
                        elif "regression" in hparams["problem"][experiment]:
                            plot_residual_fig(
                                train_result,
                                val_result,
                                test_result,
                                hparams,
                                experiment,
                                epoch,
                                mlflow_artifact_path,
                            )

                        jsonSavePath = join(mlflow_artifact_path, "result")
                        if not exists(jsonSavePath):
                            os.makedirs(jsonSavePath)
                        for name, results in zip(
                            [
                                "trainResult_" + "epoch" + str(epoch) + ".json",
                                "validationResult_" + "epoch" + str(epoch) + ".json",
                                "testResult_" + "epoch" + str(epoch) + ".json",
                            ],
                            [train_result, val_result, test_result],
                        ):
                            with open(
                                join(
                                    jsonSavePath,
                                    name,
                                ),
                                "w",
                            ) as f:
                                json.dump(results, f)

                    # Save model if train query loss is lower.
                    if (val_result["loss"] < best_test_loss) or (
                        epoch == hparams["epoch"][experiment] - 1
                    ):
                        save_exp_state(train_exp_state, epoch, mlflow_artifact_path)

                    best_test_loss = min(val_result["loss"], best_test_loss)

                    # Log metric.
                    mlflow.log_metric("Epoch", epoch, step=epoch)
                    mlflow.log_metric("train_loss", train_result["loss"], step=epoch)
                    mlflow.log_metric("validation_loss", val_result["loss"], step=epoch)
                    mlflow.log_metric("test_loss", test_result["loss"], step=epoch)
                    mlflow.log_metric(
                        "learning_rate", train_result["lr"][-1], step=epoch
                    )
                    mlflow.log_metric("best_test_loss", best_test_loss, step=epoch)
                    mlflow.log_metric(
                        "grads_norm",
                        jnp.mean(jnp.array(train_result["grads_norm"])),
                        step=epoch,
                    )

                    print_message = f"Run id: {active_run.info.run_id} \n \
                                        Epoch: {epoch} \n \
                                        time: {time.time() - start_time} s \n \
                                        training loss: {train_result['loss']} \n \
                                        validation loss: {val_result['loss']} \n \
                                        test loss: {test_result['loss']} \n"
                    mlflow.log_text(print_message, f"Message/Training_Epoch{epoch}.txt")
                    print(print_message)

    if convert_flag:
        experiment = 0
        # Get provious run ID.
        already_ran_flag, previous_run_id, starting_epoch = utils._already_ran(
            {
                k: hparams[k][experiment]
                for k in hparams.keys()
                if k != "number_of_experiment"
            }
        )
        with mlflow.start_run(
            run_id=previous_run_id, run_name=hparams["run_name"][experiment]
        ) as active_run:
            mlflow_artifact_path = unquote(urlparse(active_run.info.artifact_uri).path)
            # Restore train state.
            train_exp_state = restore_exp_state(starting_epoch, mlflow_artifact_path)
            # Define forword.
            (
                _forward,
                optimizer,
                optimizerSchedule,
                summary_message,
            ) = define_forward_and_optimizer(
                hparams, experiment, hparams["dataset_name"][experiment], DATA_SIZE
            )
            # Transform forward-pass into pure functions.
            forward = hk.without_apply_rng(hk.transform_with_state(_forward))

            convert_to_tflite(
                hparams,
                experiment,
                forward,
                train_exp_state.diff["params"],
                train_exp_state.non_diff["state"],
                tflite_save_path=join(mlflow_artifact_path, "tflite"),
            )


if __name__ == "__main__":
    # Ensure TF does not see GPU and grab all GPU memory.
    tf.config.set_visible_devices([], device_type="GPU")
    DEVICE = "PECRuns"
    DATA_ROOT = join("datasets", "PEC")

    main(
        training_flag=False,
        DEVICE=DEVICE,
        data_root=DATA_ROOT,
        convert_flag=True,
    )
