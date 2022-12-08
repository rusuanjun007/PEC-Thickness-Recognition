import matplotlib.pyplot as plt
import os
from os.path import exists, join
import numpy as np
import tensorflow as tf


import dataset_v2


def convert_tfpipeline_to_dict(dataset: tf.data.Dataset) -> dict:
    data_label_dict = {}
    for data in dataset.as_numpy_iterator():
        temp = {}
        temp_key = data["fileName"][0].decode("utf-8")
        temp[temp_key] = {"data": data["data"].reshape(-1), "metaDataAndLabel": {}}
        for data_key in data.keys():
            if data_key != "data" and data_key != "fileName":
                value = data[data_key][0]
                try:
                    value = value.decode("utf-8")
                except AttributeError:
                    pass
                temp[temp_key]["metaDataAndLabel"][data_key] = value
        data_label_dict.update(temp)
    return data_label_dict


def data_visualization(
    data_label_dict: dict,
    save_path: str,
    fig_name: str,
    target_dict: dict,
    transparent=True,
):
    """
    visualization of secific label.
    """

    colorBook = {
        0: "tab:blue",
        1: "tab:orange",
        2: "tab:green",
        3: "tab:red",
        4: "tab:purple",
        5: "tab:brown",
        6: "tab:pink",
        7: "tab:gray",
        8: "tab:olive",
        9: "tab:cyan",
    }

    fig = plt.figure(figsize=[10, 10], dpi=200)
    gs = plt.GridSpec(1, 1, figure=fig)
    data_ax = fig.add_subplot(gs[0, 0])

    legend_recorded = {}
    for data_key in data_label_dict.keys():
        label_match = True
        for target_name in list(target_dict.keys()):
            if target_name in data_label_dict[data_key]["metaDataAndLabel"].keys():
                label_match = label_match and (
                    data_label_dict[data_key]["metaDataAndLabel"][target_name]
                    in target_dict[target_name]
                )
            else:
                del target_dict[target_name]

        if label_match:
            data = data_label_dict[data_key]["data"]
            legend_marker = []
            for target_name in target_dict.keys():
                if target_name in data_label_dict[data_key]["metaDataAndLabel"].keys():
                    legend_marker.append(
                        target_name
                        + ": "
                        + str(
                            data_label_dict[data_key]["metaDataAndLabel"][target_name]
                        )
                        + " "
                    )
            legend_marker = " ".join(legend_marker)

            if legend_marker not in legend_recorded.keys():
                legend_recorded[legend_marker] = colorBook[len(legend_recorded)]
                data_ax.plot(
                    data,
                    color=legend_recorded[legend_marker],
                    linewidth=1.0,
                    label=legend_marker,
                )
            else:
                data_ax.plot(
                    data,
                    color=legend_recorded[legend_marker],
                    linewidth=1.0,
                )

    # Set fig property.
    title_name = ""
    for target_name in target_dict.keys():
        title_name = (
            title_name + target_name + ": " + str(target_dict[target_name]) + " "
        )

    data_ax.set_title(title_name, fontsize=14)
    data_ax.legend()
    data_ax.set_xlabel("Sampling point", fontsize=14)
    data_ax.set_ylabel("Signal", fontsize=14)
    plt.tight_layout()

    # Save fig.
    if not exists(save_path):
        os.makedirs(save_path)
        print(f"Create {save_path} to store image.png")
    fig.savefig(join(save_path, fig_name), transparent=transparent)

    # Close fig to release memory.
    # RuntimeWarning: More than 20 figures have been opened.
    # Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained
    # until explicitly closed and may consume too much memory.
    # (To control this warning, see the rcParam `figure.max_open_warning`).
    plt.close(fig)


def compare_datasets(data_dict1, data_dict2, save_path, transparent=False):
    compare_label = "Thickness"
    fixed_labels = {"Insulation": 0.0, "Lift-off": 0.0, "WeatherJacket": 0.0}

    def check_matched(x, target_label):
        if x[compare_label] == target_label:
            for data_keys in fixed_labels.keys():
                if x[data_keys] == fixed_labels[data_keys]:
                    continue
                else:
                    return False
            return True
        else:
            return False

    data1_labels = set(
        data_dict1[data_key]["metaDataAndLabel"][compare_label]
        for data_key in data_dict1
    )

    data2_labels = set(
        data_dict2[data_key]["metaDataAndLabel"][compare_label]
        for data_key in data_dict2
    )
    print(f"The first dataset labels are: {data1_labels}")
    print(f"The second dataset labels are: {data2_labels}")

    intersection = list(data1_labels.intersection(data2_labels))
    intersection.sort()
    print(f"The labels intersection is {intersection}")

    for target_label in intersection:
        fig = plt.figure(figsize=[12, 8], dpi=200)
        gs = plt.GridSpec(1, 1, figure=fig)
        data_ax = fig.add_subplot(gs[0, 0])

        for data_key in data_dict1:
            if check_matched(data_dict1[data_key]["metaDataAndLabel"], target_label):
                # if data_dict1[data_key]["metaDataAndLabel"][compare_label] == target_label:
                data_ax.plot(data_dict1[data_key]["data"], color="tab:blue")

        for data_key in data_dict2:
            if check_matched(data_dict2[data_key]["metaDataAndLabel"], target_label):
                # if data_dict2[data_key]["metaDataAndLabel"][compare_label] == target_label:
                data_ax.plot(data_dict2[data_key]["data"], color="tab:orange")

        fig_name = "Comparison" + compare_label + str(target_label).replace(".", "-")
        data_ax.set_title(fig_name, fontsize=14)
        data_ax.set_xlabel("Sampling point", fontsize=14)
        data_ax.set_ylabel("Signal", fontsize=14)
        plt.tight_layout()

        # Save fig.
        if not exists(save_path):
            os.makedirs(save_path)
            print(f"Create {save_path} to store image.png")
        fig.savefig(join(save_path, fig_name), transparent=transparent)

        # Close fig to release memory.
        # RuntimeWarning: More than 20 figures have been opened.
        # Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained
        # until explicitly closed and may consume too much memory.
        # (To control this warning, see the rcParam `figure.max_open_warning`).
        plt.close(fig)


if __name__ == "__main__":
    dataRoot = join("datasets", "PEC")
    jsonSavePath = join(dataRoot, "formatted_v2")
    # Q345_data, aluminum, circle_Q345_05102022, circle_Q345_07102022, Q345_test_09112022, Q345_15112022
    file_name = "Q345_15112022"
    json_file_path = join(jsonSavePath, file_name + ".json")

    # data_label_dict = dataset_v2.load_json(json_file_path)

    (train_dataset, validation_dataset, test_dataset) = dataset_v2.dataPipeline(
        json_file_path,
        splitRate=0.7,
        batchSize=1,
        start_index=0,
        n_samples=256,
        z_norm_flag=True,
        shuffle_flag=False,
    )
    data_label_dict = convert_tfpipeline_to_dict(train_dataset)

    # "Thickness", "Coating", "Insulation", "Loc", "Lift-off"
    target_dict = {
        "Thickness": [15.0, 20.0],  # [5.0, 10.0, 15.0, 20.0]
        "Insulation": [0.0],  # [0.0, 5.0, 10.0, 15.0]
        "WeatherJacket": [0.0],  # [0.0, 3.0]
        "Lift-off": [0.0],  # [0.0, 3.0, 6.0, 9.0]
        "Loc": ["Center"],
    }

    data_visualization(
        data_label_dict,
        save_path="temp",
        fig_name="test",
        target_dict=target_dict,
        transparent=False,
    )

    file_name1 = "Q345_15112022"
    json_file_path1 = join(jsonSavePath, file_name1 + ".json")
    file_name2 = "Q345_25112022"
    json_file_path2 = join(jsonSavePath, file_name2 + ".json")

    # Read dataset as dictionary.
    data_dict1 = dataset_v2.load_json(json_file_path1)
    data_dict2 = dataset_v2.load_json(json_file_path2)
    compare_datasets(data_dict1, data_dict2, "temp", False)
