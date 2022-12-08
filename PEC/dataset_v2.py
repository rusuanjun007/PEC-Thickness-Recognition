import os
import glob
from os.path import exists, join
from pickle import GLOBAL
import numpy as np
from typing import Union, List
import json
import re
import scipy
import tensorflow as tf
import itertools

# The GLOBAL_LABELS_NAME will add one-not labels to dataset.
GLOBAL_LABELS_NAME = [
    "Thickness",
    "Coating",
    "Insulation",
    "Loc",
    "Lift-off",
    "WeatherJacket",
]


def get_metadata_and_label(contents: str) -> dict:
    """
    Obtain labels from txt.
    """
    # Check data format.
    if contents.find("start") != -1:
        # Get label and value pairs.
        data_codebook = contents[: contents.find("start")].split("\n")
        data_codebook = list(filter(("").__ne__, data_codebook))

        # Generate metaData dictionary.
        metadata = {}
        for label_with_value in data_codebook:
            label = label_with_value[: label_with_value.find("=")].replace(" ", "")
            value = (
                label_with_value[label_with_value.find("=") :]
                .replace(" ", "")
                .replace("mm", "")
                .replace("=", "")
            )
            # Try to convert lable to float.
            try:
                value = float(value)
            except ValueError:
                pass
            metadata[label] = value
        # print(metadata)
        return metadata
    else:
        print(f"Data format incorrect, can not find 'start' keyword in text file.")
        return {"data format incorrect": -1}


def get_data(contents: str) -> List:
    """
    Obtain numerical data from txt.
    """
    dataList = []
    for dd in re.findall("#.+#\n", contents):
        d_number = int(dd.replace("#", "").replace("\n", ""))
        if d_number > 4500 or d_number < 480:
            continue
        dataList.append(d_number)
    # Check empty data file.
    if len(dataList) > 200:
        return dataList
    else:
        print(f"Data length abnormal {len(dataList)}")
        return [-1]


def load_txt(dataRoot: str, dataName: str) -> Union[dict, None]:
    """
    Read a single txt then return a dictionary with data and label.
    """
    dataPath = join(dataRoot, dataName)
    if exists(dataPath):
        # print(f"Read {dataPath}.")
        with open(join(dataRoot, dataName)) as f:
            contents = f.read()
            metadataAndLabel = get_metadata_and_label(contents)
            metadataAndLabel["fileName"] = dataName
            npData = get_data(contents)
        # print(f"Data shape: {len(npData)}.")
        return {"data": npData, "metaDataAndLabel": metadataAndLabel}
    else:
        print(f"{dataPath} do not exist.")
        assert False


def convert_to_mat(collectedData: dict, mat_file_name: str):
    """
    Convert dictionary to matlab .mat format.
    """
    data_list = []
    for data_key in collectedData.keys():
        data_list.append(collectedData[data_key]["data"])

    data_np = np.concatenate([[data] for data in data_list], axis=0)

    labels_dict = {}
    labels_name = GLOBAL_LABELS_NAME + ["fileName"]

    for ln in labels_name:
        if ln in collectedData[data_key]["metaDataAndLabel"].keys():
            labels_dict[ln] = np.concatenate(
                [
                    [collectedData[data_key]["metaDataAndLabel"][ln]]
                    for data_key in collectedData.keys()
                ],
                axis=0,
            )
    labels_dict["data"] = data_np

    scipy.io.savemat(
        mat_file_name,
        labels_dict,
    )
    print(f"Save data to {mat_file_name}.")


def convert_txt_to_json_and_mat(
    dataRoot: str, jsonSavePath: str, saveJsonName: str
) -> None:
    """
    Convert .txt to saved .json
    """
    print(f"Processing dataset {saveJsonName} ...")

    # Convert txt to dictionary.
    textPaths = glob.glob(join(dataRoot, saveJsonName, "*"))
    collectedData = {}
    update_flag = True
    for textPath in textPaths:
        if len(collectedData) > 0 and update_flag:
            last_data_name = data_name
        data_name = textPath.split("/")[-1]
        temp_dict = load_txt(join(dataRoot, saveJsonName), data_name)
        if len(collectedData) == 0:
            collectedData[data_name] = temp_dict
        else:
            if (
                collectedData[last_data_name]["metaDataAndLabel"].keys()
                != temp_dict["metaDataAndLabel"].keys()
            ):
                print(
                    f"{data_name} is abandoned. Metadata keys do not match with the previous one."
                )
                print(f"{temp_dict['metaDataAndLabel']}\n")
                update_flag = False
            elif len(temp_dict["data"]) == 1:
                print(f"{data_name} is abandoned. Data length abnormal.")
                print(f"{temp_dict['data']}\n")
                update_flag = False
            else:
                collectedData[data_name] = temp_dict
                update_flag = True

    # Add one hot labels.
    labels_name_dict = {}
    for label_name in GLOBAL_LABELS_NAME:
        if (
            label_name
            in collectedData[list(collectedData.keys())[0]]["metaDataAndLabel"].keys()
        ):
            labels_name_dict[label_name] = set(
                [
                    collectedData[k]["metaDataAndLabel"][label_name]
                    for k in collectedData.keys()
                ]
            )
            labels_name_dict[label_name] = list(labels_name_dict[label_name])
            labels_name_dict[label_name].sort()
    for data_key in collectedData.keys():
        for label_name in labels_name_dict.keys():
            collectedData[data_key]["metaDataAndLabel"][label_name + "Label"] = np.eye(
                len(labels_name_dict[label_name]),
                dtype=np.float32,
            )[
                labels_name_dict[label_name].index(
                    collectedData[data_key]["metaDataAndLabel"][label_name]
                )
            ].tolist()

    # Data matrix is regular. Find the minimum length of data and cap all of the rest data.
    data_list = []
    for data_key in collectedData.keys():
        data_list.append(collectedData[data_key]["data"])
    min_len = min([len(data) for data in data_list])
    for data_key in collectedData.keys():
        collectedData[data_key]["data"] = collectedData[data_key]["data"][0:min_len]

    print(f"---------------Dataset summary:--------------")
    print(f"Meta labels are {collectedData[data_key]['metaDataAndLabel'].keys()}")
    print(f"Labels in the dataset are {labels_name_dict}")
    print(f"All data length is clipped to minimum value: {min_len}\n")

    # Save as .json.
    if not exists(jsonSavePath):
        os.makedirs(jsonSavePath)
    with open(join(jsonSavePath, saveJsonName + ".json"), "w") as f:
        json.dump(collectedData, f)
    print(f"Save data to {join(jsonSavePath, saveJsonName+'.json')}.")

    # Save as .mat.
    convert_to_mat(collectedData, join(jsonSavePath, saveJsonName + ".mat"))


def load_json(jsonPath: str) -> dict:
    """
    Load json data.
    """
    if exists(jsonPath):
        print(f"Load data form {jsonPath}")
        with open(jsonPath, "r") as file:
            data = json.load(file)
        return data
    else:
        print(f"Do not find {jsonPath}")


def dataPipeline(
    jsonPath: str,
    splitRate: float,
    batchSize: int,
    start_index: int,
    n_samples: int,
    z_norm_flag: bool = True,
    shuffle_flag: bool = True,
) -> List[tf.data.Dataset]:
    """
    return [TrainDataPipeline, TestDataPipeline]
    """
    # Read dataset as dictionary.
    data_dict = load_json(jsonPath)

    # Delete data of thickness = 15 mm and collected at 14-11-2022 and 15-11-2022.
    n_deleted_data = 0
    for data_key in list(data_dict.keys()):
        if data_dict[data_key]["metaDataAndLabel"]["Thickness"] == 15:
            if "11-14" in data_key or "11-15" in data_key:
                del data_dict[data_key]
                n_deleted_data += 1
    print(f"Delete {n_deleted_data} data of Thickness = 15 mm.")

    # Duplicates data which thickness is 0 mm.
    thickness_zero_dict = {}
    repeats = 4 * 4 * 2 - 1
    for data_key in data_dict.keys():
        if data_dict[data_key]["metaDataAndLabel"]["Thickness"] == 0:
            thickness_zero_dict[data_key] = data_dict[data_key]
    for data_key in thickness_zero_dict.keys():
        for repeat_id in range(repeats):
            data_dict[data_key + str(repeat_id)] = thickness_zero_dict[data_key]

    # Check if z-normed.
    if z_norm_flag:
        # Calculate mean and std of data.
        data_numpy = np.concatenate(
            [np.array(data_dict[data_name]["data"]) for data_name in data_dict.keys()]
        )
        data_mean, data_std = np.mean(data_numpy), np.std(data_numpy)
        print(f"Dataset is z-normed with mean {data_mean}, and std {data_std}")

    else:
        print(f"Dataset is not z-normed.")

    # Data Segmentation with labels.
    file_names = list(data_dict.keys())
    labels_name_list = []
    for label_name in GLOBAL_LABELS_NAME:
        if label_name in data_dict[file_names[0]]["metaDataAndLabel"].keys():
            temp_list = list(
                set(
                    [
                        str(data_dict[k]["metaDataAndLabel"][label_name])
                        for k in data_dict.keys()
                    ]
                )
            )
            temp_list.sort()
            labels_name_list.append(temp_list)
    labels_mapped_data = {}
    for mapped_tuple in list(itertools.product(*labels_name_list)):
        labels_mapped_data["_".join(mapped_tuple)] = []
    for dataName in data_dict.keys():
        temp_key = []
        for label_name in GLOBAL_LABELS_NAME:
            if label_name in data_dict[dataName]["metaDataAndLabel"].keys():
                temp_key.append(
                    str(data_dict[dataName]["metaDataAndLabel"][label_name])
                )
        temp_key = "_".join(temp_key)
        assert temp_key in labels_mapped_data.keys()
        labels_mapped_data[temp_key].append(data_dict[dataName])

    # Clear empty labels.
    for data_key in list(labels_mapped_data.keys()):
        if len(labels_mapped_data[data_key]) == 0:
            del labels_mapped_data[data_key]

    # Split the whole dataset into training, validation and test dataset.
    trainDataset = {"data": []}
    validationDataset = {"data": []}
    testDataset = {"data": []}
    for metaLabelKey in data_dict[file_names[0]]["metaDataAndLabel"].keys():
        trainDataset[metaLabelKey] = []
        validationDataset[metaLabelKey] = []
        testDataset[metaLabelKey] = []
    np.random.seed(666)  # 666, 667
    for fn in labels_mapped_data.keys():
        data_with_same_label = np.array(labels_mapped_data[fn])
        np.random.shuffle(data_with_same_label)

        nData = len(data_with_same_label)
        nTrainData = int(nData * splitRate)
        nValidationData = int(nData * (1 - splitRate) / 2)
        nTestData = nData - nTrainData - nValidationData

        for di in range(nData):
            if di < nTrainData:
                trainDataset["data"].append(data_with_same_label[di]["data"])
            elif di >= nTrainData and di < nTrainData + nValidationData:
                validationDataset["data"].append(data_with_same_label[di]["data"])
            else:
                testDataset["data"].append(data_with_same_label[di]["data"])

            for metaLabelKey in data_dict[file_names[0]]["metaDataAndLabel"].keys():
                if di < nTrainData:
                    trainDataset[metaLabelKey].append(
                        data_with_same_label[di]["metaDataAndLabel"][metaLabelKey]
                    )
                elif di >= nTrainData and di < nTrainData + nValidationData:
                    validationDataset[metaLabelKey].append(
                        data_with_same_label[di]["metaDataAndLabel"][metaLabelKey]
                    )
                else:
                    testDataset[metaLabelKey].append(
                        data_with_same_label[di]["metaDataAndLabel"][metaLabelKey]
                    )

    # Generate tf.data.dataset pipeline.
    result = []
    for dataset in [trainDataset, validationDataset, testDataset]:
        # Only take the data in the range [start_index, start_index + n_samples]
        for ii, dd in enumerate(dataset["data"]):
            dataset["data"][ii] = dd[start_index : start_index + n_samples]

        # z-norm.
        if z_norm_flag:
            dataset["data"] = (
                np.array(dataset["data"], dtype=np.float32) - data_mean
            ) / data_std
        else:
            dataset["data"] = np.array(dataset["data"], dtype=np.float32)

        # Reshape the data with format [Height, width=1, channel=1].
        dataset["data"] = dataset["data"].reshape(dataset["data"].shape + (1, 1))
        if shuffle_flag:
            dataset = (
                tf.data.Dataset.from_tensor_slices(dataset)
                .shuffle(len(file_names), reshuffle_each_iteration=True)
                .batch(
                    batchSize, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE
                )
                .prefetch(tf.data.AUTOTUNE)
            )
        else:
            dataset = (
                tf.data.Dataset.from_tensor_slices(dataset)
                .batch(
                    batchSize, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE
                )
                .prefetch(tf.data.AUTOTUNE)
            )
        result.append(dataset)

    # for r in result:
    #     for i in r.as_numpy_iterator():
    #         print(i.keys(), i["data"].shape, i["data"].dtype)
    #     print("next")
    # print()

    return result


if __name__ == "__main__":
    data_root = join("datasets", "PEC")
    json_save_path = join(data_root, "formatted_v2")

    # aluminum, Q345_data, circle_Q345_05102022, circle_Q345_07102022, Q345_test_09112022, Q345_15112022
    # Q345_24112022 Q345_25112022
    data_name = "Q345_25112022"

    convert_txt_to_json_and_mat(data_root, json_save_path, data_name)

    # data_dict = load_json(join(json_save_path, data_name + ".json"))

    json_file_path = join(json_save_path, data_name + ".json")
    train_dataset, validation_dataset, test_dataset = dataPipeline(
        json_file_path,
        splitRate=0.7,
        batchSize=64,
        start_index=0,
        n_samples=256,
        z_norm_flag=True,
    )

    for data_dict in train_dataset.as_numpy_iterator():
        print(data_dict["data"].shape)
        break
