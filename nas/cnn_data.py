import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from core.models.data import InputData
from core.repository.task_types import TaskTypesEnum, MachineLearningTasksEnum


def from_json(file_path, task_type: TaskTypesEnum = MachineLearningTasksEnum.classification, train_size=0.75):
    data_frame = pd.read_json(file_path)
    target = data_frame['is_iceberg']
    x_band_1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in data_frame["band_1"]])
    x_band_2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in data_frame["band_2"]])
    x = np.concatenate([x_band_1[:, :, :, np.newaxis], x_band_2[:, :, :, np.newaxis],
                        ((x_band_1 + x_band_2) / 2)[:, :, :, np.newaxis]], axis=-1)
    x_train, x_test, y_train, y_test = train_test_split(x, target, random_state=1, train_size=train_size)
    train_input_data = InputData(idx=np.arange(0, len(x_train)), features=x_train, target=np.array(y_train),
                                 task_type=task_type)
    test_input_data = InputData(idx=np.arange(0, len(x_test)), features=x_test, target=np.array(y_test),
                                task_type=task_type)
    return train_input_data, test_input_data
