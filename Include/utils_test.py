from pyod.models.iforest import IForest

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
from sklearn.tree import ExtraTreeRegressor

import numpy as np
import pandas as pd
import gc

def oversample_data(X, y, oversample_fraction):
    """
    Hàm trả về tập dữ liệu sau khi được oversamle giữa minority và majority class
    """
    oversample = SMOTE(sampling_strategy=oversample_fraction)
    return oversample.fit_resample(X, y)

def under_over_sampleData(X, y):
    """
    Hàm trả về tập dữ liệu sau khi được oversamle và undersample
    giữa minority và majority class
    """
    over = SMOTE(sampling_strategy=0.2)
    under = RandomUnderSampler(sampling_strategy=0.5)
    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)
    return pipeline.fit_resample(X, y)

def save_memory(var):
    del var
    gc.collect()

def partition(X: np.ndarray, y: np.ndarray, num_partitions: int):
    """Split X and y into a number of partitions."""
    return list(
        zip(np.array_split(X, num_partitions), np.array_split(y, num_partitions))
    )

def load_dataset():
    dataset = pd.read_csv("data/clean_data.csv")
    dataset_check = pd.read_csv("data/check_data.csv")

    X_train, y_train, X_test, y_test = train_test_split(
        dataset, dataset_check, test_size=0.3, random_state=10
    )
    return X_train, y_train, X_test, y_test

def set_initial_params(model: IForest):
    max_sample = 200
    model.decision_scores_ = np.array([i for i in range(max_sample)])
    model.labels_ = np.array([i for i in range(max_sample)])

def set_model_params(
    model: IForest, params
) -> IForest:
    model.decision_scores_ = params[0]
    model.labels_ = params[1]
    return model

def get_model_parameters(model: IForest):
    return [model.decision_function, model.labels_]