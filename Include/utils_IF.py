from pyod.models.iforest import IForest

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
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

def set_initial_params(model: IsolationForest):
    model.max_samples = 100
    model.estimators_ = np.array([])


def set_model_params(
    model: IsolationForest, params
) -> IsolationForest:
    model.estimators_ = params[0]
    return model

def get_model_parameters(model: IsolationForest):
    return [model.estimators_]