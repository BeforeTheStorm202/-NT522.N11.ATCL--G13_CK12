from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split

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
    dataset = pd.read_csv("data/x_1.csv")
    dataset_check = pd.read_csv("data/y_1.csv")

    X_train, y_train, X_test, y_test = train_test_split(
        dataset, dataset_check, test_size=0.2, random_state=20
    )
    return X_train, y_train, X_test, y_test
