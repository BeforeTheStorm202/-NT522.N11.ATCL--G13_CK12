import flwr as fl
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense

import Include.utils_AE as utils

epochs = 1
random_state = 42
batch_size = 256
outliers_fraction = 0.057

class AnomalyDetector(tf.keras.Model):
    def __init__(self):
        super(AnomalyDetector, self).__init__()
        self.encoder = tf.keras.Sequential()
        self.encoder.add(tf.keras.Input(7))
        self.encoder.add(tf.keras.layers.Dense(7, activation="tanh"))
        self.encoder.add(tf.keras.layers.Dense(7, activation="tanh"))
        self.encoder.add(tf.keras.layers.Dense(6, activation="tanh"))
        self.encoder.add(tf.keras.layers.Dense(4, activation="tanh"))
        self.encoder.add(tf.keras.layers.Dense(6, activation="tanh"))
        self.encoder.add(tf.keras.layers.Dense(7, activation="sigmoid"))

    def call(self, x):
        encoded = self.encoder(x)
        return encoded

autoencoder = AnomalyDetector()

autoencoder.compile(
    optimizer='adam',
    loss='mean_squared_logarithmic_error',
    metrics=['accuracy']
)

x_train, x_test, y_train, y_test = utils.load_dataset()
partition_id = np.random.choice(10)
x_train, y_train = utils.partition(x_train, y_train, 10)[partition_id]

x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)

y_train = y_train.astype('bool').to_numpy().ravel()
y_test = y_test.astype('bool').to_numpy().ravel()

normal_train_data = x_train[~y_train]
normal_test_data = x_test[~y_test]

anomalous_train_data = x_train[y_train]
anomalous_test_data = x_test[y_test]

x_test, y_test = utils.oversample_data(x_test, y_test, outliers_fraction)

class EncoderClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return autoencoder.weights

    def fit(self, parameters, config):
        autoencoder.set_weights(parameters)
        autoencoder.fit(
            normal_train_data,
            normal_train_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            shuffle=random_state,
            validation_data=(x_test, x_test),
        )
        return autoencoder.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        autoencoder.set_weights(parameters)
        loss, accuracy = autoencoder.evaluate(x_test, y_test)
        return loss, len(x_test), {"accuracy": float(accuracy)}


fl.client.start_numpy_client(
    server_address="127.0.0.1:8080", client=EncoderClient()
)
