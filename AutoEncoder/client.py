import flwr as fl
import tensorflow as tf
import numpy as np

import Include.utils_IF as utils

from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model


if __name__ == "__main__":
    x_train, x_test, y_train, y_test = utils.load_dataset()
    #partition_id = np.random.choice(10)
    #x_train, y_train = utils.partition(x_test, y_train, 10)[partition_id]
    #sample_X_train, sample_y_train = utils.oversample_data(X_train, y_train, outliers_fraction)

    x_train = tf.cast(x_train, tf.float32)
    x_test = tf.cast(x_test, tf.float32)

    y_train = y_train.astype('bool').to_numpy()
    y_test = y_test.astype('bool').to_numpy()

    y_train = y_train.ravel()
    y_test = y_test.ravel()

    normal_train_data = x_train[y_train]
    normal_test_data = x_test[y_test]

    anomalous_train_data = x_train[~y_train]
    anomalous_test_data = x_test[~y_test]

    epochs = 20
    random_state = 42
    batch_size = 256
    outliers_fraction = 0.08169

    class AnomalyDetector(Model):
        def __init__(self):
            super(AnomalyDetector, self).__init__()
            self.encoder = tf.keras.Sequential([
                layers.Input(7),
                layers.Dense(7, activation="tanh"),
                layers.Dense(7, activation="tanh"),
                layers.Dense(6, activation="tanh"),
                layers.Dense(4, activation="tanh")
            ])

            self.decoder = tf.keras.Sequential([
                layers.Dense(4, activation="tanh"),
                layers.Dense(6, activation="tanh"),
                layers.Dense(7, activation="sigmoid")
            ])

        def call(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded

    autoencoder = AnomalyDetector()

    autoencoder.compile(
        optimizer='adam',
        loss='mean_squared_logarithmic_error',
    )

    class EncoderClient(fl.client.NumPyClient):
        def get_parameters(self, config):
            return self.get_weights()

        def fit(self, parameters, config):
            self.set_weights(parameters)
            self.fit(
                normal_train_data,
                normal_train_data,
                epochs=epochs,
                batch_size=batch_size,
                verbose=1,
                shuffle=random_state,
                validation_split=0.2,
                validation_data=(x_test, x_test),
            )
            return self.get_weights(), len(x_train), {}

        def evaluate(self, parameters, config):
            self.set_weights(parameters)
            loss, accuracy = self.evaluate(x_test, y_test)
            return loss, len(x_test), {"accuracy": float(accuracy)}

    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=EncoderClient())
