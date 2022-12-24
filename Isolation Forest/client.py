import pandas as pd
import numpy as np
import flwr as fl
import warnings

from pyod.models.iforest import IForest
from sklearn.ensemble import IsolationForest

from sklearn.metrics import log_loss

import Include.utils_IF as utils

if __name__ == "__main__":
    random_state = 42
    outliers_fraction = 0.057

    X_train, y_train, X_test, y_test = utils.load_dataset()

    partition_id = np.random.choice(10)
    X_train, y_train = utils.partition(X_train, y_train, 10)[partition_id]
    #sample_X_train, sample_y_train = utils.oversample_data(X_train, y_train, outliers_fraction) 

    model = IsolationForest(
        n_estimators=100,
        contamination=outliers_fraction,
        random_state=random_state, 
        warm_start=True
    )

    utils.set_initial_params(model=model)

    class IForestClient(fl.client.NumPyClient):
        def get_parameters(self, config):
            return utils.get_model_parameters(model)

        def fit(self, parameters, config):
            utils.set_model_params(model=model, params=parameters)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)
            print(f"Training finished for round {config['server_round']}")
            return utils.get_model_parameters(model), len(X_train), {}

        def evaluate(self, parameters, config):  # type: ignore
            utils.set_model_params(model, parameters)
            loss = log_loss(y_test, model.predict(X_test))
            accuracy = model.score(X_test, y_test)
            return loss, len(X_test), {"accuracy": accuracy}

    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080", client=IForestClient()
    )