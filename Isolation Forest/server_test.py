import pandas as pd
import numpy as np
import flwr as fl
import warnings
from typing import Dict

from pyod.models.iforest import IForest

from sklearn.ensemble import IsolationForest

import Include.utils_test as utils


def fit_round(server_round: int) -> Dict:
    """Send round number to client."""
    return {"server_round": server_round}


def get_evaluate_fn(model: IForest):
    """Return an evaluation function for server-side evaluation."""

    _, _, X_test, y_test = utils.load_dataset()

    # rename fl.common.weight to fl.common.NDArrays (https://openbase.com/python/flwr/versions)
    def evaluate(server_round, parameters: fl.common.NDArrays, config):
        utils.set_model_params(model, parameters)
        X_test['scores'] = model.decision_function(X_test)
        X_test['anomaly_score'] = model.predict(X_test)
        accuracy = 100*list(X_test['anomaly_score']).count(-1)/(y_test)
        return {"accuracy": accuracy}

    return evaluate     # keyword: closure function
    '''
    this function will remember all its value from its parent fucntion event after call
    '''


# Start Flower server for five rounds of federated learning
if __name__ == "__main__":
    model = IsolationForest()
    utils.set_initial_params(model)
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=1,
        min_fit_clients=1,
        min_evaluate_clients=1,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_round,
    )
    fl.server.start_server(
        server_address="127.0.0.1:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=3)
    )
