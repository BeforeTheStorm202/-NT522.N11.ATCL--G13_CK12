import flwr as fl
from flwr.common import Metrics
from typing import List, Tuple
"""
fl.server.start_server(
    server_address="127.0.0.1:8080",
    config=fl.server.ServerConfig(num_rounds=3)
)
"""
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    
    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

# Create FedAvg strategy
strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0, # Sample 100% of available clients for training
        fraction_evaluate=0.5, # Sample 50% of available clients for evaluation
        min_fit_clients=4,  # Never sample less than 2 clients for training
        min_evaluate_clients=2, # Never sample less than 1 clients for evaluation
        min_available_clients=4,  # Wait until all clients are available
        evaluate_metrics_aggregation_fn=weighted_average,  # <-- pass the metric aggregation function
)

# Start server
fl.server.start_server(
    server_address="127.0.0.1:8080",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy,
)
