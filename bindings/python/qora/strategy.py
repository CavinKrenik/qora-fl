"""Flower integration for Qora-FL Byzantine-tolerant federated learning.

Provides QoraStrategy, a drop-in replacement for FedAvg that uses
Byzantine-tolerant aggregation (trimmed mean, median) to handle
up to 30% malicious clients.

Usage::

    from qora import QoraStrategy

    strategy = QoraStrategy(
        aggregation_method="trimmed_mean",
        trim_fraction=0.2,
        min_fit_clients=5,
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
    )
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from qora import ByzantineAggregator

try:
    from flwr.common import (
        FitRes,
        Parameters,
        Scalar,
        ndarrays_to_parameters,
        parameters_to_ndarrays,
    )
    from flwr.server.client_proxy import ClientProxy
    from flwr.server.strategy import FedAvg
except ImportError:
    raise ImportError(
        "Flower is required for QoraStrategy. "
        "Install with: pip install qora-fl[flower]"
    ) from None


class QoraStrategy(FedAvg):
    """Byzantine-tolerant federated learning strategy using Qora-FL.

    Replaces FedAvg's naive averaging with robust aggregation to tolerate
    up to 30% malicious clients. Tracks client reputations across rounds
    and filters out low-reputation participants.

    Parameters
    ----------
    aggregation_method : str
        One of "trimmed_mean", "median", "fedavg", "krum", or "krum:N".
        Default "trimmed_mean".
    trim_fraction : float
        Fraction to trim from each end (only for trimmed_mean). Default 0.2.
    reputation_threshold : float
        Clients below this score are excluded from aggregation. Default 0.2.
    reputation_decay_rate : float
        Per-round decay rate toward default (0.5). 0.0 disables decay.
        Typical: 0.01-0.05. Default 0.0.
    **kwargs
        Additional arguments passed to ``flwr.server.strategy.FedAvg``
        (fraction_fit, min_fit_clients, initial_parameters, etc.)
    """

    def __init__(
        self,
        aggregation_method: str = "trimmed_mean",
        trim_fraction: float = 0.2,
        reputation_threshold: float = 0.2,
        reputation_decay_rate: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.aggregator = ByzantineAggregator(
            aggregation_method, trim_fraction, ban_threshold=reputation_threshold
        )
        self.reputation_threshold = reputation_threshold
        self.reputation_decay_rate = reputation_decay_rate

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model updates using Byzantine-tolerant aggregation.

        Overrides FedAvg.aggregate_fit to route updates through the Rust-backed
        Qora-FL aggregator with reputation-based client filtering.
        """
        if not results:
            return None, {}

        # Extract client data
        client_ids = []
        all_ndarrays = []
        for client_proxy, fit_res in results:
            client_ids.append(client_proxy.cid)
            all_ndarrays.append(parameters_to_ndarrays(fit_res.parameters))

        # Filter clients below reputation threshold
        filtered = [
            i
            for i, cid in enumerate(client_ids)
            if self.aggregator.get_reputation(cid) >= self.reputation_threshold
        ]

        # Fall back to all clients if filtering removes everyone
        if not filtered:
            filtered = list(range(len(client_ids)))

        # Aggregate layer-by-layer
        n_layers = len(all_ndarrays[0])
        aggregated_layers = []

        for layer_idx in range(n_layers):
            layer_updates = []
            layer_cids = []
            for i in filtered:
                arr = all_ndarrays[i][layer_idx]
                original_shape = arr.shape
                # Flatten to (1, N) for the Rust aggregator
                arr_2d = arr.reshape(1, -1).astype(np.float32)
                layer_updates.append(arr_2d)
                layer_cids.append(client_ids[i])

            # Pass client_ids only on the first layer to avoid
            # redundant reputation updates per layer
            cids = layer_cids if layer_idx == 0 else None
            aggregated_2d = self.aggregator.aggregate(layer_updates, cids)
            aggregated_layers.append(aggregated_2d.reshape(original_shape))

        parameters_aggregated = ndarrays_to_parameters(aggregated_layers)

        # Build metrics
        metrics: Dict[str, Scalar] = {
            "qora_round": server_round,
            "qora_num_clients": len(filtered),
            "qora_num_filtered": len(results) - len(filtered),
        }
        for cid in client_ids:
            metrics[f"reputation_{cid}"] = self.aggregator.get_reputation(cid)

        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics.update(self.fit_metrics_aggregation_fn(fit_metrics))

        # Apply reputation decay at end of round (if configured)
        if self.reputation_decay_rate > 0:
            self.aggregator.decay_reputations(self.reputation_decay_rate)

        return parameters_aggregated, metrics

    def get_reputation(self, client_id: str) -> float:
        """Get the reputation score for a specific client."""
        return self.aggregator.get_reputation(client_id)

    def reset_reputations(self):
        """Reset all client reputation scores to default."""
        self.aggregator.reset_reputation()
