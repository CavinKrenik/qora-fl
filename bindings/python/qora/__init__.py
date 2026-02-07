"""Qora-FL: Quorum-Oriented Robust Aggregation for Federated Learning."""

from qora._core import ByzantineAggregator, ReputationManager

__all__ = ["ByzantineAggregator", "ReputationManager"]


def __getattr__(name):
    if name == "QoraStrategy":
        try:
            from qora.strategy import QoraStrategy

            return QoraStrategy
        except ImportError:
            raise ImportError(
                "QoraStrategy requires Flower. "
                "Install with: pip install qora-fl[flower]"
            ) from None
    raise AttributeError(f"module 'qora' has no attribute {name}")
