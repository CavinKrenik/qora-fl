"""Flower-compatible strategy adapter for Qora-FL.

Connects a Flower federated-learning workflow to the Rust aggregation core.
The adapter is experimental and does not reproduce every behavior of Flower's
built-in ``FedAvg``.

The guarantee it does provide is that one Flower client result becomes exactly
one validated, complete Rust update -- one identity, one sample count, one
aggregation decision, and one reputation update.

Supported behavior
------------------
* Supported Flower range: ``>=1.5,<2``. CI tests both ends of that range
  explicitly -- **1.5.0** (the declared floor) and **1.32.1** -- rather than
  whichever version a resolver happens to pick. Versions between them are
  within the promise but are not individually exercised.
* ``accept_failures`` is honored: reported failures refuse the round unless it
  is set.
* FedAvg weights each update by the result's ``num_examples``.
* Robust methods (median, trimmed mean, Krum, Multi-Krum) treat each accepted
  client update as one participant. ``num_examples`` is *not* reinterpreted as
  an algorithmic weight there -- an attacker claiming a large sample count
  would otherwise gain proportional influence over a median.
* A client's complete model is flattened and aggregated as one update, then
  split back into the original layer shapes.
* Optional norm-bound filtering is off unless ``norm_bound`` is configured.
  When it is, the bound applies to the one flattened update per client, so a
  model is judged whole rather than layer by layer.
* Arrays are aggregated at ``float32`` precision. ``float64`` input is accepted
  and converted; its extra precision is not preserved.
* A malformed model structure rejects the round rather than being dropped.
* Reputation gating and updates are performed once, by the Rust aggregator.
* Metrics are aggregated only through a caller-supplied
  ``fit_metrics_aggregation_fn``.

Usage::

    from qora import QoraStrategy

    strategy = QoraStrategy(
        aggregation_method="trimmed_mean",
        trim_fraction=0.2,
        min_fit_clients=5,
    )
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

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


#: Methods that weight by sample count. Everything else treats one accepted
#: client update as one participant; see the module docstring.
_SAMPLE_WEIGHTED_METHODS = frozenset({"fedavg"})


@dataclass
class _ClientResult:
    """One Flower result, kept whole.

    The fields travel together so that filtering or validation cannot leave
    identities, updates, and sample counts misaligned -- the failure mode of
    building parallel lists and indexing them separately.
    """

    client_id: str
    layers: List[np.ndarray]
    num_examples: int
    metrics: Dict[str, Scalar]


class QoraStrategy(FedAvg):
    """Byzantine-tolerant Flower strategy backed by the Qora-FL Rust core.

    Parameters
    ----------
    aggregation_method : str
        One of ``"trimmed_mean"``, ``"median"``, ``"fedavg"``, ``"krum"``,
        ``"krum:f"``, ``"multi_krum"``, or ``"multi_krum:f:m"``.
        Default ``"trimmed_mean"``.
    trim_fraction : float
        Fraction trimmed from each end (trimmed_mean only). Default 0.2.
    reputation_threshold : float
        Clients below this score are excluded. Enforced by the Rust
        aggregator, not here. Default 0.2.
    reputation_decay_rate : float
        Per-round decay toward the 0.5 default. 0.0 disables. Default 0.0.
    norm_bound : float, optional
        Largest L2 norm a client's *complete flattened model* may have and
        still participate. ``None`` (the default) disables filtering entirely;
        the adapter never enables it on its own. A supplied bound must be
        finite and strictly positive, and is enforced by the Rust aggregator
        against the one flattened update per client -- not per layer, which
        would apply the bound to an arbitrary slice of the model.
    **kwargs
        Passed to ``flwr.server.strategy.FedAvg`` (``fraction_fit``,
        ``min_fit_clients``, ``accept_failures``,
        ``fit_metrics_aggregation_fn``, ...).
    """

    def __init__(
        self,
        aggregation_method: str = "trimmed_mean",
        trim_fraction: float = 0.2,
        reputation_threshold: float = 0.2,
        reputation_decay_rate: float = 0.0,
        norm_bound: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.aggregator = ByzantineAggregator(
            aggregation_method,
            trim_fraction,
            ban_threshold=reputation_threshold,
            norm_bound=norm_bound,
        )
        self.aggregation_method = aggregation_method
        self.reputation_threshold = reputation_threshold
        self.reputation_decay_rate = reputation_decay_rate
        self.norm_bound = norm_bound

    # -- Round entry point -------------------------------------------------

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate one round of client updates.

        Refuses the round -- returning ``(None, {})`` without invoking Rust or
        touching reputation -- when there are no results, or when failures were
        reported and ``accept_failures`` is False.

        Raises
        ------
        ValueError
            If a successful result is malformed. Malformed results are not
            silently discarded: dropping them would change the effective
            adversarial fraction of the round, which is exactly the quantity
            Krum's ``n >= 2f + 3`` condition is stated over.
        """
        if not results:
            return None, {}

        # Failures are refused before anything else so that a refused round
        # cannot aggregate, move reputation, or invoke the metrics callback.
        if failures and not self.accept_failures:
            return None, {}

        records = [
            self._to_record(client_proxy, fit_res) for client_proxy, fit_res in results
        ]

        shapes = self._validate_structure(records)
        flat_updates = [self._flatten(record, shapes) for record in records]
        client_ids = [record.client_id for record in records]
        weights = self._sample_weights(records)

        # One call, one gating decision, one reputation update per client.
        aggregated_flat = self.aggregator.aggregate(flat_updates, client_ids, weights)

        aggregated_layers = self._restore(aggregated_flat, shapes)
        parameters_aggregated = ndarrays_to_parameters(aggregated_layers)

        metrics = self._aggregate_metrics(records)

        if self.reputation_decay_rate > 0:
            self.aggregator.decay_reputations(self.reputation_decay_rate)

        return parameters_aggregated, metrics

    # -- Extraction and validation ----------------------------------------

    @staticmethod
    def _to_record(client_proxy: ClientProxy, fit_res: FitRes) -> _ClientResult:
        """Build one complete record, validating the sample count."""
        client_id = client_proxy.cid
        num_examples = fit_res.num_examples

        # bool is an int subclass but is never a meaningful sample count.
        if isinstance(num_examples, bool) or not isinstance(num_examples, int):
            raise ValueError(
                f"client {client_id}: num_examples must be an integer, "
                f"got {type(num_examples).__name__}"
            )
        if num_examples < 0:
            raise ValueError(
                f"client {client_id}: num_examples must not be negative, "
                f"got {num_examples}"
            )

        return _ClientResult(
            client_id=client_id,
            layers=list(parameters_to_ndarrays(fit_res.parameters)),
            num_examples=num_examples,
            metrics=dict(fit_res.metrics or {}),
        )

    @staticmethod
    def _validate_structure(
        records: Sequence[_ClientResult],
    ) -> List[Tuple[int, ...]]:
        """Check every client against the first as structural reference.

        Returns the reference layer shapes. Two models with the same total
        element count but different layer structure are *not* compatible --
        ``[(10, 4), (4,)]`` and ``[(44,)]`` both hold 44 values and mean
        entirely different things -- so shapes are compared exactly rather than
        relying on NumPy broadcasting to reconcile them.
        """
        reference = records[0]
        if not reference.layers:
            raise ValueError(f"client {reference.client_id}: model has no layers")

        shapes = [np.shape(layer) for layer in reference.layers]

        for record in records:
            if len(record.layers) != len(shapes):
                raise ValueError(
                    f"client {record.client_id}: model has {len(record.layers)} "
                    f"layers, expected {len(shapes)} "
                    f"(reference client {reference.client_id})"
                )

            for index, (layer, expected) in enumerate(zip(record.layers, shapes)):
                array = np.asarray(layer)

                if array.shape != expected:
                    raise ValueError(
                        f"client {record.client_id}: layer {index} has shape "
                        f"{array.shape}, expected {expected} "
                        f"(reference client {reference.client_id})"
                    )
                if array.dtype.kind != "f":
                    raise ValueError(
                        f"client {record.client_id}: layer {index} has dtype "
                        f"{array.dtype}, expected a floating-point dtype "
                        f"(model parameters are aggregated as float32)"
                    )
                if not np.isfinite(array).all():
                    raise ValueError(
                        f"client {record.client_id}: layer {index} contains "
                        f"non-finite values (NaN or infinity)"
                    )

        if sum(int(np.prod(shape)) for shape in shapes) == 0:
            raise ValueError(
                f"client {reference.client_id}: model contains no parameters"
            )

        return shapes

    # -- Flatten / restore -------------------------------------------------

    @staticmethod
    def _flatten(
        record: _ClientResult, shapes: Sequence[Tuple[int, ...]]
    ) -> np.ndarray:
        """Concatenate a client's whole model into one ``(1, P)`` row.

        Aggregating layer by layer would let a selection method choose a
        different client for each layer, producing a model no client actually
        submitted. Flattening first makes Krum select one coherent full-model
        update, Multi-Krum select one coherent cohort, and the reputation
        distance reflect every layer.
        """
        del shapes  # structure already validated; kept for call-site symmetry
        flat = np.concatenate(
            [
                np.ascontiguousarray(layer, dtype=np.float32).reshape(-1)
                for layer in record.layers
            ]
        )
        return flat.reshape(1, -1)

    @staticmethod
    def _restore(
        aggregated: np.ndarray, shapes: Sequence[Tuple[int, ...]]
    ) -> List[np.ndarray]:
        """Split one aggregated row back into the original layer structure."""
        flat = np.asarray(aggregated, dtype=np.float32).reshape(-1)
        sizes = [int(np.prod(shape)) for shape in shapes]
        expected = sum(sizes)

        if flat.size != expected:
            raise ValueError(
                f"aggregated update has {flat.size} values, expected {expected}"
            )

        layers: List[np.ndarray] = []
        offset = 0
        for shape, size in zip(shapes, sizes):
            # copy(): each layer owns its buffer rather than viewing into the
            # shared aggregate, so a caller mutating one cannot affect another.
            layers.append(flat[offset : offset + size].reshape(shape).copy())
            offset += size

        return layers

    # -- Weighting and metrics --------------------------------------------

    def _sample_weights(
        self, records: Sequence[_ClientResult]
    ) -> Optional[List[float]]:
        """Sample-count weights, for FedAvg only.

        Returns None for the robust methods so that ``num_examples`` is never
        quietly reinterpreted as an algorithmic weight; see the module
        docstring for why that would weaken their guarantee.
        """
        method = self.aggregation_method.strip().split(":", 1)[0]
        if method not in _SAMPLE_WEIGHTED_METHODS:
            return None

        weights = [float(record.num_examples) for record in records]
        if sum(weights) <= 0.0:
            raise ValueError(
                "FedAvg requires a positive total sample count, but every "
                "accepted client reported num_examples=0"
            )
        return weights

    def _aggregate_metrics(self, records: Sequence[_ClientResult]) -> Dict[str, Scalar]:
        """Delegate metrics to the caller's aggregation function.

        Returns ``{}`` when none is configured. Metric names carry
        application-specific meaning -- ``loss`` and ``accuracy`` do not
        combine the same way -- so the adapter does not invent an aggregation
        policy of its own.
        """
        if not self.fit_metrics_aggregation_fn:
            return {}

        pairs = [(record.num_examples, record.metrics) for record in records]
        return dict(self.fit_metrics_aggregation_fn(pairs))

    # -- Reputation --------------------------------------------------------

    def get_reputation(self, client_id: str) -> float:
        """Get the reputation score for a specific client."""
        return self.aggregator.get_reputation(client_id)

    def reset_reputations(self):
        """Reset all client reputation scores to default."""
        self.aggregator.reset_reputation()
