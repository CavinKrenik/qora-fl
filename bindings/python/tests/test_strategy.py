"""Tests for the Flower-compatible strategy adapter.

These use Flower's real ``FitRes``, ``Status``, and ``Parameters`` types rather
than mocks, so the suite fails if the adapter drifts from the actual interface.
Only the client proxy is a test double: constructing a real networked client
would add nothing, and ``cid`` is the only member the adapter touches.

The question every test here serves: does one Flower client result become
exactly one validated, complete Rust update -- one identity, one sample count,
one aggregation decision, one reputation update?
"""

import numpy as np
import pytest

flwr = pytest.importorskip("flwr", reason="Flower is required for the adapter")

from flwr.common import Code, FitRes, Status, ndarrays_to_parameters  # noqa: E402
from flwr.common import parameters_to_ndarrays  # noqa: E402

from qora import QoraStrategy  # noqa: E402


class FakeClientProxy:
    """Minimal stand-in exposing the one attribute the adapter reads."""

    def __init__(self, cid):
        self.cid = cid


def fit_res(layers, num_examples=1, metrics=None):
    """Build a real Flower FitRes around the given layers."""
    return FitRes(
        status=Status(code=Code.OK, message="ok"),
        parameters=ndarrays_to_parameters(
            [np.asarray(layer, dtype=np.float32) for layer in layers]
        ),
        num_examples=num_examples,
        metrics=metrics or {},
    )


def result(cid, layers, num_examples=1, metrics=None):
    return (FakeClientProxy(cid), fit_res(layers, num_examples, metrics))


def strategy(method="trimmed_mean", **kwargs):
    kwargs.setdefault("trim_fraction", 0.2)
    kwargs.setdefault("reputation_threshold", 0.0)
    return QoraStrategy(aggregation_method=method, **kwargs)


def layers_of(parameters):
    return parameters_to_ndarrays(parameters)


# ===== Failure policy =====


class TestFailurePolicy:
    def test_no_results_returns_none(self):
        strat = strategy()
        assert strat.aggregate_fit(1, [], []) == (None, {})

    def test_failures_refuse_the_round_when_not_accepted(self):
        strat = strategy("fedavg", accept_failures=False)
        results = [result("a", [[1.0]]), result("b", [[3.0]])]

        params, metrics = strat.aggregate_fit(1, results, [RuntimeError("boom")])

        assert params is None
        assert metrics == {}

    def test_failures_are_tolerated_when_accepted(self):
        strat = strategy("fedavg", accept_failures=True)
        results = [result("a", [[1.0]]), result("b", [[3.0]])]

        params, _ = strat.aggregate_fit(1, results, [RuntimeError("boom")])

        assert params is not None
        assert layers_of(params)[0][0] == pytest.approx(2.0)

    def test_refused_round_does_not_touch_reputation(self):
        strat = strategy("fedavg", accept_failures=False)
        results = [result("a", [[1.0]]), result("b", [[500.0]])]

        before = {cid: strat.get_reputation(cid) for cid in ("a", "b")}
        strat.aggregate_fit(1, results, [RuntimeError("boom")])
        after = {cid: strat.get_reputation(cid) for cid in ("a", "b")}

        assert before == after

    def test_refused_round_does_not_invoke_metrics_callback(self):
        calls = []

        def metrics_fn(pairs):
            calls.append(pairs)
            return {"seen": 1}

        strat = strategy(
            "fedavg", accept_failures=False, fit_metrics_aggregation_fn=metrics_fn
        )
        strat.aggregate_fit(1, [result("a", [[1.0]])], [RuntimeError("boom")])
        assert calls == []


# ===== Sample weighting =====


class TestSampleWeighting:
    def test_fedavg_honors_unequal_num_examples(self):
        # An equal average would be 5.0; weighting by 1 and 9 gives 9.0.
        strat = strategy("fedavg")
        results = [
            result("a", [[0.0]], num_examples=1),
            result("b", [[10.0]], num_examples=9),
        ]

        params, _ = strat.aggregate_fit(1, results, [])
        assert layers_of(params)[0][0] == pytest.approx(9.0)

    def test_equal_counts_match_plain_averaging(self):
        strat = strategy("fedavg")
        results = [
            result("a", [[0.0]], num_examples=5),
            result("b", [[10.0]], num_examples=5),
        ]

        params, _ = strat.aggregate_fit(1, results, [])
        assert layers_of(params)[0][0] == pytest.approx(5.0)

    def test_negative_num_examples_is_rejected(self):
        strat = strategy("fedavg")
        results = [result("a", [[1.0]], num_examples=-1)]

        with pytest.raises(ValueError, match="must not be negative"):
            strat.aggregate_fit(1, results, [])

    def test_non_integer_num_examples_is_rejected(self):
        strat = strategy("fedavg")
        results = [result("a", [[1.0]], num_examples=2.5)]

        with pytest.raises(ValueError, match="must be an integer"):
            strat.aggregate_fit(1, results, [])

    def test_all_zero_sample_counts_are_rejected_for_fedavg(self):
        strat = strategy("fedavg")
        results = [
            result("a", [[1.0]], num_examples=0),
            result("b", [[3.0]], num_examples=0),
        ]

        with pytest.raises(ValueError, match="positive total sample count"):
            strat.aggregate_fit(1, results, [])

    def test_robust_methods_stay_client_equal(self):
        # With sample weighting, "b" would dominate and pull the result toward
        # 10. The median must remain one-client-one-vote.
        strat = strategy("median")
        results = [
            result("a", [[0.0]], num_examples=1),
            result("b", [[10.0]], num_examples=999),
            result("c", [[0.0]], num_examples=1),
        ]

        params, _ = strat.aggregate_fit(1, results, [])
        assert layers_of(params)[0][0] == pytest.approx(0.0)

    def test_zero_sample_counts_are_fine_for_robust_methods(self):
        strat = strategy("median")
        results = [
            result("a", [[1.0]], num_examples=0),
            result("b", [[2.0]], num_examples=0),
            result("c", [[3.0]], num_examples=0),
        ]

        params, _ = strat.aggregate_fit(1, results, [])
        assert layers_of(params)[0][0] == pytest.approx(2.0)

    def test_weights_stay_aligned_with_their_clients(self):
        # Reversing which client carries the large count must move the result
        # the other way; if weights drifted, both orderings would agree.
        forward = strategy("fedavg").aggregate_fit(
            1,
            [
                result("a", [[0.0]], num_examples=1),
                result("b", [[10.0]], num_examples=9),
            ],
            [],
        )[0]
        reverse = strategy("fedavg").aggregate_fit(
            1,
            [
                result("a", [[0.0]], num_examples=9),
                result("b", [[10.0]], num_examples=1),
            ],
            [],
        )[0]

        assert layers_of(forward)[0][0] == pytest.approx(9.0)
        assert layers_of(reverse)[0][0] == pytest.approx(1.0)


# ===== Layer structure =====


class TestLayerStructure:
    def test_single_layer_model(self):
        strat = strategy("fedavg")
        results = [result("a", [[[1.0, 2.0]]]), result("b", [[[3.0, 4.0]]])]

        params, _ = strat.aggregate_fit(1, results, [])
        out = layers_of(params)
        assert len(out) == 1
        assert out[0].shape == (1, 2)

    def test_multi_layer_model_restores_every_shape(self):
        shapes = [(3, 4), (4,), (2, 2, 2), ()]
        model = [np.ones(shape, dtype=np.float32) for shape in shapes]

        strat = strategy("fedavg")
        results = [result("a", model), result("b", model)]

        params, _ = strat.aggregate_fit(1, results, [])
        out = layers_of(params)

        assert len(out) == len(shapes)
        for produced, expected in zip(out, shapes):
            assert produced.shape == expected

    def test_mismatched_layer_count_is_rejected(self):
        strat = strategy("fedavg")
        results = [
            result("a", [np.ones((2, 2)), np.ones(2)]),
            result("b", [np.ones((2, 2))]),
        ]

        with pytest.raises(ValueError, match="layers, expected"):
            strat.aggregate_fit(1, results, [])

    def test_mismatched_layer_shape_is_rejected(self):
        strat = strategy("fedavg")
        results = [
            result("a", [np.ones((2, 3))]),
            result("b", [np.ones((3, 2))]),
        ]

        with pytest.raises(ValueError, match="has shape"):
            strat.aggregate_fit(1, results, [])

    def test_same_element_count_but_different_shape_is_rejected(self):
        # (10, 4) and (40,) hold the same 40 values and mean different models.
        strat = strategy("fedavg")
        results = [
            result("a", [np.ones((10, 4))]),
            result("b", [np.ones(40)]),
        ]

        with pytest.raises(ValueError, match="has shape"):
            strat.aggregate_fit(1, results, [])

    def test_error_message_identifies_client_and_layer(self):
        strat = strategy("fedavg")
        results = [
            result("good", [np.ones((2, 2)), np.ones(2)]),
            result("bad", [np.ones((2, 2)), np.ones(3)]),
        ]

        with pytest.raises(ValueError) as excinfo:
            strat.aggregate_fit(1, results, [])

        message = str(excinfo.value)
        assert "bad" in message
        assert "layer 1" in message

    def test_non_contiguous_arrays_are_handled(self):
        # A transposed view is not C-contiguous.
        base = np.arange(12, dtype=np.float32).reshape(3, 4)
        view = base.T
        assert not view.flags["C_CONTIGUOUS"]

        strat = strategy("fedavg")
        results = [result("a", [view]), result("b", [view])]

        params, _ = strat.aggregate_fit(1, results, [])
        out = layers_of(params)
        assert out[0].shape == (4, 3)
        np.testing.assert_allclose(out[0], view, rtol=1e-6)

    def test_model_with_no_parameters_is_rejected(self):
        strat = strategy("fedavg")
        results = [result("a", [np.zeros(0)]), result("b", [np.zeros(0)])]

        with pytest.raises(ValueError, match="no parameters"):
            strat.aggregate_fit(1, results, [])

    def test_output_is_float32(self):
        strat = strategy("fedavg")
        model = [np.ones((2, 2), dtype=np.float64)]
        results = [result("a", model), result("b", model)]

        params, _ = strat.aggregate_fit(1, results, [])
        assert layers_of(params)[0].dtype == np.float32


# ===== Krum coherence: the reason flatten-once matters =====


class TestKrumCoherence:
    """The regression that makes flatten-once necessary.

    Selecting per layer lets Krum choose a different client for each layer and
    return a model no client submitted. These models are arranged so the
    per-layer winner genuinely differs between layers.
    """

    # (layer0, layer1) per client. The most central client differs between the
    # two layers, so independent per-layer selection yields a hybrid that no
    # client submitted -- asserted below rather than assumed. Whole-model
    # selection must instead return some single client's pair.
    MODELS = {
        "c0": (0.0, 100.0),
        "c1": (1.0, 3.0),
        "c2": (2.0, 2.0),
        "c3": (3.0, 1.0),
        "c4": (100.0, 0.0),
    }

    @staticmethod
    def _layers(pair):
        return [
            np.array([pair[0]], dtype=np.float32),
            np.array([pair[1]], dtype=np.float32),
        ]

    def _results(self):
        return [result(cid, self._layers(pair)) for cid, pair in self.MODELS.items()]

    def test_per_layer_selection_would_produce_an_unsubmitted_hybrid(self):
        # Establishes that the scenario discriminates: without this, the test
        # below could pass against a per-layer implementation too.
        from qora import ByzantineAggregator

        per_layer = []
        for layer_index in range(2):
            agg = ByzantineAggregator("krum:1", 0.0)
            updates = [
                np.array([[pair[layer_index]]], dtype=np.float32)
                for pair in self.MODELS.values()
            ]
            per_layer.append(float(agg.aggregate(updates)[0, 0]))

        hybrid = tuple(per_layer)
        assert hybrid not in set(self.MODELS.values()), (
            f"scenario is not discriminating: per-layer selection produced "
            f"{hybrid}, which is a submitted model"
        )

    def test_krum_selects_one_client_for_the_whole_model(self):
        strat = strategy("krum:1")
        params, _ = strat.aggregate_fit(1, self._results(), [])
        layer0, layer1 = layers_of(params)
        produced = (float(layer0[0]), float(layer1[0]))

        assert produced in set(self.MODELS.values()), (
            f"Krum returned {produced}, which no client submitted -- layers "
            f"were selected independently"
        )
        # And it selected from the honest cluster, not either far outlier.
        assert produced not in {self.MODELS["c0"], self.MODELS["c4"]}


# ===== Reputation =====


class TestReputation:
    def test_reputation_reflects_every_layer(self):
        # "sneaky" matches the cohort on layer 0 and deviates wildly on layer
        # 1. Under the old first-layer-only behavior it looked honest.
        honest = [np.ones(4, dtype=np.float32), np.ones(4, dtype=np.float32)]
        sneaky = [np.ones(4, dtype=np.float32), np.full(4, 500.0, dtype=np.float32)]

        strat = strategy("median")
        results = [
            result("a", honest),
            result("b", honest),
            result("c", honest),
            result("sneaky", sneaky),
        ]

        strat.aggregate_fit(1, results, [])

        assert strat.get_reputation("sneaky") < strat.get_reputation(
            "a"
        ), "a client deviating only in a later layer must still be penalized"

    def test_reputation_moves_once_per_client_per_round(self):
        # Multi-layer model: a per-layer implementation would apply the reward
        # once per layer, moving the score four times as far.
        model = [np.ones(2, dtype=np.float32) for _ in range(4)]
        strat = strategy("median")
        results = [result(cid, model) for cid in ("a", "b", "c")]

        strat.aggregate_fit(1, results, [])

        # One reward of 0.02 from the 0.5 default.
        assert strat.get_reputation("a") == pytest.approx(0.52, abs=1e-6)

    def test_rust_owns_gating_and_surfaces_all_rejected(self):
        # Unknown clients default to 0.5, so a 0.99 threshold rejects the whole
        # cohort. The adapter does no filtering of its own, so the Rust error
        # must reach Python rather than being masked by a Python-side fallback.
        strat = strategy("fedavg", reputation_threshold=0.99)
        results = [result("a", [[1.0]]), result("b", [[3.0]])]

        with pytest.raises(ValueError, match="reputation gating rejected all"):
            strat.aggregate_fit(1, results, [])

    def test_client_ids_stay_aligned_after_flattening(self):
        # Only "outlier" should be penalized; if identities drifted during
        # flattening, the penalty would land on the wrong client.
        honest = [np.ones(3, dtype=np.float32), np.ones(3, dtype=np.float32)]
        outlier = [np.full(3, 900.0, dtype=np.float32)] * 2

        strat = strategy("median")
        results = [
            result("a", honest),
            result("b", honest),
            result("c", honest),
            result("outlier", outlier),
        ]

        strat.aggregate_fit(1, results, [])

        assert strat.get_reputation("outlier") < 0.5
        for cid in ("a", "b", "c"):
            assert strat.get_reputation(cid) > 0.5

    def test_decay_applies_after_a_successful_round(self):
        strat = strategy("median", reputation_decay_rate=1.0)
        model = [np.ones(2, dtype=np.float32)]
        results = [result(cid, model) for cid in ("a", "b", "c")]

        strat.aggregate_fit(1, results, [])
        # Rewarded to 0.52, then decayed fully back to the 0.5 default.
        assert strat.get_reputation("a") == pytest.approx(0.5, abs=1e-6)


# ===== Numeric input =====


class TestNumericInput:
    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_non_finite_values_are_rejected(self, bad):
        strat = strategy("fedavg")
        results = [
            result("a", [np.array([1.0, 2.0], dtype=np.float32)]),
            result("b", [np.array([1.0, bad], dtype=np.float32)]),
        ]

        with pytest.raises(ValueError, match="non-finite"):
            strat.aggregate_fit(1, results, [])

    def test_integer_dtype_is_rejected(self):
        # Flower can carry integer arrays; they are not trainable float
        # parameters, so they are refused rather than silently converted.
        strat = QoraStrategy(aggregation_method="fedavg", reputation_threshold=0.0)
        results = [
            (
                FakeClientProxy("a"),
                FitRes(
                    status=Status(code=Code.OK, message="ok"),
                    parameters=ndarrays_to_parameters(
                        [np.array([1, 2], dtype=np.int64)]
                    ),
                    num_examples=1,
                    metrics={},
                ),
            )
        ]

        with pytest.raises(ValueError, match="floating-point dtype"):
            strat.aggregate_fit(1, results, [])

    def test_float64_input_is_accepted_and_converted(self):
        strat = strategy("fedavg")
        model = [np.array([1.0, 2.0], dtype=np.float64)]
        results = [result("a", model), result("b", model)]

        params, _ = strat.aggregate_fit(1, results, [])
        out = layers_of(params)[0]
        assert out.dtype == np.float32
        np.testing.assert_allclose(out, [1.0, 2.0], rtol=1e-6)


# ===== Metrics =====


class TestMetrics:
    def test_callback_receives_counts_and_dicts(self):
        seen = []

        def metrics_fn(pairs):
            seen.append(pairs)
            return {}

        strat = strategy("fedavg", fit_metrics_aggregation_fn=metrics_fn)
        results = [
            result("a", [[1.0]], num_examples=3, metrics={"loss": 0.5}),
            result("b", [[3.0]], num_examples=7, metrics={"loss": 0.1}),
        ]

        strat.aggregate_fit(1, results, [])

        assert seen == [[(3, {"loss": 0.5}), (7, {"loss": 0.1})]]

    def test_callback_return_value_is_passed_through(self):
        strat = strategy(
            "fedavg", fit_metrics_aggregation_fn=lambda pairs: {"loss": 0.25}
        )
        results = [result("a", [[1.0]]), result("b", [[3.0]])]

        _, metrics = strat.aggregate_fit(1, results, [])
        assert metrics == {"loss": 0.25}

    def test_no_callback_returns_empty_metrics(self):
        strat = strategy("fedavg")
        results = [result("a", [[1.0]]), result("b", [[3.0]])]

        _, metrics = strat.aggregate_fit(1, results, [])
        assert metrics == {}


# ===== Interface conformance =====


class TestFlowerInterface:
    def test_strategy_is_a_flower_strategy(self):
        from flwr.server.strategy import Strategy

        assert isinstance(strategy(), Strategy)

    def test_aggregate_fit_matches_the_flower_signature(self):
        import inspect

        from flwr.server.strategy import FedAvg

        expected = list(inspect.signature(FedAvg.aggregate_fit).parameters)
        actual = list(inspect.signature(QoraStrategy.aggregate_fit).parameters)
        assert actual == expected
