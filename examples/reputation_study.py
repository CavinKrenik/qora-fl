"""
Reputation System Hyperparameter Study
========================================
Tests decay rates and ban thresholds across federated learning rounds with
Byzantine attacks. Measures convergence accuracy, rounds to ban attackers,
and recovery after transient attacks.

Usage::

    pip install qora-fl scikit-learn
    python examples/reputation_study.py
"""

import numpy as np

from qora import ByzantineAggregator

# --- Configuration ---
import argparse
NUM_CLIENTS = 10
NUM_ROUNDS = 20
BYZANTINE_FRACTION = 0.3
LOCAL_EPOCHS = 1
LEARNING_RATE = 0.1
HIDDEN_SIZE = 128
SEED = 42

DECAY_RATES = [0.0, 0.01, 0.03, 0.05, 0.1]
BAN_THRESHOLDS = [0.0, 0.1, 0.2, 0.3, 0.4]


# --- Simple numpy MLP ---
class SimpleMLP:
    """Minimal 2-layer MLP: input_size -> hidden -> 10 (softmax)."""

    def __init__(self, input_size=784, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        self.input_size = input_size
        scale1 = np.sqrt(2.0 / input_size)
        scale2 = np.sqrt(2.0 / HIDDEN_SIZE)
        self.W1 = (rng.standard_normal((input_size, HIDDEN_SIZE)) * scale1).astype(
            np.float32
        )
        self.b1 = np.zeros(HIDDEN_SIZE, dtype=np.float32)
        self.W2 = (rng.standard_normal((HIDDEN_SIZE, 10)) * scale2).astype(np.float32)
        self.b2 = np.zeros(10, dtype=np.float32)

    def get_parameters(self):
        return [self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy()]

    def set_parameters(self, params):
        self.W1, self.b1, self.W2, self.b2 = [p.copy() for p in params]

    def forward(self, X):
        h = np.maximum(0, X @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        return exp_logits / exp_logits.sum(axis=1, keepdims=True)

    def train_one_epoch(self, X, y, lr):
        batch_size = 64
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        for start in range(0, len(X), batch_size):
            batch_idx = indices[start : start + batch_size]
            Xb, yb = X[batch_idx], y[batch_idx]
            h = np.maximum(0, Xb @ self.W1 + self.b1)
            logits = h @ self.W2 + self.b2
            logits -= logits.max(axis=1, keepdims=True)
            probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
            one_hot = np.zeros_like(probs)
            one_hot[np.arange(len(yb)), yb] = 1
            d_logits = (probs - one_hot) / len(yb)
            dW2 = h.T @ d_logits
            db2 = d_logits.sum(axis=0)
            dh = d_logits @ self.W2.T
            dh[h <= 0] = 0
            dW1 = Xb.T @ dh
            db1 = dh.sum(axis=0)
            self.W1 -= lr * dW1
            self.b1 -= lr * db1
            self.W2 -= lr * dW2
            self.b2 -= lr * db2

    def evaluate(self, X, y):
        probs = self.forward(X)
        preds = probs.argmax(axis=1)
        return (preds == y).mean()


# --- Data utilities ---
def load_data():
    """Load digit data: tries MNIST via openml, falls back to sklearn load_digits."""
    try:
        from sklearn.datasets import fetch_openml

        print("Fetching MNIST from OpenML (first run downloads ~50MB)...")
        mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
        X = mnist.data.astype(np.float32) / 255.0
        y = mnist.target.astype(np.int64)
        dataset_name = "MNIST"
    except Exception:
        try:
            from sklearn.datasets import load_digits

            print("Using sklearn load_digits (offline, 8x8 images)...")
            digits = load_digits()
            X = digits.data.astype(np.float32) / 16.0
            y = digits.target.astype(np.int64)
            dataset_name = "Digits"
        except Exception as e:
            print(f"No sklearn available ({e}). Generating synthetic data.")
            rng = np.random.default_rng(SEED)
            X = rng.standard_normal((5000, 64)).astype(np.float32)
            y = rng.integers(0, 10, size=5000).astype(np.int64)
            dataset_name = "Synthetic"

    n = len(X)
    split = int(n * 6 / 7)
    return X[:split], y[:split], X[split:], y[split:], X.shape[1], dataset_name


def partition_data(X, y, n_clients, rng):
    """IID partition across clients."""
    indices = rng.permutation(len(X))
    splits = np.array_split(indices, n_clients)
    return [(X[s], y[s]) for s in splits]


def flip_labels(y):
    """Label-flipping attack: 0<->9, 1<->8, 2<->7, ..."""
    return (9 - y).astype(y.dtype)


def flatten_params(params):
    flat = np.concatenate([p.flatten() for p in params])
    return flat.reshape(1, -1).astype(np.float32)


def unflatten_params(flat, shapes):
    params = []
    offset = 0
    for shape in shapes:
        size = int(np.prod(shape))
        params.append(flat[offset : offset + size].reshape(shape))
        offset += size
    return params


def federated_round(global_params, client_data, aggregator, n_byzantine, input_size):
    """One federated round with label-flipping attack."""
    shapes = [p.shape for p in global_params]
    updates = []
    client_ids = []

    for i, (X_i, y_i) in enumerate(client_data):
        model = SimpleMLP(input_size)
        model.set_parameters(global_params)
        if i < n_byzantine:
            y_i = flip_labels(y_i)
        for _ in range(LOCAL_EPOCHS):
            model.train_one_epoch(X_i, y_i, LEARNING_RATE)
        new_params = model.get_parameters()
        delta = [new - old for new, old in zip(new_params, global_params)]
        updates.append(flatten_params(delta))
        client_ids.append(f"client_{i}")

    aggregated_flat = aggregator.aggregate(updates, client_ids)
    aggregated_delta = unflatten_params(aggregated_flat.flatten(), shapes)
    return [g + d for g, d in zip(global_params, aggregated_delta)]


# --- Studies ---
def study_accuracy_and_banning(client_data, X_test, y_test, input_size):
    """Study 1 & 2: Final accuracy and rounds-to-ban grid."""
    n_byzantine = int(NUM_CLIENTS * BYZANTINE_FRACTION)

    acc_results = {}
    ban_results = {}

    for decay_rate in DECAY_RATES:
        acc_results[decay_rate] = {}
        ban_results[decay_rate] = {}

        for ban_threshold in BAN_THRESHOLDS:
            agg = ByzantineAggregator(
                "trimmed_mean", 0.2, ban_threshold=ban_threshold
            )

            rng = np.random.default_rng(SEED)
            model = SimpleMLP(input_size, rng)
            global_params = model.get_parameters()

            rounds_to_ban = None

            for r in range(NUM_ROUNDS):
                global_params = federated_round(
                    global_params, client_data, agg, n_byzantine, input_size
                )

                # Check if all attackers are banned
                if rounds_to_ban is None and ban_threshold > 0:
                    all_banned = all(
                        agg.get_reputation(f"client_{i}") < ban_threshold
                        for i in range(n_byzantine)
                    )
                    if all_banned:
                        rounds_to_ban = r + 1

                if decay_rate > 0:
                    agg.decay_reputations(decay_rate)

            model.set_parameters(global_params)
            acc = model.evaluate(X_test, y_test)
            acc_results[decay_rate][ban_threshold] = acc
            ban_results[decay_rate][ban_threshold] = rounds_to_ban

    return acc_results, ban_results


def study_transient_recovery(client_data, X_test, y_test, input_size):
    """Study 3: Recovery after 10 attack rounds + 10 clean rounds."""
    n_byzantine = int(NUM_CLIENTS * BYZANTINE_FRACTION)
    ban_threshold = 0.2
    results = {}

    for decay_rate in DECAY_RATES:
        agg = ByzantineAggregator(
            "trimmed_mean", 0.2, ban_threshold=ban_threshold
        )

        rng = np.random.default_rng(SEED)
        model = SimpleMLP(input_size, rng)
        global_params = model.get_parameters()

        # Phase 1: 10 rounds with attack
        for _ in range(10):
            global_params = federated_round(
                global_params, client_data, agg, n_byzantine, input_size
            )
            if decay_rate > 0:
                agg.decay_reputations(decay_rate)

        # Phase 2: 10 rounds clean (no attack)
        for _ in range(10):
            global_params = federated_round(
                global_params, client_data, agg, 0, input_size
            )
            if decay_rate > 0:
                agg.decay_reputations(decay_rate)

        model.set_parameters(global_params)
        recovery_acc = model.evaluate(X_test, y_test)

        # Check reputation of formerly-byzantine clients
        formerly_byz_reps = [
            agg.get_reputation(f"client_{i}") for i in range(n_byzantine)
        ]

        results[decay_rate] = {
            "recovery_accuracy": recovery_acc,
            "avg_rep_formerly_byz": np.mean(formerly_byz_reps),
            "min_rep_formerly_byz": np.min(formerly_byz_reps),
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Qora-FL Reputation Hyperparameter Study")
    parser.add_argument("--quick", action="store_true", help="Run a reduced grid for CI/fast verification")
    args = parser.parse_args()

    global NUM_ROUNDS, DECAY_RATES, BAN_THRESHOLDS
    if args.quick:
        NUM_ROUNDS = 8
        DECAY_RATES = [0.0, 0.03, 0.1]
        BAN_THRESHOLDS = [0.0, 0.2, 0.4]

    rng = np.random.default_rng(SEED)
    X_train, y_train, X_test, y_test, input_size, dataset_name = load_data()
    client_data = partition_data(X_train, y_train, NUM_CLIENTS, rng)

    print(f"Reputation Hyperparameter Study ({dataset_name})")
    print(f"{NUM_CLIENTS} clients, {int(BYZANTINE_FRACTION * 100)}% Byzantine, {NUM_ROUNDS} rounds")
    print()

    # Studies 1 & 2
    acc_results, ban_results = study_accuracy_and_banning(
        client_data, X_test, y_test, input_size
    )

    # Print Study 1: Final Accuracy
    print("Study 1: Final Accuracy (Decay Rate x Ban Threshold)")
    print("=" * 72)
    header = f"{'Decay \\ Ban':<12}"
    for bt in BAN_THRESHOLDS:
        header += f" {bt:>10}"
    print(header)
    print("-" * 72)
    for dr in DECAY_RATES:
        row = f"{dr:<12}"
        for bt in BAN_THRESHOLDS:
            row += f" {acc_results[dr][bt]:>10.4f}"
        print(row)

    print()

    # Print Study 2: Rounds to Ban
    print("Study 2: Rounds to Ban All Attackers")
    print("=" * 72)
    header = f"{'Decay \\ Ban':<12}"
    for bt in BAN_THRESHOLDS:
        header += f" {bt:>10}"
    print(header)
    print("-" * 72)
    for dr in DECAY_RATES:
        row = f"{dr:<12}"
        for bt in BAN_THRESHOLDS:
            val = ban_results[dr][bt]
            row += f" {str(val) if val else 'n/a':>10}"
        print(row)

    print()

    # Study 3: Recovery
    recovery = study_transient_recovery(client_data, X_test, y_test, input_size)

    print("Study 3: Recovery After Transient Attack (10 attack + 10 clean rounds)")
    print("=" * 72)
    print(f"{'Decay Rate':<12} {'Recovery Acc':>14} {'Avg Byz Rep':>14} {'Min Byz Rep':>14}")
    print("-" * 72)
    for dr in DECAY_RATES:
        r = recovery[dr]
        print(
            f"{dr:<12} {r['recovery_accuracy']:>14.4f} "
            f"{r['avg_rep_formerly_byz']:>14.4f} "
            f"{r['min_rep_formerly_byz']:>14.4f}"
        )

    print()
    print("Recommendations:")
    print("  - decay_rate 0.03: good balance of attacker exclusion and recovery")
    print("  - decay_rate 0.05: use in dynamic environments with transient faults")
    print("  - ban_threshold 0.2: effective filtering without false positives")
    print("  - ban_threshold 0.0: no banning; use when reputation is for observation only")
    print("  - See docs/TUNING.md for detailed guidance")


if __name__ == "__main__":
    main()
