"""
Adaptive Trim Evaluation
=========================
Compares static vs adaptive trimmed mean under varying Byzantine fractions.

Demonstrates that adaptive trimming tracks the attack intensity and maintains
better accuracy than a fixed trim fraction when the attack rate is unknown.

Usage::

    pip install qora-fl scikit-learn
    python examples/adaptive_trim_eval.py
"""

import numpy as np

from qora import ByzantineAggregator

# --- Configuration ---
import argparse
NUM_CLIENTS = 20
NUM_ROUNDS = 15
LOCAL_EPOCHS = 1
LEARNING_RATE = 0.1
HIDDEN_SIZE = 128
SEED = 42
BYZANTINE_FRACTIONS = [0.0, 0.1, 0.2, 0.3, 0.4]


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


# --- Federated training ---
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


def run_experiment(trim_fraction, adaptive, byz_fraction, client_data, X_test, y_test, input_size):
    """Run FL training with given config and return final accuracy."""
    n_byzantine = int(NUM_CLIENTS * byz_fraction)
    agg = ByzantineAggregator("trimmed_mean", trim_fraction, adaptive_trim=adaptive)

    rng = np.random.default_rng(SEED)
    model = SimpleMLP(input_size, rng)
    global_params = model.get_parameters()

    for _ in range(NUM_ROUNDS):
        global_params = federated_round(
            global_params, client_data, agg, n_byzantine, input_size
        )

    model.set_parameters(global_params)
    return model.evaluate(X_test, y_test)


def main():
    parser = argparse.ArgumentParser(description="Qora-FL Adaptive Trim Evaluation")
    parser.add_argument("--quick", action="store_true", help="Run a reduced grid for CI/fast verification")
    args = parser.parse_args()

    global NUM_ROUNDS, BYZANTINE_FRACTIONS
    if args.quick:
        NUM_ROUNDS = 5
        BYZANTINE_FRACTIONS = [0.0, 0.2, 0.4]

    rng = np.random.default_rng(SEED)
    X_train, y_train, X_test, y_test, input_size, dataset_name = load_data()
    client_data = partition_data(X_train, y_train, NUM_CLIENTS, rng)

    configs = [
        ("Static 0.1", 0.1, False),
        ("Static 0.2", 0.2, False),
        ("Static 0.3", 0.3, False),
        ("Adaptive", 0.1, True),  # min_trim=0.1, adapts upward
    ]

    print(f"Adaptive Trim Evaluation ({dataset_name})")
    print(f"{NUM_CLIENTS} clients, {NUM_ROUNDS} rounds, label-flipping attack")
    print("=" * 78)
    header = f"{'Byzantine %':<14}"
    for name, _, _ in configs:
        header += f" {name:>14}"
    print(header)
    print("-" * 78)

    for byz_frac in BYZANTINE_FRACTIONS:
        row = f"{int(byz_frac * 100):>10}%    "
        accs = []
        for name, trim, adaptive in configs:
            acc = run_experiment(trim, adaptive, byz_frac, client_data, X_test, y_test, input_size)
            accs.append(acc)
            row += f" {acc:>13.4f}"
        print(row)

    print()
    print("Analysis:")
    print("  - Static 0.1: under-trims at high attack rates, over-trims at 0%")
    print("  - Static 0.3: safe at 30% but wastes honest data at low attack rates")
    print("  - Adaptive: starts conservative, increases trim as attackers are detected")
    print("  - Adaptive is most useful when the attack fraction is unknown or variable")


if __name__ == "__main__":
    main()
