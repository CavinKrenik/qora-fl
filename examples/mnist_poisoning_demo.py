"""
MNIST Poisoning Demo: FedAvg vs Qora-FL
========================================
Simulates federated learning on MNIST with 30% malicious clients
performing label-flipping attacks (0<->9, 1<->8, 2<->7, ...).

Demonstrates that FedAvg degrades under attack while Qora-FL's
trimmed mean aggregation maintains model accuracy.

Requirements::

    pip install qora-fl matplotlib scikit-learn

Usage::

    python examples/mnist_poisoning_demo.py
"""

import numpy as np

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from qora import ByzantineAggregator

# --- Configuration ---
NUM_CLIENTS = 10
BYZANTINE_FRACTION = 0.3
NUM_ROUNDS = 20
LOCAL_EPOCHS = 1
LEARNING_RATE = 0.1
HIDDEN_SIZE = 128
SEED = 42


# --- Simple numpy MLP ---
class SimpleMLP:
    """Minimal 2-layer MLP: 784 -> 128 -> 10 (softmax)."""

    def __init__(self, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        scale1 = np.sqrt(2.0 / 784)
        scale2 = np.sqrt(2.0 / HIDDEN_SIZE)
        self.W1 = (rng.standard_normal((784, HIDDEN_SIZE)) * scale1).astype(np.float32)
        self.b1 = np.zeros(HIDDEN_SIZE, dtype=np.float32)
        self.W2 = (rng.standard_normal((HIDDEN_SIZE, 10)) * scale2).astype(np.float32)
        self.b2 = np.zeros(10, dtype=np.float32)

    def get_parameters(self):
        return [self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy()]

    def set_parameters(self, params):
        self.W1, self.b1, self.W2, self.b2 = [p.copy() for p in params]

    def forward(self, X):
        h = np.maximum(0, X @ self.W1 + self.b1)  # ReLU
        logits = h @ self.W2 + self.b2
        # Numerically stable softmax
        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        return exp_logits / exp_logits.sum(axis=1, keepdims=True)

    def train_one_epoch(self, X, y, lr):
        """One epoch of mini-batch SGD with batch size 64."""
        batch_size = 64
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        for start in range(0, len(X), batch_size):
            batch_idx = indices[start : start + batch_size]
            Xb, yb = X[batch_idx], y[batch_idx]

            # Forward
            h = np.maximum(0, Xb @ self.W1 + self.b1)
            logits = h @ self.W2 + self.b2
            logits -= logits.max(axis=1, keepdims=True)
            probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)

            # Cross-entropy gradient
            one_hot = np.zeros_like(probs)
            one_hot[np.arange(len(yb)), yb] = 1
            d_logits = (probs - one_hot) / len(yb)

            # Backward
            dW2 = h.T @ d_logits
            db2 = d_logits.sum(axis=0)
            dh = d_logits @ self.W2.T
            dh[h <= 0] = 0  # ReLU gradient
            dW1 = Xb.T @ dh
            db1 = dh.sum(axis=0)

            # Update
            self.W1 -= lr * dW1
            self.b1 -= lr * db1
            self.W2 -= lr * dW2
            self.b2 -= lr * db2

    def evaluate(self, X, y):
        probs = self.forward(X)
        preds = probs.argmax(axis=1)
        accuracy = (preds == y).mean()
        return accuracy


# --- Data utilities ---
def load_mnist():
    """Load MNIST using sklearn (small download, no torch needed)."""
    try:
        from sklearn.datasets import fetch_openml

        mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
        X = mnist.data.astype(np.float32) / 255.0
        y = mnist.target.astype(np.int64)
    except ImportError:
        print("sklearn not found. Generating synthetic MNIST-like data.")
        rng = np.random.default_rng(SEED)
        X = rng.standard_normal((10000, 784)).astype(np.float32)
        y = rng.integers(0, 10, size=10000).astype(np.int64)

    # Split train/test
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    return X_train, y_train, X_test, y_test


def partition_data(X, y, n_clients, rng):
    """IID partition across clients."""
    indices = rng.permutation(len(X))
    splits = np.array_split(indices, n_clients)
    return [(X[s], y[s]) for s in splits]


def flip_labels(y):
    """Label-flipping attack: 0<->9, 1<->8, 2<->7, ..."""
    return (9 - y).astype(y.dtype)


# --- Federated training ---
def flatten_params(params):
    """Flatten list of parameter arrays into a single (1, N) array."""
    flat = np.concatenate([p.flatten() for p in params])
    return flat.reshape(1, -1).astype(np.float32)


def unflatten_params(flat, shapes):
    """Restore flat array back to list of parameter arrays with original shapes."""
    params = []
    offset = 0
    for shape in shapes:
        size = int(np.prod(shape))
        params.append(flat[offset : offset + size].reshape(shape))
        offset += size
    return params


def federated_round(global_params, client_data, aggregator, n_byzantine):
    """One federated round: distribute -> local train -> aggregate."""
    shapes = [p.shape for p in global_params]
    updates = []
    client_ids = []

    for i, (X_i, y_i) in enumerate(client_data):
        model = SimpleMLP()
        model.set_parameters(global_params)

        # Byzantine clients train on flipped labels
        if i < n_byzantine:
            y_i = flip_labels(y_i)

        for _ in range(LOCAL_EPOCHS):
            model.train_one_epoch(X_i, y_i, LEARNING_RATE)

        # Compute update delta (new - old)
        new_params = model.get_parameters()
        delta = [new - old for new, old in zip(new_params, global_params)]
        updates.append(flatten_params(delta))
        client_ids.append(f"client_{i}")

    # Aggregate deltas
    aggregated_flat = aggregator.aggregate(updates, client_ids)
    aggregated_delta = unflatten_params(aggregated_flat.flatten(), shapes)

    # Apply aggregated update
    new_global = [g + d for g, d in zip(global_params, aggregated_delta)]
    return new_global


def run_experiment(method_name, client_data, X_test, y_test, n_byzantine):
    """Run NUM_ROUNDS of federated training with the given method."""
    print(f"\n--- {method_name.upper()} ---")
    agg = ByzantineAggregator(method_name, 0.3)

    rng = np.random.default_rng(SEED)
    model = SimpleMLP(rng)
    global_params = model.get_parameters()
    accuracies = []

    for r in range(NUM_ROUNDS):
        global_params = federated_round(
            global_params, client_data, agg, n_byzantine
        )
        model.set_parameters(global_params)
        acc = model.evaluate(X_test, y_test)
        accuracies.append(acc)
        print(f"  Round {r + 1:2d}: accuracy = {acc:.4f}")

    return accuracies


def main():
    rng = np.random.default_rng(SEED)
    n_byzantine = int(NUM_CLIENTS * BYZANTINE_FRACTION)

    print("Qora-FL MNIST Poisoning Demo")
    print("=" * 50)
    print(f"Clients: {NUM_CLIENTS} ({n_byzantine} Byzantine, label-flipping)")
    print(f"Rounds: {NUM_ROUNDS}, Local epochs: {LOCAL_EPOCHS}")
    print()

    print("Loading MNIST...")
    X_train, y_train, X_test, y_test = load_mnist()
    client_data = partition_data(X_train, y_train, NUM_CLIENTS, rng)

    results = {}
    for method in ["fedavg", "trimmed_mean"]:
        results[method] = run_experiment(
            method, client_data, X_test, y_test, n_byzantine
        )

    # Summary
    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    for method, accs in results.items():
        print(f"  {method:>15s}: final accuracy = {accs[-1]:.4f}")

    delta = results["trimmed_mean"][-1] - results["fedavg"][-1]
    print(f"\n  Qora-FL advantage: +{delta:.4f}")

    # Plot if matplotlib available
    if HAS_MATPLOTLIB:
        fig, ax = plt.subplots(figsize=(10, 6))
        rounds = list(range(1, NUM_ROUNDS + 1))
        ax.plot(rounds, results["fedavg"], "r--o", label="FedAvg (no defense)", alpha=0.8)
        ax.plot(
            rounds,
            results["trimmed_mean"],
            "b-s",
            label="Qora-FL (trimmed mean)",
            alpha=0.8,
        )
        ax.set_xlabel("Federated Round")
        ax.set_ylabel("Test Accuracy")
        ax.set_title(
            f"MNIST: FedAvg vs Qora-FL under {int(BYZANTINE_FRACTION*100)}% "
            f"Label-Flipping Attack"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        fig.tight_layout()
        fig.savefig("mnist_poisoning_comparison.png", dpi=150)
        print("\nPlot saved to mnist_poisoning_comparison.png")
        plt.show()
    else:
        print("\nInstall matplotlib to generate comparison plot.")


if __name__ == "__main__":
    main()
