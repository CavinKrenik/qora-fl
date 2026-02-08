"""
Comprehensive Attack Evaluation
=================================
Tests qora-fl aggregation methods against 5 attack types at varying
Byzantine fractions.

Attack types:
  1. Label flipping  - flip training labels (0<->9, 1<->8, ...)
  2. Gradient scaling - multiply honest gradients by 100x
  3. Sign flipping   - negate honest gradients
  4. Additive noise  - add Gaussian noise (std=10) to gradients
  5. ALIE            - craft updates at mean + z*std of honest distribution

Usage::

    pip install qora-fl scikit-learn
    python examples/attack_evaluation.py
    python examples/attack_evaluation.py --quick   # 5 rounds, faster
"""

import sys
import time

import numpy as np

from qora import ByzantineAggregator

# --- Configuration ---
NUM_CLIENTS = 10
NUM_ROUNDS = 15
LOCAL_EPOCHS = 1
LEARNING_RATE = 0.1
HIDDEN_SIZE = 128
SEED = 42
BYZANTINE_FRACTIONS = [0.1, 0.2, 0.3]

ATTACKS = ["label_flip", "gradient_scaling", "sign_flip", "additive_noise", "alie"]
ATTACK_NAMES = {
    "label_flip": "Label Flip",
    "gradient_scaling": "Grad Scale",
    "sign_flip": "Sign Flip",
    "additive_noise": "Add Noise",
    "alie": "ALIE",
}
METHODS = [
    ("fedavg", "FedAvg"),
    ("trimmed_mean", "TrimmedMean"),
    ("median", "Median"),
    ("krum", "Krum"),
    ("multi_krum", "Multi-Krum"),
]


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


# --- Attack implementations ---

def attack_label_flip(y):
    """Flip labels: 0<->9, 1<->8, 2<->7, ..."""
    return (9 - y).astype(y.dtype)


def attack_gradient_scaling(honest_delta, scale=100.0):
    """Multiply honest gradient by a large scaling factor."""
    return honest_delta * scale


def attack_sign_flip(honest_delta):
    """Negate the honest gradient, pushing model in opposite direction."""
    return -honest_delta


def attack_additive_noise(honest_delta, noise_std=10.0, rng=None):
    """Add large Gaussian noise to the honest gradient."""
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.normal(0, noise_std, size=honest_delta.shape).astype(np.float32)
    return honest_delta + noise


def attack_alie(honest_deltas, z_score=1.5):
    """ALIE: craft update at mean + z*std of honest gradient distribution.

    Reference: Baruch et al., "A Little Is Enough" (NeurIPS 2019)
    """
    stacked = np.stack(honest_deltas, axis=0)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0) + 1e-8  # avoid division issues
    return mean + z_score * std


# --- Federated round with configurable attack ---

def compute_honest_delta(global_params, X_i, y_i, input_size):
    """Compute an honest gradient update for one client."""
    model = SimpleMLP(input_size)
    model.set_parameters(global_params)
    for _ in range(LOCAL_EPOCHS):
        model.train_one_epoch(X_i, y_i, LEARNING_RATE)
    new_params = model.get_parameters()
    delta = [new - old for new, old in zip(new_params, global_params)]
    return flatten_params(delta)


def federated_round_with_attack(
    global_params, client_data, aggregator, n_byzantine, input_size,
    attack_type, rng
):
    """One federated round with configurable attack type."""
    shapes = [p.shape for p in global_params]
    n_clients = len(client_data)

    # Step 1: Compute honest deltas for all clients
    honest_deltas = []
    for i, (X_i, y_i) in enumerate(client_data):
        honest_deltas.append(
            compute_honest_delta(global_params, X_i, y_i, input_size)
        )

    # Step 2: Apply attack to Byzantine clients
    updates = []
    client_ids = []

    for i in range(n_clients):
        if i < n_byzantine:
            if attack_type == "label_flip":
                # Re-train with flipped labels
                X_i, y_i = client_data[i]
                model = SimpleMLP(input_size)
                model.set_parameters(global_params)
                y_flipped = attack_label_flip(y_i)
                for _ in range(LOCAL_EPOCHS):
                    model.train_one_epoch(X_i, y_flipped, LEARNING_RATE)
                new_params = model.get_parameters()
                delta = [new - old for new, old in zip(new_params, global_params)]
                updates.append(flatten_params(delta))
            elif attack_type == "gradient_scaling":
                updates.append(attack_gradient_scaling(honest_deltas[i]))
            elif attack_type == "sign_flip":
                updates.append(attack_sign_flip(honest_deltas[i]))
            elif attack_type == "additive_noise":
                updates.append(attack_additive_noise(honest_deltas[i], rng=rng))
            elif attack_type == "alie":
                # Use only honest clients' deltas for ALIE distribution estimate
                honest_only = honest_deltas[n_byzantine:]
                updates.append(attack_alie(honest_only))
            else:
                raise ValueError(f"Unknown attack type: {attack_type}")
        else:
            updates.append(honest_deltas[i])

        client_ids.append(f"client_{i}")

    aggregated_flat = aggregator.aggregate(updates, client_ids)
    aggregated_delta = unflatten_params(aggregated_flat.flatten(), shapes)
    return [g + d for g, d in zip(global_params, aggregated_delta)]


def make_aggregator(method_str, n_byzantine):
    """Create aggregator with appropriate parameters."""
    if method_str == "krum":
        return ByzantineAggregator(f"krum:{n_byzantine}", 0.3)
    elif method_str == "multi_krum":
        return ByzantineAggregator(f"multi_krum:{n_byzantine}:3", 0.3)
    else:
        return ByzantineAggregator(method_str, 0.3)


def run_single_experiment(
    method_str, attack_type, byz_fraction, client_data, X_test, y_test,
    input_size, num_rounds
):
    """Run one FL experiment, return final accuracy."""
    n_byzantine = int(NUM_CLIENTS * byz_fraction)
    agg = make_aggregator(method_str, n_byzantine)
    attack_rng = np.random.default_rng(SEED + hash(attack_type) % 10000)

    model_rng = np.random.default_rng(SEED)
    model = SimpleMLP(input_size, model_rng)
    global_params = model.get_parameters()

    for _ in range(num_rounds):
        global_params = federated_round_with_attack(
            global_params, client_data, agg, n_byzantine, input_size,
            attack_type, attack_rng
        )

    model.set_parameters(global_params)
    return model.evaluate(X_test, y_test)


def main():
    quick = "--quick" in sys.argv
    num_rounds = 5 if quick else NUM_ROUNDS

    rng = np.random.default_rng(SEED)
    X_train, y_train, X_test, y_test, input_size, dataset_name = load_data()
    client_data = partition_data(X_train, y_train, NUM_CLIENTS, rng)

    method_names = [name for _, name in METHODS]

    print("Qora-FL Comprehensive Attack Evaluation")
    print(f"Dataset: {dataset_name}, {NUM_CLIENTS} clients, {num_rounds} rounds")
    if quick:
        print("  (--quick mode: reduced rounds for fast validation)")
    print(f"Methods: {', '.join(method_names)}")
    print(f"Attacks: {', '.join(ATTACK_NAMES[a] for a in ATTACKS)}")
    print(f"Byzantine: {', '.join(f'{int(f*100)}%' for f in BYZANTINE_FRACTIONS)}")
    print()

    total = len(ATTACKS) * len(BYZANTINE_FRACTIONS) * len(METHODS)
    done = 0
    t0 = time.time()

    # results[attack][byz_frac][method_name] = accuracy
    results = {}

    for attack in ATTACKS:
        results[attack] = {}
        for byz_frac in BYZANTINE_FRACTIONS:
            results[attack][byz_frac] = {}
            for method_str, method_name in METHODS:
                acc = run_single_experiment(
                    method_str, attack, byz_frac, client_data,
                    X_test, y_test, input_size, num_rounds
                )
                results[attack][byz_frac][method_name] = acc
                done += 1
                elapsed = time.time() - t0
                remaining = (elapsed / done) * (total - done) if done > 0 else 0
                print(
                    f"  [{done:>3}/{total}] {ATTACK_NAMES[attack]:>12} "
                    f"{int(byz_frac*100):>3}% {method_name:<12} "
                    f"acc={acc:.4f}  (remaining ~{remaining:.0f}s)",
                    flush=True,
                )

    # --- Results tables ---
    print()
    print("=" * 82)
    print("RESULTS SUMMARY")
    print("=" * 82)

    for attack in ATTACKS:
        print(f"\n--- {ATTACK_NAMES[attack]} Attack ---")
        header = f"{'Byz%':<8}"
        for name in method_names:
            header += f" {name:>12}"
        print(header)
        print("-" * (8 + 13 * len(method_names)))

        for byz_frac in BYZANTINE_FRACTIONS:
            row = f"{int(byz_frac*100):>5}%   "
            for name in method_names:
                acc = results[attack][byz_frac][name]
                row += f" {acc:>12.4f}"
            print(row)

    # --- Cross-attack summary at 30% ---
    print()
    print("=" * 82)
    print("Best Method per Attack at 30% Byzantine")
    print("=" * 82)
    for attack in ATTACKS:
        byz_results = results[attack][0.3]
        # Exclude FedAvg from "best" since it's the baseline
        robust_results = {k: v for k, v in byz_results.items() if k != "FedAvg"}
        best_method = max(robust_results, key=robust_results.get)
        best_acc = robust_results[best_method]
        fedavg_acc = byz_results.get("FedAvg", 0)
        improvement = best_acc - fedavg_acc
        print(
            f"  {ATTACK_NAMES[attack]:>12}: {best_method:<12} "
            f"({best_acc:.4f})  vs FedAvg ({fedavg_acc:.4f})  "
            f"delta={improvement:+.4f}"
        )

    # --- Optional plot ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n_attacks = len(ATTACKS)
        fig, axes = plt.subplots(1, n_attacks, figsize=(4 * n_attacks, 4), sharey=True)
        if n_attacks == 1:
            axes = [axes]

        colors = {
            "FedAvg": "red", "TrimmedMean": "blue", "Median": "green",
            "Krum": "purple", "Multi-Krum": "orange",
        }
        byz_pcts = [int(bf * 100) for bf in BYZANTINE_FRACTIONS]

        for ax, attack in zip(axes, ATTACKS):
            for name in method_names:
                accs = [results[attack][bf][name] for bf in BYZANTINE_FRACTIONS]
                ax.plot(byz_pcts, accs, "o-", label=name,
                        color=colors.get(name, "gray"), markersize=5)
            ax.set_title(ATTACK_NAMES[attack], fontsize=10)
            ax.set_xlabel("Byzantine %")
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            ax.set_xticks(byz_pcts)

        axes[0].set_ylabel("Test Accuracy")
        axes[-1].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        fig.suptitle(
            f"{dataset_name}: Robustness Under Different Attacks ({num_rounds} rounds)",
            fontsize=12,
        )
        fig.tight_layout()
        fig.savefig("attack_evaluation.png", dpi=150, bbox_inches="tight")
        print(f"\nPlot saved to attack_evaluation.png")
    except ImportError:
        print("\nInstall matplotlib to generate comparison plots.")

    elapsed_total = time.time() - t0
    print(f"\nTotal evaluation time: {elapsed_total:.1f}s")


if __name__ == "__main__":
    main()
