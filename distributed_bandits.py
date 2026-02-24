from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


def sq_reward_1bit(r: float, rng: np.random.Generator) -> float:
    """1-bit stochastic quantizer for rewards in [0, 1]. Unbiased: E[\hat r | r] = r."""
    r = float(np.clip(r, 0.0, 1.0))
    return 1.0 if rng.random() < r else 0.0


def sq_levels(x: np.ndarray, levels: int, rng: np.random.Generator) -> np.ndarray:
    """
    Stochastic quantizer with integer outputs in {0, 1, ..., levels}.
    Assumes x is in [0, levels] (elementwise). Unbiased: E[\hat x | x] = x.
    """
    x = np.asarray(x, dtype=float)
    x = np.clip(x, 0.0, float(levels))
    lo = np.floor(x)
    hi = np.ceil(x)
    # If x is already integer, lo==hi and prob_hi becomes 0 (safe).
    prob_hi = x - lo
    draw = rng.random(size=x.shape)
    q = np.where(draw < prob_hi, hi, lo)
    return q.astype(np.int64)


def sq_signed_1bit(z: np.ndarray, bound: float, rng: np.random.Generator) -> np.ndarray:
    """
    1-bit stochastic quantizer for values in [-bound, bound] (elementwise).
    Output is either +bound or -bound per coordinate. Unbiased.
    """
    if bound <= 0:
        raise ValueError("bound must be > 0")
    z = np.asarray(z, dtype=float)
    z = np.clip(z, -bound, bound)
    p_pos = (z + bound) / (2.0 * bound)
    draw = rng.random(size=z.shape)
    return np.where(draw < p_pos, bound, -bound)


def log2_Q(d: int) -> float:
    """
    log2 |Q| where Q = { x in N^d : ||x||_1 <= 2d }.
    |Q| = sum_{s=0}^{2d} C(s+d-1, d-1)
    """
    if d <= 0:
        raise ValueError("d must be positive")
    total = 0
    for s in range(0, 2 * d + 1):
        total += math.comb(s + d - 1, d - 1)
    return math.log2(total)


@dataclass
class StepResult:
    chosen_arm: int
    reward: float
    expected_reward: float
    optimal_expected_reward: float
    inst_regret: float
    uplink_bits: int


class CovertypeContextSampler:
    """
    Large dataset sampler based on the Covertype dataset (581k rows, 54 features, 7 classes).

    We treat each class label as an 'arm' a in {0..6}. For each round t we sample one feature vector
    from each arm-specific empirical distribution P_a, forming a K x d context matrix X_t.
    """

    def __init__(self, rng: np.random.Generator):
        from sklearn.datasets import fetch_covtype, fetch_openml
        import time
        from urllib.error import ContentTooShortError, URLError

        last_err: Exception | None = None
        for attempt in range(6):
            try:
                ds = fetch_covtype(as_frame=False)
                last_err = None
                break
            except (ContentTooShortError, URLError, OSError, RuntimeError) as e:
                last_err = e
                # Transient network / partial download errors are common; retry.
                time.sleep(1.0 + 0.75 * attempt)

        if last_err is not None:
            # Fallback: OpenML mirror (often more reliable than figshare in restricted networks).
            try:
                ds = fetch_openml(name="covertype", version=3, as_frame=False, parser="auto")
                last_err = None
            except Exception as e2:  # pragma: no cover
                raise RuntimeError(
                    "Failed to download/load Covertype dataset (fetch_covtype retries + OpenML fallback). "
                    "If your connection is flaky, rerun; otherwise consider pre-downloading the dataset "
                    "or switching to a local dataset."
                ) from e2

        X = ds.data.astype(np.float64, copy=False)  # (n, d)
        y_raw = ds.target
        y = y_raw.astype(np.int64, copy=False) if hasattr(y_raw, "astype") else np.asarray(y_raw, dtype=np.int64)

        # Scale features to [0, 1] per-column, then enforce ||x||_2 <= 1.
        col_max = np.maximum(X.max(axis=0), 1.0)
        X = X / col_max
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / np.maximum(norms, 1.0)

        self.X_by_arm: List[np.ndarray] = []
        for a in range(1, 8):
            self.X_by_arm.append(X[y == a])

        self.K = 7
        self.d = X.shape[1]
        self._rng = rng

    def sample_contexts(self) -> np.ndarray:
        X_t = np.empty((self.K, self.d), dtype=np.float64)
        for a in range(self.K):
            pool = self.X_by_arm[a]
            idx = self._rng.integers(0, pool.shape[0])
            X_t[a] = pool[idx]
        return X_t


class OpenMLContextSampler:
    """
    Generic context sampler from any OpenML multiclass classification dataset.
    Each class label becomes an arm; per round we sample one feature vector per arm.
    """

    def __init__(self, openml_name: str, rng: np.random.Generator, version: int = 1):
        from sklearn.datasets import fetch_openml
        ds = fetch_openml(name=openml_name, version=version, as_frame=False, parser="auto")

        X = ds.data.astype(np.float64, copy=False)
        y_raw = ds.target
        # Normalise labels to consecutive integers starting at 0
        unique_labels = sorted(set(y_raw))
        label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
        y = np.array([label_map[lbl] for lbl in y_raw], dtype=np.int64)

        # Scale features to [0,1] per-column, then enforce ||x||_2 <= 1
        col_max = np.maximum(X.max(axis=0), 1.0)
        X = X / col_max
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / np.maximum(norms, 1.0)

        n_classes = len(unique_labels)
        self.X_by_arm: List[np.ndarray] = []
        for a in range(n_classes):
            subset = X[y == a]
            if len(subset) == 0:
                raise ValueError(f"Class {a} has 0 samples in dataset '{openml_name}'")
            self.X_by_arm.append(subset)

        self.K = n_classes
        self.d = X.shape[1]
        self.name = openml_name
        self._rng = rng

    def sample_contexts(self) -> np.ndarray:
        X_t = np.empty((self.K, self.d), dtype=np.float64)
        for a in range(self.K):
            pool = self.X_by_arm[a]
            idx = self._rng.integers(0, pool.shape[0])
            X_t[a] = pool[idx]
        return X_t


class PendigitsContextSampler(OpenMLContextSampler):
    """Pendigits dataset (10992 samples, d=16, K=10 digit classes)."""
    def __init__(self, rng: np.random.Generator):
        super().__init__("pendigits", rng)


class ShuttleContextSampler(OpenMLContextSampler):
    """Shuttle dataset (58000 samples, d=9, K=7 classes)."""
    def __init__(self, rng: np.random.Generator):
        super().__init__("shuttle", rng)


def make_nonnegative_theta(d: int, rng: np.random.Generator) -> np.ndarray:
    theta = rng.random(d)
    theta = theta / max(1e-12, np.linalg.norm(theta))
    return theta.astype(np.float64)


def draw_reward(x: np.ndarray, theta_star: np.ndarray, noise_std: float, rng: np.random.Generator) -> Tuple[float, float]:
    mean = float(np.clip(x @ theta_star, 0.0, 1.0))
    r = float(np.clip(mean + rng.normal(0.0, noise_std), 0.0, 1.0))
    return r, mean


class LinUCBPolicy:
    """Standard LinUCB for linear contextual bandits (centralized; no uplink constraint)."""

    def __init__(self, d: int, alpha: float = 1.0, lam: float = 1.0):
        self.d = d
        self.alpha = alpha
        self.A = lam * np.eye(d)
        self.b = np.zeros(d)
        self._uplink_bits = 0

    @property
    def uplink_bits_per_round(self) -> int:
        return self._uplink_bits

    def select_arm(self, X_t: np.ndarray) -> int:
        A_inv = np.linalg.inv(self.A)
        theta_hat = A_inv @ self.b
        # Compute UCB for each arm
        mu = X_t @ theta_hat
        bonus = self.alpha * np.sqrt(np.sum((X_t @ A_inv) * X_t, axis=1))
        return int(np.argmax(mu + bonus))

    def update(self, x: np.ndarray, r: float) -> None:
        self.A += np.outer(x, x)
        self.b += r * x


class ThompsonSamplingPolicy:
    """Linear Thompson sampling with Gaussian approximation (centralized; no uplink constraint)."""

    def __init__(self, d: int, noise_std: float = 0.05, lam: float = 1.0):
        self.d = d
        self.noise_std = noise_std
        self.A = lam * np.eye(d)
        self.b = np.zeros(d)
        self._uplink_bits = 0

    @property
    def uplink_bits_per_round(self) -> int:
        return self._uplink_bits

    def select_arm(self, X_t: np.ndarray, rng: np.random.Generator) -> int:
        A_inv = np.linalg.inv(self.A)
        mu = A_inv @ self.b
        cov = (self.noise_std**2) * A_inv
        theta_s = rng.multivariate_normal(mu, cov)
        return int(np.argmax(X_t @ theta_s))

    def update(self, x: np.ndarray, r: float) -> None:
        self.A += np.outer(x, x)
        self.b += r * x


class FullContextTransmissionLinUCB:
    """
    Baseline: agent sends the full context matrix (all arms) + reward each round.
    Central runs LinUCB exactly.
    """

    def __init__(self, d: int, K: int, alpha: float = 1.0, lam: float = 1.0):
        self.d = d
        self.K = K
        self._linucb = LinUCBPolicy(d=d, alpha=alpha, lam=lam)
        self._uplink_bits = 32 * (K * d + 1)  # 32-bit float per coordinate + 32-bit reward

    @property
    def uplink_bits_per_round(self) -> int:
        return self._uplink_bits

    def select_arm(self, X_t: np.ndarray) -> int:
        return self._linucb.select_arm(X_t)

    def update(self, x: np.ndarray, r: float) -> None:
        self._linucb.update(x, r)


class Algorithm2UnknownDist:
    """NeurIPS 2022 Algorithm 2 (unknown context distribution), as per paper pseudocode."""

    def __init__(self, d: int, m: int = 3, reg: float = 1e-3):
        self.d = d
        self.m = int(m)
        self.reg = float(reg)

        # Pseudocode uses V~0=0 and pseudo-inverse; we use small ridge for numerical stability.
        self.V_tilde = self.reg * np.eye(d)
        self.u = np.zeros(d)
        self.theta_hat = np.zeros(d)

        bits_q = int(math.ceil(log2_Q(d)))
        self._uplink_bits = bits_q + 2 * d + 1  # h(X_t) + signs + e2 + reward

    @property
    def uplink_bits_per_round(self) -> int:
        return self._uplink_bits

    def select_arm(self, X_t: np.ndarray) -> int:
        return int(np.argmax(X_t @ self.theta_hat))

    def update(self, x: np.ndarray, r: float, rng: np.random.Generator) -> None:
        # Agent-side quantization
        abs_x = np.abs(x)
        s = np.sign(x)
        s[s == 0] = 1.0

        X_int = sq_levels(self.m * abs_x, levels=self.m, rng=rng)  # integer vector in [0, m]
        X_hat = (s * X_int) / float(self.m)

        diff = (x * x) - (X_hat * X_hat)
        bound = 3.0 / float(self.m)
        e2 = sq_signed_1bit(diff, bound=bound, rng=rng)
        X_hat2 = (X_hat * X_hat) + e2  # unbiased estimate of x^2

        r_hat = sq_reward_1bit(r, rng=rng)

        # Central learner updates
        self.u += r_hat * X_hat

        outer = np.outer(X_hat, X_hat)
        outer_offdiag = outer - np.diag(np.diag(outer))
        self.V_tilde += outer_offdiag + np.diag(X_hat2)

        # theta_hat = V_tilde^{-1} u
        self.theta_hat = np.linalg.solve(self.V_tilde, self.u)


class Algorithm1KnownDist:
    """
    NeurIPS 2022 Algorithm 1 (known context distribution), implemented with a finite discretization
    of Theta to make X = {X*(theta)} explicit.

    - Offline: sample a grid of theta candidates, compute X*(theta) via Monte Carlo.
    - Online: run LinUCB on the finite action set X; send the corresponding theta to the agent.
    - Uplink: 1-bit reward only.
    """

    def __init__(
        self,
        d: int,
        K: int,
        context_pools: List[np.ndarray],
        rng: np.random.Generator,
        theta_grid_size: int = 200,
        mc_samples: int = 200,
        alpha: float = 1.0,
        lam: float = 1.0,
    ):
        self.d = d
        self.K = K
        self._uplink_bits = 1
        self._rng = rng

        self.theta_grid = self._sample_theta_grid(theta_grid_size)
        self.X_actions = self._compute_X_star_actions(context_pools, mc_samples)

        # Subroutine * as LinUCB on the finite set X_actions
        self._lin = LinUCBPolicy(d=d, alpha=alpha, lam=lam)

    @property
    def uplink_bits_per_round(self) -> int:
        return self._uplink_bits

    def _sample_theta_grid(self, n: int) -> np.ndarray:
        thetas = self._rng.normal(size=(n, self.d))
        norms = np.linalg.norm(thetas, axis=1, keepdims=True)
        thetas = thetas / np.maximum(norms, 1e-12)
        return thetas.astype(np.float64)

    def _compute_X_star_actions(self, context_pools: List[np.ndarray], mc_samples: int) -> np.ndarray:
        # Approximate X*(theta) = E[argmax_{a} <x_a, theta>] by Monte Carlo.
        X_actions = np.empty((self.theta_grid.shape[0], self.d), dtype=np.float64)
        for i, theta in enumerate(self.theta_grid):
            best_vectors = []
            for _ in range(mc_samples):
                X_t = []
                for a in range(self.K):
                    pool = context_pools[a]
                    idx = self._rng.integers(0, pool.shape[0])
                    X_t.append(pool[idx])
                X_t = np.stack(X_t, axis=0)
                best_vectors.append(X_t[int(np.argmax(X_t @ theta))])
            X_actions[i] = np.mean(best_vectors, axis=0)
        # Enforce Assumption 1: ||x||_2 <= 1
        norms = np.linalg.norm(X_actions, axis=1, keepdims=True)
        X_actions = X_actions / np.maximum(norms, 1.0)
        return X_actions

    def select_arm(self, X_t: np.ndarray) -> Tuple[int, int]:
        # Central chooses x_t in X via LinUCB over finite X_actions
        A_inv = np.linalg.inv(self._lin.A)
        theta_hat = A_inv @ self._lin.b
        mu = self.X_actions @ theta_hat
        bonus = self._lin.alpha * np.sqrt(np.sum((self.X_actions @ A_inv) * self.X_actions, axis=1))
        idx = int(np.argmax(mu + bonus))

        theta_to_send = self.theta_grid[idx]
        chosen_arm = int(np.argmax(X_t @ theta_to_send))
        return chosen_arm, idx

    def update(self, action_idx: int, r: float, rng: np.random.Generator) -> None:
        r_hat = sq_reward_1bit(r, rng=rng)
        x_action = self.X_actions[action_idx]
        # Update subroutine on (x_action, r_hat)
        self._lin.update(x_action, r_hat)


def run_simulation(
    T: int,
    sampler: CovertypeContextSampler,
    theta_star: np.ndarray,
    noise_std: float,
    rng: np.random.Generator,
    alg1_theta_grid_size: int = 200,
    alg1_mc_samples: int = 200,
    linucb_alpha: float = 1.0,
) -> Dict[str, Dict[str, np.ndarray]]:
    K, d = sampler.K, sampler.d

    # Provide Alg1 the empirical pools as "known distributions"
    context_pools = sampler.X_by_arm
    alg1 = Algorithm1KnownDist(
        d=d,
        K=K,
        context_pools=context_pools,
        rng=rng,
        theta_grid_size=alg1_theta_grid_size,
        mc_samples=alg1_mc_samples,
        alpha=linucb_alpha,
    )
    alg2 = Algorithm2UnknownDist(d=d, m=3)
    lin = LinUCBPolicy(d=d, alpha=linucb_alpha)
    ts = ThompsonSamplingPolicy(d=d, noise_std=max(1e-6, noise_std))
    full = FullContextTransmissionLinUCB(d=d, K=K, alpha=linucb_alpha)

    algs = ["Alg1 (known dist)", "Alg2 (unknown dist)", "LinUCB", "Thompson", "Full context (LinUCB)"]
    cum_regret = {k: np.zeros(T) for k in algs}
    cum_bits = {k: np.zeros(T, dtype=np.int64) for k in algs}
    inst_regret = {k: np.zeros(T) for k in algs}
    inst_bits = {k: np.zeros(T, dtype=np.int64) for k in algs}

    reg_running = {k: 0.0 for k in algs}
    bits_running = {k: 0 for k in algs}

    for t in range(T):
        X_t = sampler.sample_contexts()
        exp_rewards = X_t @ theta_star
        opt_exp = float(np.max(exp_rewards))

        # --- Alg1
        a1, a1_idx = alg1.select_arm(X_t)
        r1, exp1 = draw_reward(X_t[a1], theta_star, noise_std, rng)
        alg1.update(a1_idx, r1, rng)
        ir1 = float(opt_exp - exp_rewards[a1])
        reg_running["Alg1 (known dist)"] += ir1
        bits_running["Alg1 (known dist)"] += alg1.uplink_bits_per_round
        cum_regret["Alg1 (known dist)"][t] = reg_running["Alg1 (known dist)"]
        cum_bits["Alg1 (known dist)"][t] = bits_running["Alg1 (known dist)"]
        inst_regret["Alg1 (known dist)"][t] = ir1
        inst_bits["Alg1 (known dist)"][t] = alg1.uplink_bits_per_round

        # --- Alg2
        a2 = alg2.select_arm(X_t)
        r2, exp2 = draw_reward(X_t[a2], theta_star, noise_std, rng)
        alg2.update(X_t[a2], r2, rng)
        ir2 = float(opt_exp - exp_rewards[a2])
        reg_running["Alg2 (unknown dist)"] += ir2
        bits_running["Alg2 (unknown dist)"] += alg2.uplink_bits_per_round
        cum_regret["Alg2 (unknown dist)"][t] = reg_running["Alg2 (unknown dist)"]
        cum_bits["Alg2 (unknown dist)"][t] = bits_running["Alg2 (unknown dist)"]
        inst_regret["Alg2 (unknown dist)"][t] = ir2
        inst_bits["Alg2 (unknown dist)"][t] = alg2.uplink_bits_per_round

        # --- LinUCB
        a3 = lin.select_arm(X_t)
        r3, exp3 = draw_reward(X_t[a3], theta_star, noise_std, rng)
        lin.update(X_t[a3], r3)
        ir3 = float(opt_exp - exp_rewards[a3])
        reg_running["LinUCB"] += ir3
        bits_running["LinUCB"] += lin.uplink_bits_per_round
        cum_regret["LinUCB"][t] = reg_running["LinUCB"]
        cum_bits["LinUCB"][t] = bits_running["LinUCB"]
        inst_regret["LinUCB"][t] = ir3
        inst_bits["LinUCB"][t] = lin.uplink_bits_per_round

        # --- Thompson
        a4 = ts.select_arm(X_t, rng)
        r4, exp4 = draw_reward(X_t[a4], theta_star, noise_std, rng)
        ts.update(X_t[a4], r4)
        ir4 = float(opt_exp - exp_rewards[a4])
        reg_running["Thompson"] += ir4
        bits_running["Thompson"] += ts.uplink_bits_per_round
        cum_regret["Thompson"][t] = reg_running["Thompson"]
        cum_bits["Thompson"][t] = bits_running["Thompson"]
        inst_regret["Thompson"][t] = ir4
        inst_bits["Thompson"][t] = ts.uplink_bits_per_round

        # --- Full context
        a5 = full.select_arm(X_t)
        r5, exp5 = draw_reward(X_t[a5], theta_star, noise_std, rng)
        full.update(X_t[a5], r5)
        ir5 = float(opt_exp - exp_rewards[a5])
        reg_running["Full context (LinUCB)"] += ir5
        bits_running["Full context (LinUCB)"] += full.uplink_bits_per_round
        cum_regret["Full context (LinUCB)"][t] = reg_running["Full context (LinUCB)"]
        cum_bits["Full context (LinUCB)"][t] = bits_running["Full context (LinUCB)"]
        inst_regret["Full context (LinUCB)"][t] = ir5
        inst_bits["Full context (LinUCB)"][t] = full.uplink_bits_per_round

    return {"cum_regret": cum_regret, "cum_bits": cum_bits,
            "inst_regret": inst_regret, "inst_bits": inst_bits}


def aggregate_runs(runs: List[Dict[str, Dict[str, np.ndarray]]]) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    keys = list(runs[0]["cum_regret"].keys())
    metrics = [m for m in ["cum_regret", "cum_bits", "inst_regret", "inst_bits"] if m in runs[0]]
    out: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {m: {} for m in metrics}
    for metric in metrics:
        for k in keys:
            arr = np.stack([r[metric][k] for r in runs], axis=0)  # (n_runs, T)
            mean = arr.mean(axis=0)
            std = arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(mean)
            out[metric][k] = {"mean": mean, "std": std}
    return out

