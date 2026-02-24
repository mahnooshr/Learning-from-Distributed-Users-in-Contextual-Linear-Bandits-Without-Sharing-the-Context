"""
Chapter 6 — Experimental Evaluation
Run all 5 algorithms on multiple datasets, produce per-dataset figures and
a cross-dataset comparison table + figure.

Usage:
  .venv/bin/python run_chapter6_experiments.py --T 10000 --seeds 0 1 2 3 4
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class Config:
    T: int
    noise_std: float
    seeds: List[int]
    alg1_theta_grid_size: int
    alg1_mc_samples: int
    linucb_alpha: float
    out_dir_images: str
    out_dir_tables: str


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Chapter 6 multi-dataset experiment.")
    p.add_argument("--T", type=int, default=10_000)
    p.add_argument("--noise-std", type=float, default=0.01)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--alg1-theta-grid-size", type=int, default=150)
    p.add_argument("--alg1-mc-samples", type=int, default=150)
    p.add_argument("--linucb-alpha", type=float, default=1.0)
    p.add_argument("--out-images", type=str, default="imagess")
    p.add_argument("--out-tables", type=str, default="tables")
    a = p.parse_args()
    return Config(
        T=a.T, noise_std=a.noise_std, seeds=list(a.seeds),
        alg1_theta_grid_size=a.alg1_theta_grid_size,
        alg1_mc_samples=a.alg1_mc_samples,
        linucb_alpha=a.linucb_alpha,
        out_dir_images=a.out_images, out_dir_tables=a.out_tables,
    )


# ── Styling constants ──────────────────────────────────────────────
COLORS = {
    "Alg1 (known dist)": "#d62728",
    "Alg2 (unknown dist)": "#ff7f0e",
    "LinUCB": "#1f77b4",
    "Thompson": "#2ca02c",
    "Full context (LinUCB)": "#9467bd",
}
MARKERS = {
    "Alg1 (known dist)": "s",
    "Alg2 (unknown dist)": "^",
    "LinUCB": "o",
    "Thompson": "D",
    "Full context (LinUCB)": "v",
}
REGRET_ALGS = ["Alg1 (known dist)", "Alg2 (unknown dist)", "LinUCB", "Thompson"]
COMM_ALGS = ["Alg1 (known dist)", "Alg2 (unknown dist)", "Full context (LinUCB)"]
ALL_ALGS = ["Alg1 (known dist)", "Alg2 (unknown dist)", "LinUCB", "Thompson", "Full context (LinUCB)"]


# ── Per-dataset figure generation ───────────────────────────────────
def generate_dataset_figures(
    ds_name: str, agg: dict, T: int, d: int, K: int, n_seeds: int,
    out_dir: str, plt,
) -> None:
    """Produce 8 figures for a single dataset, saved with dataset prefix."""
    xs = np.arange(T)
    pfx = ds_name.lower()

    # Fig 1: cumulative regret
    fig, ax = plt.subplots(figsize=(11, 5))
    for name in REGRET_ALGS:
        m = agg["cum_regret"][name]["mean"]
        se = agg["cum_regret"][name]["std"] / np.sqrt(n_seeds)
        ax.plot(xs, m, label=name, color=COLORS[name], linewidth=2)
        ax.fill_between(xs, m - 1.96 * se, m + 1.96 * se, color=COLORS[name], alpha=0.12)
    ax.set_xlabel("Round  $t$"); ax.set_ylabel("Cumulative regret  $R_T$")
    ax.set_title(f"{ds_name} — Cumulative regret  (d={d}, K={K}, T={T})")
    ax.legend(loc="upper left"); ax.grid(True, alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"{pfx}_fig1_cumulative_regret.png")); plt.close(fig)

    # Fig 2: log-log regret
    fig, ax = plt.subplots(figsize=(11, 5))
    for name in REGRET_ALGS:
        m = agg["cum_regret"][name]["mean"]
        ax.plot(xs[1:], m[1:], label=name, color=COLORS[name], linewidth=2)
    ref = np.sqrt(xs[1:].astype(float)) * (agg["cum_regret"]["LinUCB"]["mean"][-1] / np.sqrt(T))
    ax.plot(xs[1:], ref, "k--", alpha=0.4, label=r"$\propto \sqrt{T}$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Round $t$ (log)"); ax.set_ylabel("$R_t$ (log)")
    ax.set_title(f"{ds_name} — Regret scaling (log-log)")
    ax.legend(loc="lower right"); ax.grid(True, alpha=0.25, which="both")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"{pfx}_fig2_regret_loglog.png")); plt.close(fig)

    # Fig 3: instantaneous regret (smoothed)
    W = max(T // 50, 20)
    fig, ax = plt.subplots(figsize=(11, 5))
    for name in REGRET_ALGS:
        ir = agg["inst_regret"][name]["mean"]
        sm = np.convolve(ir, np.ones(W) / W, mode="valid")
        ax.plot(np.arange(W - 1, T), sm, label=name, color=COLORS[name], linewidth=1.8)
    ax.set_xlabel("Round $t$"); ax.set_ylabel("Inst. regret (moving avg)")
    ax.set_title(f"{ds_name} — Per-round regret convergence (window={W})")
    ax.legend(loc="upper right"); ax.grid(True, alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"{pfx}_fig3_instantaneous_regret.png")); plt.close(fig)

    # Fig 4: comm vs regret scatter (distributed only)
    fig, ax = plt.subplots(figsize=(9, 6))
    for name in COMM_ALGS:
        bv = float(agg["cum_bits"][name]["mean"][-1])
        rv = float(agg["cum_regret"][name]["mean"][-1])
        ax.scatter(bv, rv, s=180, marker=MARKERS[name], color=COLORS[name], zorder=5, edgecolors="k", linewidths=0.5)
        ax.annotate(name, (bv, rv), fontsize=10, xytext=(10, 8), textcoords="offset points")
    ax.set_xscale("log"); ax.set_xlabel("Total uplink bits (log)"); ax.set_ylabel("Final $R_T$")
    ax.set_title(f"{ds_name} — Communication–Regret trade-off")
    ax.grid(True, alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"{pfx}_fig4_pareto_comm_regret.png")); plt.close(fig)

    # Fig 5: regret per bit bar
    fig, ax = plt.subplots(figsize=(9, 5))
    rpb = [float(agg["cum_regret"][n]["mean"][-1]) / float(agg["cum_bits"][n]["mean"][-1]) for n in COMM_ALGS]
    bars = ax.bar(COMM_ALGS, rpb, color=[COLORS[n] for n in COMM_ALGS], edgecolor="k", linewidth=0.5)
    ax.set_ylabel("$R_T$ / total bits"); ax.set_title(f"{ds_name} — Communication efficiency")
    ax.grid(True, alpha=0.25, axis="y")
    for bar, val in zip(bars, rpb):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.2e}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"{pfx}_fig5_regret_per_bit.png")); plt.close(fig)

    # Fig 6: cumulative bits
    fig, ax = plt.subplots(figsize=(11, 5))
    for name in COMM_ALGS:
        m = agg["cum_bits"][name]["mean"].astype(float)
        ax.plot(xs, m, label=name, color=COLORS[name], linewidth=2)
    ax.set_xlabel("Round $t$"); ax.set_ylabel("Cumulative uplink bits")
    ax.set_title(f"{ds_name} — Uplink communication over time")
    ax.legend(loc="upper left"); ax.grid(True, alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"{pfx}_fig6_cumulative_bits.png")); plt.close(fig)

    # Fig 7: normalised regret
    fig, ax = plt.subplots(figsize=(11, 5))
    sqrt_t = np.sqrt(np.arange(1, T + 1, dtype=float))
    for name in REGRET_ALGS:
        m = agg["cum_regret"][name]["mean"]
        ax.plot(xs, m / sqrt_t, label=name, color=COLORS[name], linewidth=1.8)
    ax.set_xlabel("Round $t$"); ax.set_ylabel(r"$R_t / \sqrt{t}$")
    ax.set_title(f"{ds_name} — Normalised regret")
    ax.legend(loc="upper right"); ax.grid(True, alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"{pfx}_fig7_normalised_regret.png")); plt.close(fig)

    # Fig 8: summary bars
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    x1 = np.arange(len(REGRET_ALGS))
    vals_r = [float(agg["cum_regret"][n]["mean"][-1]) for n in REGRET_ALGS]
    errs_r = [float(agg["cum_regret"][n]["std"][-1]) for n in REGRET_ALGS]
    ax1.bar(x1, vals_r, yerr=errs_r, capsize=4, color=[COLORS[n] for n in REGRET_ALGS], edgecolor="k", linewidth=0.5)
    ax1.set_xticks(x1); ax1.set_xticklabels([n.replace(" (", "\n(") for n in REGRET_ALGS], fontsize=9)
    ax1.set_ylabel("Final $R_T$"); ax1.set_title("(a) Regret"); ax1.grid(True, alpha=0.25, axis="y")
    x2 = np.arange(len(COMM_ALGS))
    vals_b = [float(agg["cum_bits"][n]["mean"][-1]) / T for n in COMM_ALGS]
    bars2 = ax2.bar(x2, vals_b, color=[COLORS[n] for n in COMM_ALGS], edgecolor="k", linewidth=0.5)
    ax2.set_xticks(x2); ax2.set_xticklabels([n.replace(" (", "\n(") for n in COMM_ALGS], fontsize=9)
    ax2.set_ylabel("Bits/round (log)"); ax2.set_yscale("log")
    ax2.set_title("(b) Communication"); ax2.grid(True, alpha=0.25, axis="y")
    for bar, val in zip(bars2, vals_b):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.15, f"{val:.0f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle(f"{ds_name} — Summary", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"{pfx}_fig8_summary_bars.png"), bbox_inches="tight"); plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────
def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    matplotlib.rcParams.update({
        "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 13,
        "legend.fontsize": 10, "figure.dpi": 120, "savefig.dpi": 200,
    })

    from distributed_bandits import (
        CovertypeContextSampler,
        PendigitsContextSampler,
        ShuttleContextSampler,
        aggregate_runs,
        make_nonnegative_theta,
        run_simulation,
    )

    cfg = parse_args()
    os.makedirs(cfg.out_dir_images, exist_ok=True)
    os.makedirs(cfg.out_dir_tables, exist_ok=True)

    # Define datasets  (name, sampler_class)
    DATASETS: List[Tuple[str, type]] = [
        ("Covertype", CovertypeContextSampler),
        ("Pendigits", PendigitsContextSampler),
        ("Shuttle",   ShuttleContextSampler),
    ]

    # Collect cross-dataset summary rows
    cross_rows: List[dict] = []

    for ds_name, SamplerCls in DATASETS:
        print(f"\n{'='*60}")
        print(f"  Dataset: {ds_name}")
        print(f"{'='*60}")

        runs = []
        sampler = None
        for i, seed in enumerate(cfg.seeds):
            print(f"  seed {seed} ({i+1}/{len(cfg.seeds)}) …", flush=True)
            rng = np.random.default_rng(seed)
            sampler = SamplerCls(rng)
            theta_star = make_nonnegative_theta(sampler.d, rng)
            runs.append(
                run_simulation(
                    T=cfg.T, sampler=sampler, theta_star=theta_star,
                    noise_std=cfg.noise_std, rng=rng,
                    alg1_theta_grid_size=cfg.alg1_theta_grid_size,
                    alg1_mc_samples=cfg.alg1_mc_samples,
                    linucb_alpha=cfg.linucb_alpha,
                )
            )

        assert sampler is not None
        agg = aggregate_runs(runs)
        d, K = sampler.d, sampler.K

        # Per-dataset figures
        generate_dataset_figures(ds_name, agg, cfg.T, d, K, len(cfg.seeds), cfg.out_dir_images, plt)

        # Collect for cross-dataset table
        for alg in ALL_ALGS:
            r_mean = float(agg["cum_regret"][alg]["mean"][-1])
            r_std  = float(agg["cum_regret"][alg]["std"][-1])
            b_total = float(agg["cum_bits"][alg]["mean"][-1])
            cross_rows.append({
                "Dataset": ds_name,
                "d": d,
                "K": K,
                "Algorithm": alg,
                "FinalRegret": r_mean,
                "RegretStd": r_std,
                "BitsPerRound": b_total / cfg.T,
                "TotalBits": b_total,
            })

        print(f"  → Figures saved with prefix '{ds_name.lower()}_'")

    # ── Cross-dataset summary table ─────────────────────────────────
    df = pd.DataFrame(cross_rows)
    df.to_csv(os.path.join(cfg.out_dir_tables, "cross_dataset_summary.csv"), index=False)
    df.to_latex(os.path.join(cfg.out_dir_tables, "cross_dataset_summary.tex"), index=False, float_format="%.2f")

    # ── Cross-dataset comparison figure (regret) ────────────────────
    ds_names = [n for n, _ in DATASETS]
    n_ds = len(ds_names)
    n_algs = len(REGRET_ALGS)
    bar_width = 0.18
    x = np.arange(n_ds)

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, alg in enumerate(REGRET_ALGS):
        vals = []
        errs = []
        for ds_name in ds_names:
            row = [r for r in cross_rows if r["Dataset"] == ds_name and r["Algorithm"] == alg][0]
            vals.append(row["FinalRegret"])
            errs.append(row["RegretStd"])
        offset = (i - n_algs / 2 + 0.5) * bar_width
        ax.bar(x + offset, vals, bar_width, yerr=errs, capsize=3,
               label=alg, color=COLORS[alg], edgecolor="k", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(ds_names, fontsize=11)
    ax.set_ylabel("Final cumulative regret  $R_T$")
    ax.set_title(f"Cross-dataset regret comparison  (T={cfg.T})")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.out_dir_images, "cross_dataset_regret.png"))
    plt.close(fig)

    # ── Cross-dataset comparison figure (bits/round) ────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    n_comm = len(COMM_ALGS)
    for i, alg in enumerate(COMM_ALGS):
        vals = []
        for ds_name in ds_names:
            row = [r for r in cross_rows if r["Dataset"] == ds_name and r["Algorithm"] == alg][0]
            vals.append(row["BitsPerRound"])
        offset = (i - n_comm / 2 + 0.5) * bar_width
        ax.bar(x + offset, vals, bar_width, label=alg, color=COLORS[alg], edgecolor="k", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(ds_names, fontsize=11)
    ax.set_ylabel("Uplink bits per round  (log scale)")
    ax.set_yscale("log")
    ax.set_title(f"Cross-dataset communication cost  (T={cfg.T})")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.out_dir_images, "cross_dataset_bits.png"))
    plt.close(fig)

    # ── Config file ─────────────────────────────────────────────────
    with open(os.path.join(cfg.out_dir_tables, "run_config.txt"), "w", encoding="utf-8") as f:
        f.write(f"T={cfg.T}\n")
        f.write(f"noise_std={cfg.noise_std}\n")
        f.write(f"seeds={cfg.seeds}\n")
        f.write(f"alg1_theta_grid_size={cfg.alg1_theta_grid_size}\n")
        f.write(f"alg1_mc_samples={cfg.alg1_mc_samples}\n")
        f.write(f"linucb_alpha={cfg.linucb_alpha}\n")
        f.write(f"datasets={[n for n,_ in DATASETS]}\n")

    print(f"\n{'='*60}")
    print(f"All done!")
    print(f"  Figures → {cfg.out_dir_images}/")
    print(f"  Tables  → {cfg.out_dir_tables}/")
    print(f"  Datasets: {[n for n,_ in DATASETS]}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
