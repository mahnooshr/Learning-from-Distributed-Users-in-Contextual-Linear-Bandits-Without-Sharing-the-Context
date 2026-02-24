# Learning from Distributed Users in Contextual Linear Bandits Without Sharing the Context

## Overview

This repository contains the experimental evaluation code for comparing **communication-efficient distributed contextual linear bandit algorithms** against standard baselines.

The work is based on the NeurIPS 2022 paper:
> **"Learning from Distributed Users in Contextual Linear Bandits Without Sharing the Context"**
> by Osama A. Hanna, Lin F. Yang, and Christina Fragouli (UCLA)

## Algorithms Compared

| Algorithm | Communication Cost | Description |
|---|---|---|
| **Algorithm 1** (known context distribution) | 0 bits/context + 1 bit/reward | Reduces multi-context problem to single-context via X*(θ); sends only 1-bit quantised reward |
| **Algorithm 2** (unknown context distribution) | ≈5d bits/context + 1 bit/reward | Quantises context vector and diagonal correction; sends ≈5d+1 bits/round |
| **LinUCB** | 0 (centralised) | Standard LinUCB — no communication constraints (upper-confidence bound) |
| **Thompson Sampling** | 0 (centralised) | Linear Thompson Sampling — no communication constraints (posterior sampling) |
| **Full Context Transmission** | 32·K·d + 32 bits/round | Baseline: sends full-precision context + reward every round |

## Datasets

Experiments run on **three public datasets** (via scikit-learn / OpenML):

| Dataset | Samples | Features (d) | Classes (K) | Character |
|---|---|---|---|---|
| **Covertype** | 581,012 | 54 | 7 | High-dimensional |
| **Pendigits** | 10,992 | 16 | 10 | Medium-d, more arms |
| **Shuttle** | 58,000 | 9 | 7 | Low-dimensional |

Each class label is treated as an "arm". Per round, one feature vector is sampled from each arm's empirical distribution.

## Project Structure

```
├── distributed_bandits.py          # All algorithm implementations + simulation engine
├── run_chapter6_experiments.py     # Chapter 6 figure/table generator (CLI)
├── code.ipynb                      # Interactive Jupyter notebook version
├── requirements.txt                # Python dependencies
├── imagess/                        # Generated figures (per-dataset + cross-dataset)
├── tables/                         # Generated tables (CSV + LaTeX)
└── README.md
```

## Quick Start

### 1. Create virtual environment and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run experiments (generates all figures + tables)

```bash
python run_chapter6_experiments.py --T 10000 --seeds 0 1 2 3 4
```

Options:
- `--T` : Horizon length (default: 10,000)
- `--seeds` : Random seeds for multiple runs (default: 0 1 2 3 4)
- `--noise-std` : Reward noise σ (default: 0.01)
- `--out-images` : Output directory for figures (default: `imagess`)
- `--out-tables` : Output directory for tables (default: `tables`)

### 3. Or use the Jupyter notebook

```bash
jupyter notebook code.ipynb
```
Set the kernel to the `.venv` environment.

## Output

### Figures (per dataset: Covertype, Pendigits, Shuttle)

| Figure | Description |
|---|---|
| Fig 1 | Cumulative regret (± 95% CI) — **regret-only algorithms** |
| Fig 2 | Log-log regret scaling (√T reference) |
| Fig 3 | Smoothed instantaneous regret (convergence speed) |
| Fig 4 | Communication vs regret scatter — **distributed methods only** |
| Fig 5 | Regret per uplink bit (communication efficiency bar chart) |
| Fig 6 | Cumulative uplink bits over time |
| Fig 7 | Normalised regret R_t / √t |
| Fig 8 | Summary bars: (a) regret comparison, (b) bits/round |

### Cross-dataset

- `cross_dataset_regret.png` — Grouped bar chart comparing regret across datasets
- `cross_dataset_bits.png` — Grouped bar chart comparing communication cost
- `cross_dataset_summary.csv` / `.tex` — Summary table

## Key Results

1. **Thompson Sampling** achieves the lowest regret (centralised, no communication constraint).
2. **Algorithm 1** (known distribution) provides extreme communication savings (1 bit/round) with moderate regret increase.
3. **Algorithm 2** (unknown distribution) uses ≈5d bits/round, matching the paper's theoretical prediction.
4. **Full-context transmission** wastes orders-of-magnitude more bandwidth for negligible regret gain over LinUCB.
5. Results are **consistent across all three datasets** with different dimensionalities.

## License

This project is for academic/thesis purposes.
