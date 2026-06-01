#!/usr/bin/env python
"""Generate figures from the experiment and bootstrap outputs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from options_diffusion.config import RESULTS_DIR


SHORT_NAME = {
    "solo_diff": "Solo Diff",
    "solo_nw": "NW",
    "tc_diff": "TC Diff",
    "t_copula": "t-Copula",
}
COLORS = {"solo_diff": "#4c72b0",
           "solo_nw": "#55a868",
          "tc_diff": "#c44e52", "t_copula": "#8172b2"}
PLOT_METRICS = ["SWD", "MMD2", "Frechet", "Energy", "CorrDist"]


def _ensure_out(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)


def plot_cross_seed(experiment_dir: Path, out_dir: Path, exclude: list[str] | None = None):
    """Bar chart of mean +/- std across seeds, pooled over tickers."""
    parquet = experiment_dir / "summary_pooled.parquet"
    if not parquet.exists():
        print(f"  skip cross-seed: {parquet} not found")
        return
    df = pd.read_parquet(parquet)
    if exclude:
        df = df[~df["method"].isin(exclude)]
    methods = sorted(df["method"].unique())

    fig, axes = plt.subplots(1, len(PLOT_METRICS), figsize=(3.5 * len(PLOT_METRICS), 4))
    for ax, metric in zip(axes, PLOT_METRICS):
        sub = df[df["metric"] == metric].set_index("method").reindex(methods)
        ax.bar(
            [SHORT_NAME.get(m, m) for m in methods],
            sub["mean"].values,
            yerr=sub["std"].values,
            color=[COLORS.get(m, "gray") for m in methods],
            capsize=4,
        )
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=30)
    plt.suptitle("Cross-seed metrics (pooled across tickers, +/- 1 std)", y=1.02)
    plt.tight_layout()
    out = out_dir / "cross_seed_pooled.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  wrote {out}")


def plot_per_ticker(experiment_dir: Path, out_dir: Path, exclude: list[str] | None = None):
    """One bar chart per ticker, all metrics shown."""
    parquet = experiment_dir / "summary_per_ticker.parquet"
    if not parquet.exists():
        print(f"  skip per-ticker: {parquet} not found")
        return
    df = pd.read_parquet(parquet)
    if exclude:
        df = df[~df["method"].isin(exclude)]
    tickers = sorted(df["ticker"].unique())
    methods = sorted(df["method"].unique())

    for ticker in tickers:
        sub = df[df["ticker"] == ticker]
        fig, axes = plt.subplots(1, len(PLOT_METRICS), figsize=(3.5 * len(PLOT_METRICS), 4))
        for ax, metric in zip(axes, PLOT_METRICS):
            r = sub[sub["metric"] == metric].set_index("method").reindex(methods)
            ax.bar(
                [SHORT_NAME.get(m, m) for m in methods],
                r["mean"].values,
                yerr=r["std"].values,
                color=[COLORS.get(m, "gray") for m in methods],
                capsize=4,
            )
            ax.set_title(metric)
            ax.tick_params(axis="x", rotation=30)
        plt.suptitle(f"{ticker} - cross-seed metrics", y=1.02)
        plt.tight_layout()
        out = out_dir / f"per_ticker_{ticker}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
    print(f"  wrote {len(tickers)} per-ticker plots in {out_dir}")


def plot_bootstrap(bootstrap_dir: Path, out_dir: Path, exclude: list[str] | None = None):
    """Histograms of paired metric differences for each ticker."""
    parquet = bootstrap_dir / "bootstrap_diffs.parquet"
    if not parquet.exists():
        print(f"  skip bootstrap: {parquet} not found")
        return
    df = pd.read_parquet(parquet)
    if exclude:
        # Drop any comparison that involves an excluded method.
        mask = df["comparison"].apply(lambda c: any(m in c for m in exclude))
        df = df[~mask]
    tickers = sorted(df["ticker"].unique())
    comparisons = sorted(df["comparison"].unique())
    metrics = sorted(df["metric"].unique())

    for ticker in tickers:
        sub = df[df["ticker"] == ticker]
        fig, axes = plt.subplots(len(comparisons), len(metrics),
                                 figsize=(4 * len(metrics), 3 * len(comparisons)),
                                 sharex=False)
        if len(comparisons) == 1:
            axes = np.array([axes])
        for r, lbl in enumerate(comparisons):
            for c, k in enumerate(metrics):
                ax = axes[r, c]
                vals = sub[(sub["comparison"] == lbl) & (sub["metric"] == k)]["diff"].values
                ax.hist(vals, bins=40, alpha=0.7, color="steelblue",
                        edgecolor="white", linewidth=0.3)
                ax.axvline(0, color="red", linestyle="--", linewidth=1)
                win = (vals < 0).mean() * 100
                ax.set_title(f"{k}  (win {win:.0f}%)", fontsize=9)
                if c == 0:
                    ax.set_ylabel(lbl, fontsize=8)
        fig.suptitle(f"{ticker} - bootstrap metric differences", fontsize=12, y=1.01)
        plt.tight_layout()
        out = out_dir / f"bootstrap_{ticker}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
    print(f"  wrote {len(tickers)} bootstrap plots in {out_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, default=RESULTS_DIR / "experiment")
    parser.add_argument("--bootstrap-dir", type=Path, default=RESULTS_DIR / "bootstrap")
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR / "figures")
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        metavar="METHOD",
        help=(
            "Method key(s) to drop from all plots, e.g. --exclude solo_diff. "
            "Use this to ignore methods without honest training runs whose "
            "inflated values distort the y-axis. "
            f"Known methods: {', '.join(SHORT_NAME)}."
        ),
    )
    args = parser.parse_args()

    if args.exclude:
        print(f"  excluding methods: {', '.join(args.exclude)}")

    _ensure_out(args.out_dir)
    plot_cross_seed(args.experiment_dir, args.out_dir, args.exclude)
    plot_per_ticker(args.experiment_dir, args.out_dir, args.exclude)
    plot_bootstrap(args.bootstrap_dir, args.out_dir, args.exclude)


if __name__ == "__main__":
    main()
