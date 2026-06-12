#!/usr/bin/env python
"""Figures for the risk analysis produced by run_risk_analysis.py.

Writes to results/figures/risk/:
  var_coverage.png          empirical vs nominal VaR violation rate, both
                            tails, with 95% binomial bands around nominal
  expected_shortfall.png    realized vs model tail loss on violation days
  var_timeline_<ticker>.png realized vs 90% predictive band, violations marked
  mr_hit_pnl.png            mean-reversion hit rate and PnL/trade vs threshold
  mr_cum_pnl.png            cumulative strategy PnL through the test period
"""
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
    "tc_diff": "TC Diff",
    "solo_nw": "NW",
    "t_copula": "t-Copula",
    "uncond_hist": "Uncond Hist",
}
COLORS = {
    "solo_diff": "#4c72b0",
    "tc_diff": "#c44e52",
    "solo_nw": "#55a868",
    "t_copula": "#8172b2",
    "uncond_hist": "#7f7f7f",
}
HEADLINE_FACTOR = "atm_3m"
HEADLINE_TAU = 0.90


def _methods_in(df: pd.DataFrame) -> list[str]:
    order = list(SHORT_NAME)
    present = set(df["method"].unique())
    return [m for m in order if m in present] + sorted(present - set(order))


def plot_var_coverage(var_df: pd.DataFrame, out: Path):
    """Empirical violation rate vs nominal alpha, pooled over tickers+factors."""
    pooled = var_df[var_df["ticker"] == "ALL"]
    methods = _methods_in(pooled)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, tail in zip(axes, ["lower", "upper"]):
        sub = pooled[pooled["tail"] == tail]
        alphas = sorted(sub["alpha"].unique())
        # 95% binomial band around nominal given pooled n_obs
        n_tot = sub.groupby("alpha")["n_obs"].sum() / max(sub["factor"].nunique(), 1)
        lo_band, hi_band = [], []
        for a in alphas:
            n = float(n_tot.loc[a])
            se = np.sqrt(a * (1 - a) / max(n, 1))
            lo_band.append(max(a - 1.96 * se, 0))
            hi_band.append(a + 1.96 * se)
        ax.fill_between(alphas, lo_band, hi_band, color="k", alpha=0.08,
                        label="95% band (per ticker-factor n)")
        ax.plot(alphas, alphas, "k--", lw=1, label="perfect coverage")
        for m in methods:
            g = (sub[sub["method"] == m].groupby("alpha")
                 .apply(lambda x: x["n_viol"].sum() / x["n_obs"].sum(),
                        include_groups=False))
            ax.plot(g.index, g.values, "o-", color=COLORS.get(m, "gray"),
                    label=SHORT_NAME.get(m, m))
        ax.set_xlabel("nominal VaR level α")
        ax.set_title(f"{tail} tail")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("empirical violation rate")
    axes[1].legend(fontsize=8, loc="upper left")
    fig.suptitle("VaR coverage (pooled over tickers and factors): "
                 "above the line = risk underestimated")
    plt.tight_layout()
    fig.savefig(out / "var_coverage.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_expected_shortfall(var_df: pd.DataFrame, out: Path):
    """Realized vs model tail loss on violation days, per alpha and tail.

    Paired bars: if the dark (realized) bar extends past the light (model)
    bar, actual tail losses are bigger than the model predicted.
    """
    sub = var_df[(var_df["ticker"] != "ALL") & (var_df["n_viol"] > 0)]
    methods = _methods_in(sub)
    combos = [("lower", 0.01), ("lower", 0.05), ("upper", 0.01), ("upper", 0.05)]
    fig, axes = plt.subplots(1, len(combos), figsize=(3.4 * len(combos), 4.2))
    for ax, (tail, alpha) in zip(axes, combos):
        g = sub[(sub["tail"] == tail) & (sub["alpha"] == alpha)]
        ys = np.arange(len(methods))
        for i, m in enumerate(methods):
            gm = g[g["method"] == m]
            if gm.empty:
                continue
            w = gm["n_viol"].values
            realized = np.average(gm["realized_es"], weights=w)
            model = np.average(gm["model_es"], weights=w)
            color = COLORS.get(m, "gray")
            ax.barh(i + 0.18, realized, height=0.34, color=color,
                    label="realized" if i == 0 else None)
            ax.barh(i - 0.18, model, height=0.34, color=color, alpha=0.45,
                    label="model" if i == 0 else None)
        ax.set_yticks(ys, [SHORT_NAME.get(m, m) for m in methods]
                      if ax is axes[0] else [""] * len(methods))
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(f"{tail} tail, α = {alpha:g}", fontsize=10)
        ax.set_xlabel("mean tail loss on violation days (σ)")
        ax.grid(axis="x", alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Expected shortfall: solid = realized, faded = model prediction "
                 "(matching lengths = tail sizes well estimated)")
    plt.tight_layout()
    fig.savefig(out / "expected_shortfall.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_var_timelines(series_df: pd.DataFrame, out: Path,
                       factor: str = HEADLINE_FACTOR):
    sub_all = series_df[series_df["factor"] == factor]
    methods = _methods_in(sub_all)
    for ticker in sorted(sub_all["ticker"].unique()):
        sub_t = sub_all[sub_all["ticker"] == ticker]
        fig, axes = plt.subplots(len(methods), 1,
                                 figsize=(11, 2.2 * len(methods)),
                                 sharex=True, sharey=True)
        if len(methods) == 1:
            axes = [axes]
        for ax, m in zip(axes, methods):
            g = sub_t[sub_t["method"] == m].sort_values("date")
            dates = pd.to_datetime(g["date"])
            ax.fill_between(dates, g["q05"], g["q95"], alpha=0.25,
                            color=COLORS.get(m, "gray"), label="90% band")
            ax.plot(dates, g["realized"], "k-", lw=0.8, label="realized")
            v = g["valid"] & ((g["realized"] < g["q05"]) | (g["realized"] > g["q95"]))
            ax.plot(dates[v], g["realized"][v], "rx", ms=6, label="violation")
            ax.set_ylabel(SHORT_NAME.get(m, m), fontsize=9)
            ax.grid(alpha=0.3)
        axes[0].legend(fontsize=7, ncol=3, loc="upper left")
        fig.suptitle(f"{ticker} {factor}: realized change vs 90% predictive band")
        plt.tight_layout()
        fig.savefig(out / f"var_timeline_{ticker}.png", dpi=150,
                    bbox_inches="tight")
        plt.close(fig)


def plot_mr_hit_pnl(mr_df: pd.DataFrame, out: Path):
    methods = _methods_in(mr_df)
    taus = sorted(mr_df["tau"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    panels = [
        ("hit_rate", "reversion hit rate", 0.5),
        ("mean_pnl", "mean PnL per trade (σ units)", 0.0),
        ("model_dir_acc", "model next-day direction accuracy", 0.5),
    ]
    for ax, (col, title, refline) in zip(axes, panels):
        for m in methods:
            g = mr_df[mr_df["method"] == m]
            ys = []
            for tau in taus:
                gt = g[g["tau"] == tau]
                w = gt["n_signals"].clip(lower=0)
                if col == "mean_pnl":
                    ys.append(gt["total_pnl"].sum() / max(gt["n_signals"].sum(), 1))
                else:
                    vals, ww = gt[col].values, w.values.astype(float)
                    ok = np.isfinite(vals) & (ww > 0)
                    ys.append(np.average(vals[ok], weights=ww[ok]) if ok.any() else np.nan)
            ax.plot(taus, ys, "o-", color=COLORS.get(m, "gray"),
                    label=SHORT_NAME.get(m, m))
        ax.axhline(refline, color="k", linestyle="--", lw=1)
        ax.set_xlabel("predictive-quantile threshold τ")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Mean-reversion rule: signal when realized move crosses the "
                 "model's τ / (1-τ) predictive quantile (pooled)")
    plt.tight_layout()
    fig.savefig(out / "mr_hit_pnl.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_mr_cum_pnl(sig_df: pd.DataFrame, out: Path, tau: float = HEADLINE_TAU):
    sub = sig_df[sig_df["tau"] == tau].copy()
    if sub.empty:
        print("  skip mr_cum_pnl: no signals at tau =", tau)
        return
    methods = _methods_in(sub)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for m in methods:
        g = sub[sub["method"] == m].sort_values("date")
        if g.empty:
            continue
        daily = g.groupby("date")["pnl"].sum()
        ax.plot(pd.to_datetime(daily.index), daily.cumsum().values,
                color=COLORS.get(m, "gray"), label=f"{SHORT_NAME.get(m, m)} "
                f"({len(g)} trades)")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("cumulative PnL (σ units, sum over tickers+factors)")
    ax.set_title(f"Mean-reversion strategy cumulative PnL, τ = {tau}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out / "mr_cum_pnl.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--risk-dir", type=Path, default=RESULTS_DIR / "risk")
    parser.add_argument("--out-dir", type=Path,
                        default=RESULTS_DIR / "figures" / "risk")
    parser.add_argument("--exclude", nargs="*", default=[], metavar="METHOD",
                        help="Method key(s) to drop from all plots.")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    def load(name):
        p = args.risk_dir / name
        if not p.exists():
            print(f"missing {p} — run scripts/run_risk_analysis.py first")
            sys.exit(1)
        df = pd.read_parquet(p)
        if args.exclude and "method" in df.columns:
            df = df[~df["method"].isin(args.exclude)]
        return df

    var_df = load("var_backtest.parquet")
    mr_df = load("mean_reversion.parquet")
    series_df = load("series.parquet")
    sig_df = load("mr_signals.parquet")

    plot_var_coverage(var_df, args.out_dir)
    plot_expected_shortfall(var_df, args.out_dir)
    plot_var_timelines(series_df, args.out_dir)
    plot_mr_hit_pnl(mr_df, args.out_dir)
    plot_mr_cum_pnl(sig_df, args.out_dir)
    print(f"wrote figures to {args.out_dir}")


if __name__ == "__main__":
    main()
