#!/usr/bin/env python
"""CPU-only diagnostic for the VaR variance-scale issue. No model, no GPU.

Reads a `series.parquet` written by run_risk_analysis.py and reports, per method:

  1. A dispersion table: realized std, the std of the per-day predictive mean,
     the mean per-day predictive spread, the implied marginal std, and the
     z-std = std((realized - pred_mean)/spread) -- the factor that would
     calibrate the spread (>1 => under-dispersed, the model claims tighter
     bands than reality).

  2. A VaR-coverage-vs-inflation sweep. The saved quantiles q01/q05/q10 and
     q90/q95/q99 are exactly the lower/upper VaR thresholds at alpha =
     0.01/0.05/0.10, so coverage under an inflation factor S is recomputed
     exactly (no model) via  q_new = pred_mean + S * (q_old - pred_mean).
     The S that minimizes mean |coverage - nominal| is the coverage-optimal
     factor per method.

This lets you confirm the scale deficit and read off the calibrating S without
re-running the diffusion. To then APPLY it consistently to VaR + ES + PIT +
mean-reversion, re-run:  run_risk_analysis.py --var-scale-mode auto  (or
--var-scale-mode manual --var-scale <S>).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# A normal's q95-q05 spans this many standard deviations; used to back out an
# approximate per-day predictive std from the two saved quantiles.
_Z_90 = 2 * 1.6448536269514722

LOWER = [(0.01, "q01"), (0.05, "q05"), (0.10, "q10")]
UPPER = [(0.01, "q99"), (0.05, "q95"), (0.10, "q90")]
METHOD_ORDER = ["solo_diff", "tc_diff", "solo_nw", "t_copula", "uncond_hist"]


def _coverage(g: pd.DataFrame, S: float) -> dict:
    """Empirical violation rate per (tail, alpha) under spread inflation S."""
    pm = g["pred_mean"].values
    real = g["realized"].values
    out = {}
    for a, col in LOWER:
        out[("lower", a)] = float((real < pm + S * (g[col].values - pm)).mean())
    for a, col in UPPER:
        out[("upper", a)] = float((real > pm + S * (g[col].values - pm)).mean())
    return out


def _cov_error(g: pd.DataFrame, S: float) -> float:
    c = _coverage(g, S)
    return float(np.mean([abs(c[("lower", a)] - a) + abs(c[("upper", a)] - a)
                          for a in (0.01, 0.05, 0.10)]) / 2)


def _optimal_S(g: pd.DataFrame, grid: np.ndarray) -> float:
    errs = [_cov_error(g, S) for S in grid]
    return float(grid[int(np.argmin(errs))])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--series", type=Path, required=True,
                    help="Path to a series.parquet from run_risk_analysis.py.")
    ap.add_argument("--factor", default=None,
                    help="Restrict to one risk factor (default: all pooled).")
    ap.add_argument("--sweep", type=str, default="1,2,3,4,5,6",
                    help="Comma-separated S values for the printed sweep.")
    args = ap.parse_args()

    df = pd.read_parquet(args.series)
    df = df[df["valid"]].copy()
    if args.factor:
        df = df[df["factor"] == args.factor]
    methods = [m for m in METHOD_ORDER if m in df["method"].unique().tolist()]
    methods += [m for m in df["method"].unique() if m not in methods]

    print(f"\nseries: {args.series}  |  rows (valid): {len(df)}  |  "
          f"factor: {args.factor or 'ALL pooled'}")

    # ---- 1. Dispersion table ----
    print("\n" + "=" * 90)
    print("PER-METHOD DISPERSION  (z-std = std((realized-pred_mean)/spread) = inflation S needed)")
    print("=" * 90)
    print(f'{"method":12s} {"realized_std":>12s} {"predmean_std":>12s} '
          f'{"spread":>8s} {"implied_marg":>12s} {"z_std=S":>9s}')
    for m in methods:
        g = df[df["method"] == m]
        spread = (g["q95"] - g["q05"]).values / _Z_90
        with np.errstate(invalid="ignore", divide="ignore"):
            z = (g["realized"].values - g["pred_mean"].values) / np.where(
                spread < 1e-12, np.nan, spread)
        z = z[np.isfinite(z)]
        implied = np.sqrt(g["pred_mean"].std() ** 2 + np.mean(spread ** 2))
        print(f'{m:12s} {g["realized"].std():12.3f} {g["pred_mean"].std():12.3f} '
              f'{spread.mean():8.3f} {implied:12.3f} {np.std(z):9.3f}')

    # ---- 2. VaR coverage vs inflation S ----
    sweep = [float(x) for x in args.sweep.split(",")]
    grid = np.round(np.arange(0.5, 8.01, 0.1), 2)
    print("\n" + "=" * 90)
    print("VaR COVERAGE vs INFLATION S   (target: lower/upper = nominal alpha)")
    print("=" * 90)
    for m in methods:
        g = df[df["method"] == m]
        s_opt = _optimal_S(g, grid)
        print(f"\n-- {m} --   coverage-optimal S = {s_opt:.2f}  "
              f"(|cov-nom| {_cov_error(g, 1.0):.4f} at S=1  ->  "
              f"{_cov_error(g, s_opt):.4f} at S={s_opt:.2f})")
        print(f'{"S":>5s} | {"L.01":>5s} {"L.05":>5s} {"L.10":>5s}  '
              f'{"U.01":>5s} {"U.05":>5s} {"U.10":>5s}')
        for S in sweep + [s_opt]:
            c = _coverage(g, S)
            tag = "  <- optimal" if S == s_opt else ""
            print(f'{S:5.2f} | {c[("lower",.01)]:5.3f} {c[("lower",.05)]:5.3f} '
                  f'{c[("lower",.10)]:5.3f}  {c[("upper",.01)]:5.3f} '
                  f'{c[("upper",.05)]:5.3f} {c[("upper",.10)]:5.3f}{tag}')
        print(f'{"nom":>5s} | {0.01:5.3f} {0.05:5.3f} {0.10:5.3f}  '
              f'{0.01:5.3f} {0.05:5.3f} {0.10:5.3f}')

    print("\nApply consistently (VaR+ES+PIT+mean-reversion) with:\n"
          "  python scripts/run_risk_analysis.py --var-scale-mode auto\n"
          "  python scripts/run_risk_analysis.py --var-scale-mode manual --var-scale <S>")


if __name__ == "__main__":
    main()
