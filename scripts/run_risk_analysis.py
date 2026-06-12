#!/usr/bin/env python
"""Risk-focused model comparison: VaR backtests and a mean-reversion strategy.

For each method this script draws an *ensemble* of --n-samples per test day
(rather than the single draw per day used by run_experiment.py), giving a full
predictive distribution per (day, factor). From those ensembles it computes:

  * VaR backtests at 1/5/10%, both tails: violation rate vs nominal (with
    pooled 'ALL' rows aggregating across tickers for power at the 1% level)
    and an expected-shortfall comparison — on violation days, the average
    realized exceedance vs the model's own conditional tail expectation.
  * Mean-reversion strategy: when the realized move sits beyond the model's
    tau / (1-tau) predictive quantile (tau = 0.80/0.90/0.95), bet on next-day
    reversion. Scores both the strategy PnL and whether the model's own
    conditional mean for t+1 predicts the reversion.

Realized test values are standardized with train stats but NOT winsorized
(unlike run_experiment.py): clipping realized tails at train percentiles
corrupts exactly the tail behaviour VaR measures. Invalid (stale-fit) days are
excluded from scoring. An unconditional-historical baseline (`uncond_hist`,
resampling train rows) is included so the value of conditioning is visible.

Caveat inherited from the package: the conditioning vector includes SAME-DAY
VIX, so all methods see contemporaneous market vol. Comparisons across methods
are fair, but absolute numbers are not pure out-of-sample forecasts, and the
mean-reversion PnL is not directly tradable as-is.

Smoke test (CPU, ~minutes):

    python scripts/run_risk_analysis.py --epochs-solo 50 --epochs-pooled 100 \
        --n-samples 32 --tickers MU PLTR

Full run (GPU):

    python scripts/run_risk_analysis.py
    python scripts/plot_risk_analysis.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from options_diffusion.config import (
    CACHE_DIR, COND_EPOCHS_POOLED, COND_EPOCHS_SOLO, DEVICE, RESULTS_DIR,
    SECTOR_TICKERS, SVI_RISK_FACTOR_NAMES,
)
from options_diffusion.data.preprocess import PreparedData, prepare_data
from options_diffusion.eval.experiment import (
    _train_pooled_diffusion, _train_solo_diffusion, set_global_seed,
)
from options_diffusion.eval.risk import (
    MR_TAUS, VAR_ALPHAS, mean_reversion_eval, pit_from_ensemble, var_backtest,
)
from options_diffusion.models.baselines import CondPCABootstrap, CondTCopula

RISK_METHODS = ["solo_diff", "tc_diff", "solo_nw", "t_copula", "uncond_hist"]
SERIES_QS = (0.01, 0.05, 0.10, 0.90, 0.95, 0.99)


def realized_unwinsorized(prep: PreparedData, ticker: str) -> tuple[np.ndarray, np.ndarray]:
    """Test-set changes standardized with train stats, tails NOT clipped."""
    r_test = prep.changes[ticker][prep.test_idx[ticker]].values
    pp = prep.preprocess[ticker]
    realized = (r_test - pp["mu"]) / pp["sigma"]
    valid = prep.masks[ticker][prep.test_idx[ticker]] & np.isfinite(realized)
    return np.nan_to_num(realized, nan=0.0), valid


def _tiled_sample(sample_fn, cond: np.ndarray, n_samples: int, seed: int,
                  batch_rows: int) -> np.ndarray:
    """Draw n_samples per conditioning row by tiling cond; returns (n, K, d)."""
    n = len(cond)
    tiled = np.repeat(cond, n_samples, axis=0)
    chunks = []
    for s in range(0, len(tiled), batch_rows):
        chunks.append(sample_fn(tiled[s:s + batch_rows], seed=seed + s))
    out = np.vstack(chunks)
    return out.reshape(n, n_samples, -1)


def build_ensembles(prep: PreparedData, ticker: str, methods: list[str],
                    models: dict, n_samples: int, seed: int,
                    batch_rows: int) -> dict[str, np.ndarray]:
    """Per-method predictive ensembles of shape (n_test, n_samples, n_factors)."""
    cond_te = prep.cond_ewma_test[ticker]
    s0, s1 = prep.test_offsets[ticker]
    ens = {}

    if "solo_diff" in methods:
        solo = models["solo_diff"][ticker]
        ens["solo_diff"] = _tiled_sample(
            lambda c, seed: solo.sample(c.astype(np.float32), seed=seed),
            cond_te, n_samples, seed, batch_rows)

    if "tc_diff" in methods:
        tc = models["tc_diff"]
        tc_cond = prep.tc_cond_ewma_test[s0:s1]
        tc_ids = prep.ticker_ids_test[s0:s1]
        # Tile ids alongside cond explicitly (cond rows and ids must align).
        tiled_ids = np.repeat(tc_ids, n_samples, axis=0)
        tiled_cond = np.repeat(tc_cond, n_samples, axis=0)
        chunks = []
        for s in range(0, len(tiled_cond), batch_rows):
            chunks.append(tc.sample(tiled_cond[s:s + batch_rows].astype(np.float32),
                                    tiled_ids[s:s + batch_rows], seed=seed + s))
        ens["tc_diff"] = np.vstack(chunks).reshape(len(tc_cond), n_samples, -1)

    if "solo_nw" in methods:
        nw = models["solo_nw"][ticker]
        ens["solo_nw"] = _tiled_sample(nw.sample, cond_te, n_samples, seed, batch_rows)

    if "t_copula" in methods:
        cop = models["t_copula"][ticker]
        ens["t_copula"] = _tiled_sample(cop.sample, cond_te, n_samples, seed, batch_rows)

    if "uncond_hist" in methods:
        X_tr = prep.std_train[ticker]
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(X_tr), size=(len(cond_te), n_samples))
        ens["uncond_hist"] = X_tr[idx]

    return ens


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR / "risk")
    parser.add_argument("--seed", type=int, default=0,
                        help="Training + sampling seed (single seed; rerun "
                             "with others to gauge training noise).")
    parser.add_argument("--n-samples", type=int, default=128,
                        help="Ensemble draws per test day per method.")
    parser.add_argument("--epochs-solo", type=int, default=COND_EPOCHS_SOLO)
    parser.add_argument("--epochs-pooled", type=int, default=COND_EPOCHS_POOLED)
    parser.add_argument("--methods", nargs="*", default=list(RISK_METHODS),
                        choices=RISK_METHODS)
    parser.add_argument("--tickers", nargs="*", default=SECTOR_TICKERS)
    parser.add_argument("--batch-rows", type=int, default=8192,
                        help="Max rows per diffusion sampling batch.")
    parser.add_argument("--bw-scale", type=float, default=1.0,
                        help="Multiply NW / t-copula kernel bandwidths by this "
                             "factor. With per-dim Silverman bandwidths in 19 "
                             "dims the product-kernel weights collapse onto a "
                             "single training row at test time, making those "
                             "baselines' conditional distributions degenerate "
                             "point masses; try 2-5 to compare them with "
                             "non-degenerate predictive distributions.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    verbose = not args.quiet

    print(f"Device: {DEVICE}")
    prep = prepare_data(args.tickers, args.cache_dir)
    print(f"Tickers: {args.tickers} | test days/ticker: "
          f"{[int(prep.test_idx[t].sum()) for t in args.tickers]}")
    print(f"Methods: {args.methods} | ensemble size: {args.n_samples}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(args.seed)

    # ---- Train everything once (one seed) ----
    models: dict = {"solo_diff": {}, "solo_nw": {}, "t_copula": {}}
    if "tc_diff" in args.methods:
        print(f"Training pooled TC-Diff ({args.epochs_pooled} epochs)...")
        models["tc_diff"], _ = _train_pooled_diffusion(prep, args.epochs_pooled, verbose)
    for ticker in prep.tickers:
        X_tr = prep.std_train[ticker]
        cond_tr = prep.cond_ewma_train[ticker]
        m_tr = prep.masks[ticker][prep.train_idx[ticker]]
        if "solo_diff" in args.methods:
            if verbose:
                print(f"Training solo diffusion for {ticker} ({args.epochs_solo} epochs)...")
            models["solo_diff"][ticker], _ = _train_solo_diffusion(
                X_tr, cond_tr, m_tr, epochs=args.epochs_solo, verbose=False)
        if "solo_nw" in args.methods:
            nw = CondPCABootstrap().fit(X_tr, cond_tr)
            nw.bw = nw.bw * args.bw_scale
            models["solo_nw"][ticker] = nw
        if "t_copula" in args.methods:
            cop = CondTCopula().fit(X_tr, cond_tr, n_bins=3)
            cop.bw = cop.bw * args.bw_scale
            models["t_copula"][ticker] = cop

    # ---- Evaluate ----
    var_rows, mr_rows, series_rows, signal_rows = [], [], [], []
    # Pooled violation counts across tickers: (method, factor, tail, alpha)
    pooled_viol: dict[tuple, list[int]] = {}

    for ticker in prep.tickers:
        t0 = time.time()
        print(f"\n[{ticker}] drawing ensembles ({args.n_samples}/day x "
              f"{int(prep.test_idx[ticker].sum())} days)...")
        ens = build_ensembles(prep, ticker, args.methods, models,
                              args.n_samples, args.seed, args.batch_rows)
        realized, valid = realized_unwinsorized(prep, ticker)
        dates = prep.changes[ticker].index[prep.test_idx[ticker]]

        for method, samples in ens.items():
            for fi, factor in enumerate(SVI_RISK_FACTOR_NAMES):
                sf = samples[:, :, fi]
                rf = realized[:, fi]
                vf = valid[:, fi]
                pit = pit_from_ensemble(sf, rf)
                pred_mean = sf.mean(axis=1)

                # VaR backtests, both tails
                for tail in ("lower", "upper"):
                    for alpha in VAR_ALPHAS:
                        row = var_backtest(sf, rf, vf, alpha, tail)
                        row.update({"ticker": ticker, "method": method,
                                    "factor": factor})
                        var_rows.append(row)
                        key = (method, factor, tail, alpha)
                        agg = pooled_viol.setdefault(key, [0, 0])
                        agg[0] += row["n_obs"]
                        agg[1] += row["n_viol"]

                # Mean-reversion rule at each threshold
                for tau in MR_TAUS:
                    summ, per_sig = mean_reversion_eval(pit, rf, pred_mean, vf, tau)
                    summ.update({"ticker": ticker, "method": method,
                                 "factor": factor})
                    mr_rows.append(summ)
                    for j, ti in enumerate(per_sig["t_index"]):
                        signal_rows.append({
                            "ticker": ticker, "method": method, "factor": factor,
                            "tau": tau, "date": dates[ti],
                            "direction": per_sig["direction"][j],
                            "next_ret": per_sig["next_ret"][j],
                            "pnl": per_sig["pnl"][j],
                        })

                # Per-day series for plots
                qs = {f"q{int(q*100):02d}": np.quantile(sf, q, axis=1)
                      for q in SERIES_QS}
                for di in range(len(rf)):
                    series_rows.append({
                        "ticker": ticker, "method": method, "factor": factor,
                        "date": dates[di], "realized": rf[di],
                        "valid": bool(vf[di]), "pit": pit[di],
                        "pred_mean": pred_mean[di],
                        **{k: v[di] for k, v in qs.items()},
                    })
        print(f"[{ticker}] done in {(time.time() - t0)/60:.1f} min")

    # Pooled-across-tickers coverage rows.
    for (method, factor, tail, alpha), (n_obs, n_viol) in pooled_viol.items():
        var_rows.append({
            "ticker": "ALL", "method": method, "factor": factor,
            "tail": tail, "alpha": alpha, "n_obs": n_obs, "n_viol": n_viol,
            "viol_rate": n_viol / n_obs if n_obs else np.nan,
            "realized_es": np.nan, "model_es": np.nan, "es_ratio": np.nan,
        })

    # ---- Save ----
    out = args.out_dir
    pd.DataFrame(var_rows).to_parquet(out / "var_backtest.parquet")
    pd.DataFrame(mr_rows).to_parquet(out / "mean_reversion.parquet")
    pd.DataFrame(series_rows).to_parquet(out / "series.parquet")
    pd.DataFrame(signal_rows).to_parquet(out / "mr_signals.parquet")
    (out / "run_config.json").write_text(json.dumps({
        "seed": args.seed, "n_samples": args.n_samples,
        "epochs_solo": args.epochs_solo, "epochs_pooled": args.epochs_pooled,
        "methods": args.methods, "tickers": args.tickers,
        "bw_scale": args.bw_scale,
    }, indent=2))

    # ---- Console summary ----
    var_df = pd.DataFrame(var_rows)
    print("\n" + "=" * 88)
    print("VaR COVERAGE, POOLED ACROSS TICKERS+FACTORS (empirical violation rate vs nominal)")
    print("=" * 88)
    pooled = (var_df[var_df["ticker"] == "ALL"]
              .groupby(["method", "tail", "alpha"])
              .apply(lambda g: g["n_viol"].sum() / g["n_obs"].sum(),
                     include_groups=False)
              .unstack(["tail", "alpha"]).round(4))
    print(pooled.to_string())

    print("\nEXPECTED SHORTFALL on violation days (realized vs model, 5% lower tail):")
    es = var_df[(var_df["ticker"] != "ALL") & (var_df["tail"] == "lower")
                & (var_df["alpha"] == 0.05) & (var_df["n_viol"] > 0)]
    es_sum = (es.groupby("method")
              .apply(lambda g: pd.Series({
                  "realized_es": np.average(g["realized_es"], weights=g["n_viol"]),
                  "model_es": np.average(g["model_es"], weights=g["n_viol"]),
                  "n_viol": int(g["n_viol"].sum())}), include_groups=False))
    print(es_sum.round(3).to_string())

    mr_df = pd.DataFrame(mr_rows)
    print("\nMean-reversion strategy, pooled across tickers x factors:")
    mr_pool = (mr_df.groupby(["method", "tau"])
               .apply(lambda g: pd.Series({
                   "n_signals": int(g["n_signals"].sum()),
                   "hit_rate": np.average(g["hit_rate"].fillna(0),
                                          weights=g["n_signals"].clip(lower=0) + 1e-9),
                   "mean_pnl": (g["total_pnl"].sum()
                                / max(g["n_signals"].sum(), 1)),
                   "model_dir_acc": np.average(g["model_dir_acc"].fillna(0),
                                               weights=g["n_signals"].clip(lower=0) + 1e-9),
               }), include_groups=False)
               .reset_index())
    print(mr_pool.to_string(index=False))
    print(f"\nWrote parquets to {out}. Run scripts/plot_risk_analysis.py for figures.")


if __name__ == "__main__":
    main()
