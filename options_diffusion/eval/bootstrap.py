"""Paired bootstrap test comparing TC-Diff against NW and t-Copula on the test set.

This complements `experiment.py` by varying the *test-set sampling*, not the
training seed. Models are evaluated once per test point and cached; bootstrap
resamples (real, generated) index pairs together so model inference runs only
once per ticker per fit.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.preprocess import PreparedData
from ..models.baselines import CondPCABootstrap, CondTCopula
from ..models.diffusion import TickerCondDDPM
from .metrics import compute_all_metrics

BOOT_METRIC_KEYS = ["SWD", "MMD2", "Frechet", "Energy", "CorrDist"]
LABEL_TC = "TC-Diff"
LABEL_NW = "NW"
LABEL_COP = "t-Copula"
LABEL_D1 = "TC-Diff - NW"
LABEL_D2 = "TC-Diff - t-Copula"


def _gen_cache_per_ticker(prep: PreparedData, tc_diff: TickerCondDDPM, seed_for_baselines: int):
    """Run each model once per test point, cache the generated samples."""
    gen_cache = {}
    for ti, ticker in enumerate(prep.tickers):
        cond_te = prep.cond_ewma_test[ticker]
        s0, s1 = prep.test_offsets[ticker]
        tc_ids = prep.ticker_ids_test[s0:s1]

        gen_tc = tc_diff.sample(
            prep.tc_cond_ewma_test[s0:s1].astype(np.float32), tc_ids, seed=ti)

        nw = CondPCABootstrap().fit(prep.std_train[ticker], prep.cond_ewma_train[ticker])
        gen_nw = nw.sample(cond_te, seed=ti)

        cop = CondTCopula().fit(prep.std_train[ticker], prep.cond_ewma_train[ticker], n_bins=3)
        gen_cop = cop.sample(cond_te, seed=ti)

        gen_cache[ticker] = {LABEL_TC: gen_tc, LABEL_NW: gen_nw, LABEL_COP: gen_cop}
        print(f"  {ticker}: cached {len(cond_te)} samples x 3 methods")
    return gen_cache


def run_paired_bootstrap(
    prep: PreparedData,
    tc_diff: TickerCondDDPM,
    out_dir: Path,
    n_boot: int = 2000,
    boot_seed: int = 42,
):
    """Run paired bootstrap, save per-ticker diff arrays to disk.

    Returns the diff_stats dict for plotting downstream.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Pre-generating cached test samples for bootstrap ...")
    gen_cache = _gen_cache_per_ticker(prep, tc_diff, boot_seed)

    diff_stats = {
        ticker: {lbl: {k: [] for k in BOOT_METRIC_KEYS}
                 for lbl in [LABEL_D1, LABEL_D2]}
        for ticker in prep.tickers
    }

    rng = np.random.default_rng(boot_seed)
    for ticker in prep.tickers:
        X_te = prep.std_test[ticker]
        n_te = len(X_te)
        gc = gen_cache[ticker]
        t0 = time.time()
        print(f"{ticker}: bootstrapping {n_boot} resamples of size {n_te} ...")
        for _ in range(n_boot):
            idx = rng.choice(n_te, size=n_te, replace=True)
            X_b = X_te[idx]
            m_tc = compute_all_metrics(X_b, gc[LABEL_TC][idx])
            m_nw = compute_all_metrics(X_b, gc[LABEL_NW][idx])
            m_cop = compute_all_metrics(X_b, gc[LABEL_COP][idx])
            for k in BOOT_METRIC_KEYS:
                diff_stats[ticker][LABEL_D1][k].append(m_tc[k] - m_nw[k])
                diff_stats[ticker][LABEL_D2][k].append(m_tc[k] - m_cop[k])
        print(f"  done in {(time.time()-t0)/60:.1f} min")

    # Save as a long-format DataFrame
    rows = []
    for ticker in prep.tickers:
        for lbl in [LABEL_D1, LABEL_D2]:
            for k in BOOT_METRIC_KEYS:
                for b, v in enumerate(diff_stats[ticker][lbl][k]):
                    rows.append({"ticker": ticker, "comparison": lbl, "metric": k,
                                 "boot": b, "diff": v})
    df = pd.DataFrame(rows)
    df.to_parquet(out_dir / "bootstrap_diffs.parquet")
    return diff_stats


def summarize_bootstrap(diff_stats):
    """Per-ticker table: mean +/- std and win-rate (TC-Diff better => negative)."""
    summary_rows = []
    for ticker, by_lbl in diff_stats.items():
        for lbl, by_metric in by_lbl.items():
            for k, vals in by_metric.items():
                v = np.asarray(vals)
                summary_rows.append({
                    "ticker": ticker, "comparison": lbl, "metric": k,
                    "mean": float(v.mean()), "std": float(v.std()),
                    "win_rate": float((v < 0).mean()),
                    "n_boot": len(v),
                })
    return pd.DataFrame(summary_rows)
