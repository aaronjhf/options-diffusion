"""Multi-seed training + evaluation driver.

The main source of stochasticity exposed here is the *training seed* (PyTorch +
NumPy + Python random). For each seed we re-train both diffusion models from
scratch and re-fit the baselines (which then resample with the same seed),
then evaluate every method once on the fixed test set. Aggregating across
seeds gives mean +/- std per (ticker, method, metric).
"""
from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..config import (
    BATCH_SIZE, COND_DIM_EWMA, COND_EPOCHS_POOLED, COND_EPOCHS_SOLO, N_RF,
    N_TIMESTEPS, TICKER_EMBED_DIM,
)
from ..data.preprocess import PreparedData
from ..models.baselines import CondPCABootstrap, CondTCopula
from ..models.diffusion import ConditionalDDPM, TickerCondDDPM
from .metrics import METRIC_KEYS, compute_all_metrics


METHODS = ["solo_diff", "solo_nw", "tc_diff", "t_copula"]


def set_global_seed(seed: int) -> None:
    """Seed every RNG that influences training or sampling."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _train_pooled_diffusion(prep: PreparedData, epochs: int, verbose: bool):
    ddpm = TickerCondDDPM(
        data_dim=prep.X_train.shape[1],
        n_tickers=len(prep.tickers),
        cond_input_dim=COND_DIM_EWMA,
        n_timesteps=N_TIMESTEPS,
        ticker_embed_dim=TICKER_EMBED_DIM,
    )
    losses = ddpm.train_model(
        prep.X_train.astype(np.float32),
        prep.tc_cond_ewma_train.astype(np.float32),
        prep.ticker_ids_train,
        mask=prep.mask_train,
        epochs=epochs, batch_size=BATCH_SIZE, verbose=verbose,
    )
    return ddpm, losses


def _train_solo_diffusion(X_tr, cond_tr, m_tr, epochs: int, verbose: bool):
    ddpm = ConditionalDDPM(
        data_dim=X_tr.shape[1],
        cond_input_dim=COND_DIM_EWMA,
        n_timesteps=N_TIMESTEPS,
    )
    losses = ddpm.train_model(
        X_tr.astype(np.float32),
        cond_tr.astype(np.float32),
        mask=m_tr.astype(np.float32),
        epochs=epochs, batch_size=BATCH_SIZE, verbose=verbose,
    )
    return ddpm, losses


def evaluate_one_seed(prep: PreparedData, seed: int, *,
                      epochs_solo: int = COND_EPOCHS_SOLO,
                      epochs_pooled: int = COND_EPOCHS_POOLED,
                      methods=tuple(METHODS),
                      verbose: bool = True) -> list[dict]:
    """Train and evaluate one seed. Returns a list of metric rows."""
    set_global_seed(seed)
    rows = []
    t0 = time.time()

    # ---- Pooled (ticker-conditioned) diffusion: one fit shared across tickers ----
    if "tc_diff" in methods:
        if verbose:
            print(f"  [seed {seed}] training pooled ticker-conditioned diffusion...")
        tc_diff, _ = _train_pooled_diffusion(prep, epochs_pooled, verbose)
        if verbose:
            print(f"  [seed {seed}] pooled done in {(time.time()-t0)/60:.1f} min")
    else:
        tc_diff = None

    # ---- Per-ticker ----
    for ti, ticker in enumerate(prep.tickers):
        if verbose:
            print(f"  [seed {seed}] ticker {ti+1}/{len(prep.tickers)}: {ticker}")
        X_tr = prep.std_train[ticker]
        X_te = prep.std_test[ticker]
        cond_tr = prep.cond_ewma_train[ticker]
        cond_te = prep.cond_ewma_test[ticker]
        m_tr = prep.masks[ticker][prep.train_idx[ticker]]
        s0, s1 = prep.test_offsets[ticker]

        # 1. Solo diffusion
        if "solo_diff" in methods:
            solo, _ = _train_solo_diffusion(X_tr, cond_tr, m_tr,
                                            epochs=epochs_solo, verbose=False)
            gen = solo.sample(cond_te.astype(np.float32), seed=seed)
            for k, v in compute_all_metrics(X_te, gen).items():
                rows.append({"seed": seed, "ticker": ticker, "method": "solo_diff",
                             "metric": k, "value": v})

        # 2. NW (PCA bootstrap) baseline
        if "solo_nw" in methods:
            nw = CondPCABootstrap().fit(X_tr, cond_tr)
            gen = nw.sample(cond_te, seed=seed)
            for k, v in compute_all_metrics(X_te, gen).items():
                rows.append({"seed": seed, "ticker": ticker, "method": "solo_nw",
                             "metric": k, "value": v})

        # 3. Ticker-conditioned diffusion (already trained pooled)
        if "tc_diff" in methods:
            tc_ids_ti = prep.ticker_ids_test[s0:s1]
            gen = tc_diff.sample(
                prep.tc_cond_ewma_test[s0:s1].astype(np.float32),
                tc_ids_ti, seed=seed,
            )
            for k, v in compute_all_metrics(X_te, gen).items():
                rows.append({"seed": seed, "ticker": ticker, "method": "tc_diff",
                             "metric": k, "value": v})

        # 4. Conditional t-copula
        if "t_copula" in methods:
            cop = CondTCopula().fit(X_tr, cond_tr, n_bins=3)
            gen = cop.sample(cond_te, seed=seed)
            for k, v in compute_all_metrics(X_te, gen).items():
                rows.append({"seed": seed, "ticker": ticker, "method": "t_copula",
                             "metric": k, "value": v})

    if verbose:
        print(f"  [seed {seed}] complete in {(time.time()-t0)/60:.1f} min")
    return rows


def run_experiment(prep: PreparedData, seeds: list[int], out_dir: Path, *,
                   epochs_solo: int = COND_EPOCHS_SOLO,
                   epochs_pooled: int = COND_EPOCHS_POOLED,
                   methods=tuple(METHODS),
                   verbose: bool = True) -> pd.DataFrame:
    """Run all seeds, writing per-seed rows to `out_dir` for resilience."""
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for s in seeds:
        seed_file = out_dir / f"seed_{s:04d}.parquet"
        if seed_file.exists():
            print(f"[seed {s}] resuming from {seed_file.name}")
            all_rows.extend(pd.read_parquet(seed_file).to_dict("records"))
            continue
        rows = evaluate_one_seed(prep, s,
                                 epochs_solo=epochs_solo,
                                 epochs_pooled=epochs_pooled,
                                 methods=methods, verbose=verbose)
        pd.DataFrame(rows).to_parquet(seed_file)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df.to_parquet(out_dir / "all_seeds.parquet")
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Mean +/- std across seeds, indexed by (ticker, method, metric)."""
    g = df.groupby(["ticker", "method", "metric"])["value"]
    summary = g.agg(["mean", "std", "count"]).reset_index()
    return summary


def summarize_pooled(df: pd.DataFrame) -> pd.DataFrame:
    """Mean +/- std across (seeds x tickers), per (method, metric)."""
    g = df.groupby(["method", "metric"])["value"]
    return g.agg(["mean", "std", "count"]).reset_index()
