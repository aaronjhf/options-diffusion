"""Compute risk-factor changes, split, standardize, and build EWMA conditioning."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import yfinance as yf

from ..config import (
    COND_DIM_EWMA, EWMA_LONG_ALPHA, EWMA_SHORT_ALPHA, GAP_DAYS, LOG_DIFF_FEATURES,
    N_RF, SVI_RISK_FACTOR_NAMES, TRAIN_FRAC,
)


@dataclass
class PreparedData:
    """Container for everything the experiment script needs.

    All per-ticker arrays are keyed by ticker symbol. Stacked arrays are
    ordered according to `tickers` (the ticker list used to build them).
    """
    tickers: list[str]
    # per-ticker
    rf_levels: Dict[str, pd.DataFrame]
    changes: Dict[str, pd.DataFrame]
    masks: Dict[str, np.ndarray]
    vix: Dict[str, pd.Series]
    train_idx: Dict[str, np.ndarray]
    test_idx: Dict[str, np.ndarray]
    std_train: Dict[str, np.ndarray]
    std_test: Dict[str, np.ndarray]
    preprocess: Dict[str, dict]
    cond_ewma_train: Dict[str, np.ndarray]
    cond_ewma_test: Dict[str, np.ndarray]
    # stacked
    X_train: np.ndarray
    X_test: np.ndarray
    tc_cond_ewma_train: np.ndarray
    tc_cond_ewma_test: np.ndarray
    ticker_ids_train: np.ndarray
    ticker_ids_test: np.ndarray
    mask_train: np.ndarray
    train_offsets: Dict[str, tuple]
    test_offsets: Dict[str, tuple]
    # housekeeping
    split_date: pd.Timestamp
    gap_end_date: pd.Timestamp


def compute_changes(rf_df: pd.DataFrame) -> pd.DataFrame:
    """Daily changes: log-diff for ATM features, simple-diff for the rest."""
    values = rf_df.values
    n_cols = values.shape[1]
    changes = np.full((len(values) - 1, n_cols), np.nan)
    for i, name in enumerate(rf_df.columns):
        if name in LOG_DIFF_FEATURES:
            with np.errstate(divide="ignore", invalid="ignore"):
                changes[:, i] = np.diff(np.log(np.maximum(values[:, i], 1e-10)))
        else:
            changes[:, i] = np.diff(values[:, i])
    return pd.DataFrame(changes, index=rf_df.index[1:], columns=rf_df.columns)


def build_validity_mask(changes_df: pd.DataFrame) -> np.ndarray:
    """True where finite and not stale (zero)."""
    vals = changes_df.values
    return np.isfinite(vals) & (vals != 0.0)


def load_vix(start: str = "2022-01-01", end: str = "2026-12-31") -> pd.Series:
    """Fetch ^VIX daily close from yfinance, tz-naive index."""
    df = yf.download("^VIX", start=start, end=end, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        s = df[("Close", "^VIX")]
    else:
        s = df["Close"]
    s.index = pd.to_datetime(s.index).tz_localize(None)
    s.name = "VIX"
    return s


def _ewma(arr: np.ndarray, alpha: float) -> np.ndarray:
    """Row-wise EWMA matching pandas adjust=False semantics."""
    if arr.ndim == 1:
        return pd.Series(arr).ewm(alpha=alpha, adjust=False).mean().values
    return pd.DataFrame(arr).ewm(alpha=alpha, adjust=False).mean().values


def _build_vix_ewma_cond(vix_all: np.ndarray) -> np.ndarray:
    """[VIX_level, EWMA_short(log-diff), EWMA_long(log-diff)] same-day, no lag."""
    log_vix = np.log(np.maximum(vix_all, 1e-10))
    ldiff1 = np.concatenate([[0.0], np.diff(log_vix)])
    return np.column_stack([
        vix_all, _ewma(ldiff1, EWMA_SHORT_ALPHA), _ewma(ldiff1, EWMA_LONG_ALPHA),
    ])


def _build_factor_ewma_cond(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Short + long EWMAs over standardized factors, LAGGED by 1 day."""
    _es = _ewma(X, EWMA_SHORT_ALPHA)
    _el = _ewma(X, EWMA_LONG_ALPHA)
    return (
        np.vstack([np.zeros((1, X.shape[1])), _es[:-1]]),
        np.vstack([np.zeros((1, X.shape[1])), _el[:-1]]),
    )


def prepare_data(tickers: list[str], cache_dir: Path) -> PreparedData:
    """Load cached risk factors, build everything the experiment needs."""
    # ---- Load cached risk factor levels ----
    rf_levels = {}
    for t in tickers:
        p = cache_dir / f"{t}_svi_risk_factors.parquet"
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Run scripts/prepare_data.py first "
                f"(or place the cached parquets in {cache_dir})."
            )
        rf_levels[t] = pd.read_parquet(p)

    # ---- Changes + validity mask ----
    changes = {t: compute_changes(rf_levels[t]) for t in tickers}
    masks = {t: build_validity_mask(changes[t]) for t in tickers}

    # ---- VIX ----
    vix_series = load_vix()
    vix = {
        t: vix_series.reindex(changes[t].index, method="ffill").ffill().bfill()
        for t in tickers
    }

    # ---- Train / test split with calendar gap ----
    all_dates = sorted(set().union(*[set(changes[t].index) for t in tickers]))
    split_date = all_dates[int(len(all_dates) * TRAIN_FRAC)]
    gap_end_date = split_date + pd.Timedelta(days=GAP_DAYS * 2)

    train_idx, test_idx = {}, {}
    for t in tickers:
        idx = changes[t].index
        train_idx[t] = idx <= split_date
        test_idx[t] = idx >= gap_end_date

    # ---- Winsorize, standardize (fit on train only) ----
    std_train, std_test, preprocess = {}, {}, {}
    for t in tickers:
        r_train = changes[t][train_idx[t]].values.copy()
        r_test = changes[t][test_idx[t]].values.copy()
        m_train = masks[t][train_idx[t]]

        masked_train = np.where(m_train, r_train, np.nan)
        lo = np.nanpercentile(masked_train, 1, axis=0)
        hi = np.nanpercentile(masked_train, 99, axis=0)
        r_train_w = np.clip(r_train, lo, hi)
        r_test_w = np.clip(r_test, lo, hi)

        masked_w = np.where(m_train, r_train_w, np.nan)
        mu = np.nanmean(masked_w, axis=0)
        sigma = np.nanstd(masked_w, axis=0)
        sigma = np.where(sigma < 1e-10, 1.0, sigma)

        std_train[t] = np.nan_to_num((r_train_w - mu) / sigma, nan=0.0)
        std_test[t] = np.nan_to_num((r_test_w - mu) / sigma, nan=0.0)
        preprocess[t] = {"mu": mu, "sigma": sigma, "lo": lo, "hi": hi}

    # ---- EWMA conditioning (19-dim per row: 3 VIX + 8 short + 8 long) ----
    cond_ewma_train, cond_ewma_test = {}, {}
    for t in tickers:
        vix_all = vix[t].values.astype(np.float64)
        vix_cond_all = _build_vix_ewma_cond(vix_all)
        vix_cond_train = vix_cond_all[train_idx[t]]
        vix_cond_test = vix_cond_all[test_idx[t]]

        e_short_tr, e_long_tr = _build_factor_ewma_cond(std_train[t])
        e_short_te, e_long_te = _build_factor_ewma_cond(std_test[t])

        cond_ewma_train[t] = np.hstack([vix_cond_train, e_short_tr, e_long_tr])
        cond_ewma_test[t] = np.hstack([vix_cond_test, e_short_te, e_long_te])

    # ---- Stacked arrays ----
    X_train = np.vstack([std_train[t] for t in tickers])
    X_test = np.vstack([std_test[t] for t in tickers])
    tc_cond_ewma_train = np.vstack([cond_ewma_train[t] for t in tickers])
    tc_cond_ewma_test = np.vstack([cond_ewma_test[t] for t in tickers])

    ticker_to_id = {t: i for i, t in enumerate(tickers)}
    ticker_ids_train = np.concatenate([
        np.full(train_idx[t].sum(), ticker_to_id[t], dtype=np.int64) for t in tickers
    ])
    ticker_ids_test = np.concatenate([
        np.full(test_idx[t].sum(), ticker_to_id[t], dtype=np.int64) for t in tickers
    ])
    mask_train = np.vstack([masks[t][train_idx[t]] for t in tickers]).astype(np.float32)

    train_offsets, test_offsets = {}, {}
    _off = 0
    for t in tickers:
        n = train_idx[t].sum()
        train_offsets[t] = (_off, _off + n); _off += n
    _off = 0
    for t in tickers:
        n = test_idx[t].sum()
        test_offsets[t] = (_off, _off + n); _off += n

    assert tc_cond_ewma_train.shape[1] == COND_DIM_EWMA
    assert X_train.shape[1] == N_RF

    return PreparedData(
        tickers=tickers,
        rf_levels=rf_levels, changes=changes, masks=masks, vix=vix,
        train_idx=train_idx, test_idx=test_idx,
        std_train=std_train, std_test=std_test, preprocess=preprocess,
        cond_ewma_train=cond_ewma_train, cond_ewma_test=cond_ewma_test,
        X_train=X_train, X_test=X_test,
        tc_cond_ewma_train=tc_cond_ewma_train, tc_cond_ewma_test=tc_cond_ewma_test,
        ticker_ids_train=ticker_ids_train, ticker_ids_test=ticker_ids_test,
        mask_train=mask_train,
        train_offsets=train_offsets, test_offsets=test_offsets,
        split_date=split_date, gap_end_date=gap_end_date,
    )
