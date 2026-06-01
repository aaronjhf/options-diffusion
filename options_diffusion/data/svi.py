"""SVI fitting from raw option quotes + risk-factor extraction."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from ..config import (
    M_GRID, N_DTE, N_MONEYNESS, SVI_RISK_FACTOR_NAMES, T_GRID,
    TRADING_DAYS_PER_YEAR,
)


# Raw SVI parameter bounds (a, b, rho, m, sigma)
_SVI_LB = np.array([0.0, 1e-6, -0.999, -2.0, 1e-4])
_SVI_UB = np.array([5.0, 5.0,   0.999,  2.0, 5.0])
_SVI_X0 = np.array([0.04, 0.2, -0.3, 0.0, 0.1])
MIN_QUOTES = 3


def svi_total_variance(k, a, b, rho, m, sigma):
    """Raw SVI total variance w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))."""
    k = np.asarray(k, dtype=np.float64)
    diff = k - m
    return a + b * (rho * diff + np.sqrt(diff**2 + sigma**2))


def _svi_vec(k, params):
    return svi_total_variance(k, *params)


def fit_svi_slice(k, iv, T, x0=None, rmse_retry=0.02):
    """Fit SVI to one expiry slice. Returns (params, rmse_vol, k_min, k_max)."""
    k = np.asarray(k, dtype=np.float64).ravel()
    iv = np.asarray(iv, dtype=np.float64).ravel()
    if len(k) < MIN_QUOTES:
        raise ValueError(f"Need >= {MIN_QUOTES} quotes, got {len(k)}")
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")
    w_obs = iv ** 2 * T
    warm = x0 is not None

    def _do_fit(init):
        init_c = np.clip(init, _SVI_LB + 1e-8, _SVI_UB - 1e-8)
        return least_squares(lambda p: _svi_vec(k, p) - w_obs,
                             init_c, bounds=(_SVI_LB, _SVI_UB), method="trf", max_nfev=2000)

    if x0 is None:
        atm_var = float(np.median(w_obs))
        init = _SVI_X0.copy()
        init[0] = max(atm_var * 0.5, 1e-6)
        init[1] = max(atm_var * 0.5, 1e-6)
    else:
        init = np.asarray(x0, dtype=np.float64).ravel()

    result = _do_fit(init)
    params = result.x
    w_fit = _svi_vec(k, params)
    iv_fit = np.sqrt(np.maximum(w_fit / T, 0.0))
    rmse = float(np.sqrt(np.mean((iv_fit - iv) ** 2)))

    if warm and rmse > rmse_retry:
        atm_var = float(np.median(w_obs))
        retry_init = _SVI_X0.copy()
        retry_init[0] = max(atm_var * 0.5, 1e-6)
        retry_init[1] = max(atm_var * 0.5, 1e-6)
        r2 = _do_fit(retry_init)
        w2 = _svi_vec(k, r2.x)
        iv2 = np.sqrt(np.maximum(w2 / T, 0.0))
        rmse2 = float(np.sqrt(np.mean((iv2 - iv) ** 2)))
        if rmse2 < rmse:
            params, rmse = r2.x, rmse2

    return params, rmse, float(k.min()), float(k.max())


def evaluate_surface_point(k_target, T_target, expiry_params):
    """Evaluate interpolated SVI surface at (k, T). Returns (iv, extrapolated)."""
    slices = sorted(expiry_params, key=lambda x: x[0])
    Ts = np.array([s[0] for s in slices])
    T_min, T_max = Ts[0], Ts[-1]
    k_arr = np.array([k_target], dtype=np.float64)

    if len(slices) == 1:
        T_s, p = slices[0]
        if T_target < 0.5 * T_s or T_target > 1.5 * T_s:
            return np.nan, True
        w = _svi_vec(k_arr, p) * (T_target / T_s)
        return float(np.sqrt(max(w[0] / T_target, 0.0))), not math.isclose(T_target, T_s, rel_tol=1e-9)

    if T_target < T_min:
        if T_target < 0.5 * T_min:
            return np.nan, True
        w = _svi_vec(k_arr, slices[0][1]) * (T_target / T_min)
        return float(np.sqrt(max(w[0] / T_target, 0.0))), True

    if T_target > T_max:
        if T_target > 1.5 * T_max:
            return np.nan, True
        w = _svi_vec(k_arr, slices[-1][1]) * (T_target / T_max)
        return float(np.sqrt(max(w[0] / T_target, 0.0))), True

    idx_hi = int(np.searchsorted(Ts, T_target, side="left"))
    if math.isclose(T_target, Ts[idx_hi], rel_tol=1e-9):
        w = _svi_vec(k_arr, slices[idx_hi][1])
        return float(np.sqrt(max(w[0] / T_target, 0.0))), False
    if idx_hi == 0:
        idx_hi = 1
    idx_lo = idx_hi - 1
    T_lo, p_lo = slices[idx_lo]
    T_hi, p_hi = slices[idx_hi]
    alpha = (T_target - T_lo) / (T_hi - T_lo)
    w_interp = (1 - alpha) * _svi_vec(k_arr, p_lo) + alpha * _svi_vec(k_arr, p_hi)
    return float(np.sqrt(max(w_interp[0] / T_target, 0.0))), False


def evaluate_surface_grid(expiry_params):
    """Evaluate SVI surface on the standard (M_GRID, T_GRID) grid."""
    n_total = N_MONEYNESS * N_DTE
    iv_vec = np.full(n_total, np.nan)
    mask = np.ones(n_total, dtype=bool)
    k_values = np.log(M_GRID)
    for j, dte in enumerate(T_GRID):
        T = float(dte) / TRADING_DAYS_PER_YEAR
        for i, k_val in enumerate(k_values):
            idx = j * N_MONEYNESS + i
            try:
                iv, extrap = evaluate_surface_point(float(k_val), T, expiry_params)
                iv_vec[idx] = iv
                mask[idx] = not extrap
            except (ValueError, Exception):
                iv_vec[idx] = np.nan
                mask[idx] = False
    return iv_vec, mask


def extract_risk_factors(iv_vector):
    """Extract 8 risk factors from a single-day IV grid."""
    grid = iv_vector.reshape(N_DTE, N_MONEYNESS)

    atm_candidates = np.where(np.abs(M_GRID - 1.0) < 0.015)[0]
    if len(atm_candidates) == 0:
        atm_candidates = np.array([np.argmin(np.abs(M_GRID - 1.0))])

    def atm(dte_idx):
        vals = [grid[dte_idx, c] for c in atm_candidates if np.isfinite(grid[dte_idx, c])]
        return float(np.mean(vals)) if vals else np.nan

    put_idx = np.argmin(np.abs(M_GRID - 0.95))
    call_idx = np.argmin(np.abs(M_GRID - 1.05))

    dte_6w = np.argmin(np.abs(T_GRID - 42))
    dte_3m = np.argmin(np.abs(T_GRID - 90))
    dte_6m = np.argmin(np.abs(T_GRID - 180))

    atm_6w = atm(dte_6w)
    atm_3m = atm(dte_3m)
    atm_6m = atm(dte_6m)

    def _skew(dte_idx, atm_val):
        v = grid[dte_idx, put_idx]
        return float(v) - atm_val if np.isfinite(v) else np.nan

    skew_6w = _skew(dte_6w, atm_6w)
    skew_3m = _skew(dte_3m, atm_3m)
    skew_6m = _skew(dte_6m, atm_6m)

    term_slope = atm_6m - atm_6w if np.isfinite(atm_6m) and np.isfinite(atm_6w) else np.nan

    curvature_3m = np.nan
    if np.isfinite(grid[dte_3m, put_idx]) and np.isfinite(grid[dte_3m, call_idx]):
        curvature_3m = (float(grid[dte_3m, put_idx]) + float(grid[dte_3m, call_idx])) / 2.0 - atm_3m

    return {
        "atm_6w": atm_6w, "atm_3m": atm_3m, "atm_6m": atm_6m,
        "skew_6w": skew_6w, "skew_3m": skew_3m, "skew_6m": skew_6m,
        "term_slope": term_slope, "curvature_3m": curvature_3m,
    }


def fit_svi_from_quotes(ticker: str, quotes_df: pd.DataFrame, cache_dir: Path,
                        force: bool = False, verbose: bool = True):
    """Fit SVI per (date, expiry), evaluate on grid, extract risk factors.

    Caches three artifacts per ticker:
      - {ticker}_svi_risk_factors.parquet  (n_days, 8)
      - {ticker}_iv_surface.parquet        (n_days, N_DTE*N_MONEYNESS)
      - {ticker}_obs_mask.npy              (n_days, N_DTE*N_MONEYNESS)
    """
    rf_cache = cache_dir / f"{ticker}_svi_risk_factors.parquet"
    iv_cache = cache_dir / f"{ticker}_iv_surface.parquet"
    mask_cache = cache_dir / f"{ticker}_obs_mask.npy"

    if not force and rf_cache.exists() and iv_cache.exists() and mask_cache.exists():
        if verbose:
            print(f"{ticker}: loading cached SVI risk factors")
        return (
            pd.read_parquet(rf_cache),
            pd.read_parquet(iv_cache),
            np.load(mask_cache),
        )

    if verbose:
        print(f"{ticker}: fitting SVI from raw quotes...")
    grid_cols = [f"M{M_GRID[i]:.3f}_T{T_GRID[j]}"
                 for j in range(N_DTE) for i in range(N_MONEYNESS)]

    trading_dates = sorted(quotes_df["date"].unique())
    all_rf, all_iv_vecs, all_masks, daily_rmses = [], [], [], []
    prev_params = {}

    for day in trading_dates:
        day_quotes = quotes_df[quotes_df["date"] == day]
        expiry_params, new_params, successful_fits = [], {}, []

        for exp_date, exp_group in day_quotes.groupby("expiration"):
            k_vals = np.log(exp_group["moneyness"].values)
            iv_vals = exp_group["iv"].values
            T_yrs = exp_group["T_years"].iloc[0]
            if len(k_vals) < MIN_QUOTES:
                continue
            x0 = prev_params.get(exp_date)
            try:
                params, rmse_vol, k_min, k_max = fit_svi_slice(k_vals, iv_vals, T_yrs, x0=x0)
                expiry_params.append((T_yrs, params))
                new_params[exp_date] = params
                successful_fits.append((T_yrs, rmse_vol, k_min, k_max))
            except (ValueError, RuntimeError):
                continue

        prev_params.update(new_params)

        if not expiry_params:
            all_rf.append({n: np.nan for n in SVI_RISK_FACTOR_NAMES})
            all_iv_vecs.append(np.full(N_DTE * N_MONEYNESS, np.nan))
            all_masks.append(np.zeros(N_DTE * N_MONEYNESS, dtype=bool))
            daily_rmses.append(np.nan)
            continue

        iv_vec, mask = evaluate_surface_grid(expiry_params)
        all_iv_vecs.append(iv_vec)
        all_masks.append(mask)
        daily_rmses.append(np.mean([r for _, r, _, _ in successful_fits]))

        if np.all(np.isnan(iv_vec)):
            all_rf.append({n: np.nan for n in SVI_RISK_FACTOR_NAMES})
            continue
        all_rf.append(extract_risk_factors(iv_vec))

    rf_df = pd.DataFrame(all_rf, index=pd.to_datetime(trading_dates),
                         columns=SVI_RISK_FACTOR_NAMES)
    rf_df = rf_df.ffill(limit=2).bfill(limit=1)
    n_before_drop = len(rf_df)
    rf_df = rf_df.dropna()
    rf_df.to_parquet(rf_cache)

    iv_surface_df = pd.DataFrame(
        np.array(all_iv_vecs), index=pd.to_datetime(trading_dates), columns=grid_cols)
    iv_surface_df.to_parquet(iv_cache)

    mask_arr = np.array(all_masks)
    np.save(mask_cache, mask_arr)

    if verbose:
        rmses_clean = [r for r in daily_rmses if not np.isnan(r)]
        print(f"  {ticker}: {len(rf_df)} days of risk factors")
        if rmses_clean:
            print(f"    RMSE: mean={np.mean(rmses_clean):.4f}  max={np.max(rmses_clean):.4f}")
        print(f"    Days dropped by dropna: {n_before_drop - len(rf_df)}")

    return rf_df, iv_surface_df, mask_arr
