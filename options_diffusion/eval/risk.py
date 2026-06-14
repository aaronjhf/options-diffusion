"""Ensemble-based risk evaluation: VaR backtests with expected shortfall,
and a mean-reversion trading rule.

Unlike `metrics.py` (which compares one generated draw per test day against
the realized set in aggregate), everything here operates on a per-day
predictive *ensemble* of shape (n_days, n_draws) for a single factor, scored
against the realized 1-D series. This is what allows day-by-day VaR
violations and CDF-threshold trading rules.

Realized values should be standardized with train stats but NOT winsorized:
clipping the test tails at train percentiles corrupts tail backtests.
Days flagged invalid (stale/missing SVI fits) are excluded from scoring but
kept in the index so day-(t+1) lookups stay aligned.
"""
from __future__ import annotations

import numpy as np

VAR_ALPHAS = (0.01, 0.05, 0.10)
MR_TAUS = (0.80, 0.90, 0.95)

_EPS = 1e-12


def inflate_spread(samples: np.ndarray, S: float) -> np.ndarray:
    """Scale each day's ensemble around its own mean by factor S.

    Variance recalibration: returns m + S*(samples - m) with m the per-day
    ensemble mean. S > 1 widens an over-confident (under-dispersed) predictive
    distribution; S < 1 tightens a conservative one. The per-day mean (the
    conditional forecast) is preserved, only the spread is rescaled. Identity
    when S == 1, so callers can apply it unconditionally.

    Applying one transform here keeps VaR, ES, PIT, and the mean-reversion
    signal mutually consistent (they all read the same recalibrated ensemble).
    """
    if S == 1.0:
        return samples
    m = samples.mean(axis=1, keepdims=True)
    return m + S * (samples - m)


def dispersion_ratio(samples: np.ndarray, realized: np.ndarray,
                     valid: np.ndarray) -> float:
    """Inflation factor that would calibrate the ensemble spread.

    Std of the standardized residual z = (realized - ensemble_mean)/ensemble_std
    over valid days. A perfectly calibrated ensemble gives ~1; > 1 means the
    predictive distribution is too narrow (realized scatters wider than the
    ensemble claims) and should be inflated by roughly this factor.
    """
    m = samples.mean(axis=1)
    sd = samples.std(axis=1)
    z = (realized - m) / np.where(sd < _EPS, np.nan, sd)
    z = z[valid]
    z = z[np.isfinite(z)]
    return float(np.std(z)) if len(z) else np.nan


def var_backtest(samples: np.ndarray, realized: np.ndarray, valid: np.ndarray,
                 alpha: float, tail: str) -> dict:
    """Backtest VaR at level `alpha` for one tail ('lower' or 'upper').

    samples: (n_days, n_draws) predictive ensemble.

    The model's VaR for each day is the alpha (or 1-alpha) quantile of that
    day's ensemble; a violation is a realized move beyond it. A correct model
    violates on an alpha fraction of days. Expected shortfall (ES) checks the
    *size* of tail losses: on violation days, the average realized exceedance
    is compared with the model's own conditional tail expectation
    (es_ratio > 1 means real tail losses are bigger than the model says).
    """
    q_level = alpha if tail == "lower" else 1.0 - alpha
    var_q = np.quantile(samples, q_level, axis=1)
    if tail == "lower":
        viol = realized < var_q
        tail_mask = samples <= var_q[:, None]
    else:
        viol = realized > var_q
        tail_mask = samples >= var_q[:, None]

    viol = viol & valid
    n_obs = int(valid.sum())
    n_viol = int(viol.sum())

    # Per-day model ES (conditional tail expectation), compared on violation days.
    tail_vals = np.where(tail_mask, samples, np.nan)
    with np.errstate(invalid="ignore"):
        model_es_daily = np.nanmean(tail_vals, axis=1)
    model_es_daily = np.where(np.isnan(model_es_daily), var_q, model_es_daily)
    if n_viol > 0:
        realized_es = float(realized[viol].mean())
        model_es = float(model_es_daily[viol].mean())
        es_ratio = realized_es / model_es if abs(model_es) > _EPS else np.nan
    else:
        realized_es, model_es, es_ratio = np.nan, np.nan, np.nan

    return {
        "tail": tail, "alpha": alpha,
        "n_obs": n_obs, "n_viol": n_viol,
        "viol_rate": n_viol / n_obs if n_obs else np.nan,
        "realized_es": realized_es, "model_es": model_es, "es_ratio": es_ratio,
    }


def pit_from_ensemble(samples: np.ndarray, realized: np.ndarray) -> np.ndarray:
    """Position of each realized move in its day's predictive CDF, in (0, 1).

    This is the quantity the mean-reversion rule thresholds: a value near 1
    means the realized move was larger than almost all of the model's draws.
    """
    less = (samples < realized[:, None]).sum(axis=1)
    equal = (samples == realized[:, None]).sum(axis=1)
    return (less + 0.5 * equal + 0.5) / (samples.shape[1] + 1.0)


def mean_reversion_eval(pit: np.ndarray, realized: np.ndarray,
                        pred_mean: np.ndarray, valid: np.ndarray,
                        tau: float) -> tuple[dict, dict]:
    """Evaluate a CDF-threshold mean-reversion rule.

    Signal: on day t, if the realized move sits above the model's tau
    predictive quantile (pit >= tau), go short the factor change for day t+1;
    if below the (1-tau) quantile, go long. PnL is in train-sigma units of
    the factor change.

    Two distinct questions are scored:
      * strategy: does the realized t+1 move actually revert? (hit_rate,
        mean_pnl, t-stat)
      * model:    does the model's own conditional mean for t+1 predict that
        reversion? (model_agree_frac = model mean points in the reversion
        direction; model_dir_acc = model mean sign matches realized sign)

    Returns (summary dict, per-signal arrays dict).
    """
    ok = valid[:-1] & valid[1:]
    upper = (pit[:-1] >= tau) & ok
    lower = (pit[:-1] <= 1.0 - tau) & ok
    direction = np.where(upper, -1.0, np.where(lower, 1.0, 0.0))
    sig = direction != 0.0

    d = direction[sig]
    nxt = realized[1:][sig]
    pnl = d * nxt
    nz = nxt != 0.0
    hit_rate = float((np.sign(nxt[nz]) == d[nz]).mean()) if nz.any() else np.nan

    mu_next = pred_mean[1:][sig]
    model_agree = float((np.sign(mu_next) == d).mean()) if sig.any() else np.nan
    nz2 = nz & (mu_next != 0.0)
    model_dir_acc = (float((np.sign(mu_next[nz2]) == np.sign(nxt[nz2])).mean())
                     if nz2.any() else np.nan)

    n_sig = int(sig.sum())
    if n_sig > 1 and pnl.std(ddof=1) > _EPS:
        tstat = float(pnl.mean() / (pnl.std(ddof=1) / np.sqrt(n_sig)))
    else:
        tstat = np.nan

    summary = {
        "tau": tau, "n_signals": n_sig,
        "n_days": int(ok.sum()),
        "hit_rate": hit_rate,
        "mean_pnl": float(pnl.mean()) if n_sig else np.nan,
        "total_pnl": float(pnl.sum()) if n_sig else 0.0,
        "pnl_tstat": tstat,
        "model_agree_frac": model_agree,
        "model_dir_acc": model_dir_acc,
    }
    per_signal = {
        "t_index": np.flatnonzero(sig),
        "direction": d,
        "next_ret": nxt,
        "pnl": pnl,
        "model_pred_mean_next": mu_next,
    }
    return summary, per_signal
