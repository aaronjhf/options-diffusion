"""Distributional metrics: SWD, MMD2, Frechet, KS, CorrDist, Energy."""
from __future__ import annotations

import numpy as np
from scipy import linalg as la
from scipy.spatial.distance import cdist, pdist
from scipy.stats import ks_2samp, wasserstein_distance


METRIC_KEYS = ["SWD", "MMD2", "Frechet", "KS_max", "CorrDist", "Energy"]


def sliced_wasserstein(x, y, n_proj=200, seed=0):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    rng = np.random.default_rng(seed)
    d = x.shape[1]
    dirs = rng.standard_normal((n_proj, d))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    return float(np.mean([wasserstein_distance(x @ v, y @ v) for v in dirs]))


def frechet_dist(x, y, eps=1e-6):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    mu_x, mu_y = x.mean(0), y.mean(0)
    d = x.shape[1]
    cx = np.cov(x, rowvar=False) + eps * np.eye(d)
    cy = np.cov(y, rowvar=False) + eps * np.eye(d)
    diff = mu_x - mu_y
    sq = np.real(la.sqrtm(cx @ cy))
    return float(diff @ diff + max(np.trace(cx + cy - 2 * sq), 0.0))


def marginal_ks(x, y):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    stats = [ks_2samp(x[:, j], y[:, j]) for j in range(x.shape[1])]
    return {
        "max_stat": max(s.statistic for s in stats),
        "min_pval": min(s.pvalue for s in stats),
    }


def corr_dist(x, y):
    cx = np.atleast_2d(np.corrcoef(x, rowvar=False))
    cy = np.atleast_2d(np.corrcoef(y, rowvar=False))
    return float(np.linalg.norm(cx - cy, "fro"))


def energy_dist(X, Y):
    X, Y = np.asarray(X, np.float64), np.asarray(Y, np.float64)

    def mpd(A, B):
        return float(np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)).mean())

    return max(2 * mpd(X, Y) - mpd(X, X) - mpd(Y, Y), 0.0)


def mmd2(x, y, bw=None):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    if bw is None:
        pw = pdist(np.vstack([x, y]), "euclidean")
        nz = pw[pw > 0]
        if len(nz) == 0:
            return 0.0
        bw = float(np.median(nz))
    gamma = 1.0 / (2.0 * bw ** 2)

    def rbf(a, b):
        return np.exp(-gamma * cdist(a, b, "sqeuclidean"))

    kxx = rbf(x, x); np.fill_diagonal(kxx, 0)
    kyy = rbf(y, y); np.fill_diagonal(kyy, 0)
    kxy = rbf(x, y)
    n, m = len(x), len(y)
    return float(kxx.sum() / (n * (n - 1)) + kyy.sum() / (m * (m - 1)) - 2 * kxy.sum() / (n * m))


def compute_all_metrics(real, gen):
    return {
        "SWD": sliced_wasserstein(real, gen),
        "MMD2": mmd2(real, gen),
        "Frechet": frechet_dist(real, gen),
        "KS_max": marginal_ks(real, gen)["max_stat"],
        "CorrDist": corr_dist(real, gen),
        "Energy": energy_dist(real, gen),
    }
