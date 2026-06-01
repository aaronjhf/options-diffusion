"""Statistical baselines: PCA Nadaraya-Watson bootstrap, conditional t-copula."""
from __future__ import annotations

import numpy as np
import scipy.stats as _stats
from scipy.optimize import minimize_scalar
from scipy.special import gammaln
from sklearn.decomposition import PCA


def silverman_bw(v: np.ndarray) -> float:
    """Silverman's rule-of-thumb bandwidth for 1-D kernel density."""
    n = len(v)
    s = np.std(v, ddof=1)
    iqr = np.subtract(*np.percentile(v, [75, 25]))
    spread = min(s, iqr / 1.34) if iqr > 0 else s
    if spread <= 0:
        spread = 1.0
    return 0.9 * spread * n ** (-0.2)


def multivariate_kernel_weights(c_query: np.ndarray, c_train: np.ndarray, bw: np.ndarray) -> np.ndarray:
    """Gaussian product-kernel weights from query point to each training row."""
    diff = (c_train - c_query) / bw
    log_w = -0.5 * np.sum(diff ** 2, axis=1)
    log_w -= log_w.max()
    w = np.exp(log_w)
    total = w.sum()
    return w / total if total > 0 else np.ones(len(w)) / len(w)


def _weighted_quantile(values, weights, q):
    idx = np.argsort(values)
    cum_w = np.cumsum(weights[idx])
    cum_w /= cum_w[-1]
    return np.interp(q, cum_w, values[idx])


class CondPCABootstrap:
    """PCA + Nadaraya-Watson kernel bootstrap on the conditioning vector."""

    def __init__(self, n_components=0.95, bandwidth=None, verbose=False):
        self.n_components = n_components
        self.bandwidth = bandwidth
        self.verbose = verbose

    def fit(self, X_train, cond_train):
        self.pca = PCA(n_components=self.n_components)
        self.Z_train = self.pca.fit_transform(X_train)
        self.cond_train = np.asarray(cond_train, dtype=np.float64)
        if self.cond_train.ndim == 1:
            self.cond_train = self.cond_train[:, None]
        D = self.cond_train.shape[1]
        if self.bandwidth is not None:
            self.bw = np.broadcast_to(np.asarray(self.bandwidth, dtype=np.float64), (D,)).copy()
        else:
            self.bw = np.array([silverman_bw(self.cond_train[:, d]) for d in range(D)])
        if self.verbose:
            print(f"  PCA: {self.pca.n_components_} components, "
                  f"{self.pca.explained_variance_ratio_.sum()*100:.1f}% var, "
                  f"bw={np.array2string(self.bw, precision=4)}")
        return self

    def sample(self, cond, seed=None):
        cond = np.asarray(cond, dtype=np.float64)
        if cond.ndim == 1:
            cond = cond[:, None]
        rng = np.random.RandomState(seed)
        out_pca = np.empty((len(cond), self.Z_train.shape[1]))
        for i in range(len(cond)):
            w = multivariate_kernel_weights(cond[i], self.cond_train, self.bw)
            idx = rng.choice(len(self.Z_train), p=w)
            out_pca[i] = self.Z_train[idx]
        return self.pca.inverse_transform(out_pca)


class CondTCopula:
    """VIX-regime t-copula with NW conditional marginals (full kernel on cond)."""

    def __init__(self, verbose=False):
        self.verbose = verbose

    def fit(self, X_train, cond_train, n_bins=3):
        self.X_tr = np.asarray(X_train, float)
        self.c_tr = np.asarray(cond_train, float)
        d = self.X_tr.shape[1]
        self.bw = np.array([silverman_bw(self.c_tr[:, k])
                            for k in range(self.c_tr.shape[1])])

        vix = self.c_tr[:, 0]
        self.edges = np.quantile(vix, np.linspace(0, 1, n_bins + 1))
        self.edges[0] -= 1e-9
        self.n_bins = n_bins
        self.R, self.df = [], []

        for b in range(n_bins):
            mask = (vix > self.edges[b]) & (vix <= self.edges[b + 1])
            Xb = self.X_tr[mask]
            u = np.array([_stats.rankdata(Xb[:, j], "average") / (len(Xb) + 1)
                          for j in range(d)]).T
            Z = _stats.norm.ppf(u)
            R = np.corrcoef(Z.T)
            R = (R + R.T) / 2
            np.fill_diagonal(R, 1.0)
            self.R.append(R)
            try:
                R_inv = np.linalg.inv(R)
                sign, ldet = np.linalg.slogdet(R)
                nb_ = len(Z)

                def neg_ll(nu, _Z=Z, _R_inv=R_inv, _ldet=ldet, _nb=nb_, _d=d):
                    quad = np.einsum("ni,ij,nj->n", _Z, _R_inv, _Z)
                    ll = _nb * (gammaln((nu + _d) / 2) - gammaln(nu / 2)
                                - (_d / 2) * np.log(nu * np.pi) - 0.5 * _ldet)
                    ll -= ((nu + _d) / 2) * np.sum(np.log(1 + quad / nu))
                    return -ll

                res = minimize_scalar(neg_ll, bounds=(2.0, 50.0), method="bounded")
                self.df.append(float(res.x) if res.success else 5.0)
            except Exception:
                self.df.append(5.0)
        if self.verbose:
            print(f'  t-Copula fitted: {n_bins} VIX bins, df={[f"{v:.1f}" for v in self.df]}')
        return self

    def sample(self, cond_test, seed=None):
        rng = np.random.default_rng(seed)
        c_te = np.asarray(cond_test, float)
        n, d = len(c_te), self.X_tr.shape[1]
        out = np.empty((n, d))
        vix = c_te[:, 0]

        for i in range(n):
            b = min(np.searchsorted(self.edges[1:], vix[i]), self.n_bins - 1)
            nu, R = self.df[b], self.R[b]
            L = np.linalg.cholesky(R)
            z_norm = rng.standard_normal(d) @ L.T
            chi2 = rng.chisquare(nu)
            z_t = z_norm * np.sqrt(nu / chi2)
            u = _stats.t.cdf(z_t, df=nu)

            w = multivariate_kernel_weights(c_te[i], self.c_tr, self.bw)
            for j in range(d):
                out[i, j] = _weighted_quantile(self.X_tr[:, j], w, u[j])
        return out
