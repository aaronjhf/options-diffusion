"""Conditional and ticker-conditioned DDPMs (FiLM blocks, cosine LR, EMA)."""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import (
    CLIP_RANGE, C_DIM, DEVICE, DROPOUT, EMA_DECAY, HIDDEN_DIMS, LR, LR_MIN,
    N_TIMESTEPS, T_DIM, TICKER_EMBED_DIM,
)
from .nets import CondBlock, SinusoidalEmbedding


def cosine_beta_schedule(T: int, s: float = 0.008) -> np.ndarray:
    steps = np.arange(T + 1, dtype=np.float64)
    f = np.cos((steps / T + s) / (1 + s) * np.pi / 2) ** 2
    alpha_bar = f / f[0]
    betas = 1.0 - alpha_bar[1:] / np.maximum(alpha_bar[:-1], 1e-10)
    return np.clip(betas, 0.0001, 0.999)


class ConditionalNoiseNet(nn.Module):
    """Noise predictor conditioned on (timestep, exogenous cond)."""

    def __init__(self, data_dim, cond_input_dim=3, hidden_dims=HIDDEN_DIMS,
                 t_dim=T_DIM, c_dim=C_DIM, dropout=DROPOUT):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(t_dim),
            nn.Linear(t_dim, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim),
        )
        self.cond_embed = nn.Sequential(
            nn.Linear(cond_input_dim, c_dim), nn.SiLU(),
            nn.Linear(c_dim, c_dim), nn.SiLU(),
            nn.Linear(c_dim, c_dim),
        )
        dims = [data_dim] + list(hidden_dims)
        self.blocks = nn.ModuleList([
            CondBlock(dims[i], dims[i + 1], t_dim, c_dim, dropout)
            for i in range(len(dims) - 1)
        ])
        self.out_proj = nn.Linear(hidden_dims[-1], data_dim)

    def forward(self, x, t, cond):
        t_emb = self.time_embed(t)
        c_emb = self.cond_embed(cond)
        h = x
        for block in self.blocks:
            h = block(h, t_emb, c_emb)
        return self.out_proj(h)


class TickerCondNoiseNet(nn.Module):
    """Noise predictor with both exogenous cond and a learnable ticker embedding."""

    def __init__(self, data_dim, n_tickers, cond_input_dim=3, hidden_dims=HIDDEN_DIMS,
                 t_dim=T_DIM, c_dim=C_DIM, ticker_embed_dim=TICKER_EMBED_DIM, dropout=DROPOUT):
        super().__init__()
        self.ticker_embed = nn.Embedding(n_tickers, ticker_embed_dim)
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(t_dim),
            nn.Linear(t_dim, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim),
        )
        self.cond_embed = nn.Sequential(
            nn.Linear(cond_input_dim, c_dim), nn.SiLU(),
            nn.Linear(c_dim, c_dim), nn.SiLU(),
            nn.Linear(c_dim, c_dim),
        )
        c_dim_total = c_dim + ticker_embed_dim
        dims = [data_dim] + list(hidden_dims)
        self.blocks = nn.ModuleList([
            CondBlock(dims[i], dims[i + 1], t_dim, c_dim_total, dropout)
            for i in range(len(dims) - 1)
        ])
        self.out_proj = nn.Linear(hidden_dims[-1], data_dim)

    def forward(self, x, t, cond, ticker_ids):
        t_emb = self.time_embed(t)
        c_emb = self.cond_embed(cond)
        tk_emb = self.ticker_embed(ticker_ids)
        c_full = torch.cat([c_emb, tk_emb], dim=-1)
        h = x
        for block in self.blocks:
            h = block(h, t_emb, c_full)
        return self.out_proj(h)


class _DDPMBase:
    """Shared diffusion buffers + helpers."""

    def __init__(self, data_dim, n_timesteps=N_TIMESTEPS, clip_range=CLIP_RANGE,
                 lr=LR, lr_min=LR_MIN):
        self.data_dim = data_dim
        self.n_timesteps = n_timesteps
        self.clip_range = clip_range
        self.lr = lr
        self.lr_min = lr_min

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1.0 - betas
        alpha_bar = np.cumprod(alphas)
        alpha_bar = np.clip(alpha_bar, 1e-10, 1.0)
        alpha_bar_prev = np.append(1.0, alpha_bar[:-1])

        self.sqrt_ab = torch.tensor(np.sqrt(alpha_bar), dtype=torch.float32, device=DEVICE)
        self.sqrt_1m_ab = torch.tensor(np.sqrt(1.0 - alpha_bar), dtype=torch.float32, device=DEVICE)
        self.post_var = torch.tensor(
            betas * (1 - alpha_bar_prev) / np.maximum(1 - alpha_bar, 1e-10),
            dtype=torch.float32, device=DEVICE,
        )
        self.post_coef1 = torch.tensor(
            betas * np.sqrt(alpha_bar_prev) / np.maximum(1 - alpha_bar, 1e-10),
            dtype=torch.float32, device=DEVICE,
        )
        self.post_coef2 = torch.tensor(
            (1 - alpha_bar_prev) * np.sqrt(alphas) / np.maximum(1 - alpha_bar, 1e-10),
            dtype=torch.float32, device=DEVICE,
        )

        self.cond_mean = None
        self.cond_std = None
        self.ema_decay = EMA_DECAY

    def _ext(self, buf, t, shape):
        return buf.gather(0, t).view(-1, *([1] * (len(shape) - 1)))

    def q_sample(self, x0, t, noise):
        return self._ext(self.sqrt_ab, t, x0.shape) * x0 + self._ext(self.sqrt_1m_ab, t, x0.shape) * noise

    def _fit_cond_stats(self, cond):
        cond = np.asarray(cond, dtype=np.float64)
        if cond.ndim == 1:
            cond = cond[:, None]
        self.cond_mean = cond.mean(axis=0)
        self.cond_std = cond.std(axis=0)
        self.cond_std = np.where(self.cond_std < 1e-8, 1.0, self.cond_std)
        return cond

    def _normalize_cond(self, cond):
        cond = np.asarray(cond, dtype=np.float64)
        if cond.ndim == 1:
            cond = cond[:, None]
        return (cond - self.cond_mean) / self.cond_std


class ConditionalDDPM(_DDPMBase):
    """Per-ticker (solo) conditional diffusion."""

    def __init__(self, data_dim, cond_input_dim=3, n_timesteps=N_TIMESTEPS,
                 hidden_dims=HIDDEN_DIMS, t_dim=T_DIM, c_dim=C_DIM,
                 dropout=DROPOUT, lr=LR, lr_min=LR_MIN, clip_range=CLIP_RANGE):
        super().__init__(data_dim, n_timesteps, clip_range, lr, lr_min)
        self.cond_input_dim = cond_input_dim
        self.net = ConditionalNoiseNet(
            data_dim, cond_input_dim, hidden_dims, t_dim, c_dim, dropout
        ).to(DEVICE)
        self.ema_state = copy.deepcopy(self.net.state_dict())

    def train_model(self, X, cond, mask=None, epochs=6000, batch_size=64, verbose=True):
        cond = self._fit_cond_stats(cond)
        N = X.shape[0]
        X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        cond_t = torch.tensor((cond - self.cond_mean) / self.cond_std,
                              dtype=torch.float32, device=DEVICE)
        mask_t = torch.tensor(mask, dtype=torch.float32, device=DEVICE) if mask is not None else None

        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=self.lr_min)
        self.net.train()
        losses = []

        for ep in range(epochs):
            perm = torch.randperm(N, device=DEVICE)
            ep_loss = 0.0; nb = 0
            for s in range(0, N, batch_size):
                idx = perm[s:s + batch_size]
                x0 = X_t[idx]; cb = cond_t[idx]
                t = torch.randint(0, self.n_timesteps, (len(idx),), device=DEVICE)
                noise = torch.randn_like(x0)
                x_noisy = self.q_sample(x0, t, noise)
                pred = self.net(x_noisy, t, cb)
                if mask_t is not None:
                    m = mask_t[idx]
                    loss = ((pred - noise) ** 2 * m).sum() / m.sum().clamp(min=1)
                else:
                    loss = F.mse_loss(pred, noise)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                ep_loss += loss.item(); nb += 1
            sched.step()
            with torch.no_grad():
                net_sd = self.net.state_dict()
                for k in self.ema_state:
                    self.ema_state[k] = self.ema_decay * self.ema_state[k] + (1 - self.ema_decay) * net_sd[k]
            avg = ep_loss / max(nb, 1)
            losses.append(avg)
            if verbose and (ep + 1) % max(1, epochs // 10) == 0:
                cur_lr = sched.get_last_lr()[0]
                print(f"  epoch {ep+1:5d}/{epochs}  loss={avg:.6f}  lr={cur_lr:.2e}")
        self.net.load_state_dict(self.ema_state)
        self.net.eval()
        return losses

    @torch.no_grad()
    def sample(self, cond, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        self.net.eval()
        cond = self._normalize_cond(cond)
        n = len(cond)
        cond_t = torch.tensor(cond, dtype=torch.float32, device=DEVICE)
        x = torch.randn(n, self.data_dim, device=DEVICE)
        for i in reversed(range(self.n_timesteps)):
            t = torch.full((n,), i, device=DEVICE, dtype=torch.long)
            pred = self.net(x, t, cond_t)
            x0_pred = (x - self._ext(self.sqrt_1m_ab, t, x.shape) * pred) / self._ext(self.sqrt_ab, t, x.shape)
            x0_pred = x0_pred.clamp(*self.clip_range)
            mean = self._ext(self.post_coef1, t, x.shape) * x0_pred + self._ext(self.post_coef2, t, x.shape) * x
            if i > 0:
                x = mean + torch.sqrt(self._ext(self.post_var, t, x.shape)) * torch.randn_like(x)
            else:
                x = mean
        return x.cpu().numpy()


class TickerCondDDPM(_DDPMBase):
    """Pooled diffusion sharing weights across tickers via a learned ticker embedding."""

    def __init__(self, data_dim, n_tickers, cond_input_dim=3, n_timesteps=N_TIMESTEPS,
                 hidden_dims=HIDDEN_DIMS, t_dim=T_DIM, c_dim=C_DIM,
                 ticker_embed_dim=TICKER_EMBED_DIM, dropout=DROPOUT,
                 lr=LR, lr_min=LR_MIN, clip_range=CLIP_RANGE):
        super().__init__(data_dim, n_timesteps, clip_range, lr, lr_min)
        self.n_tickers = n_tickers
        self.cond_input_dim = cond_input_dim
        self.net = TickerCondNoiseNet(
            data_dim, n_tickers, cond_input_dim, hidden_dims,
            t_dim, c_dim, ticker_embed_dim, dropout,
        ).to(DEVICE)
        self.ema_state = copy.deepcopy(self.net.state_dict())

    def train_model(self, X, cond, ticker_ids, mask=None, epochs=11000, batch_size=64, verbose=True):
        cond = self._fit_cond_stats(cond)
        N = X.shape[0]
        X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        cond_t = torch.tensor((cond - self.cond_mean) / self.cond_std,
                              dtype=torch.float32, device=DEVICE)
        tk_t = torch.tensor(ticker_ids, dtype=torch.long, device=DEVICE)
        mask_t = torch.tensor(mask, dtype=torch.float32, device=DEVICE) if mask is not None else None

        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=self.lr_min)
        self.net.train()
        losses = []

        for ep in range(epochs):
            perm = torch.randperm(N, device=DEVICE)
            ep_loss = 0.0; nb = 0
            for s in range(0, N, batch_size):
                idx = perm[s:s + batch_size]
                x0 = X_t[idx]; cb = cond_t[idx]; tk_b = tk_t[idx]
                t = torch.randint(0, self.n_timesteps, (len(idx),), device=DEVICE)
                noise = torch.randn_like(x0)
                x_noisy = self.q_sample(x0, t, noise)
                pred = self.net(x_noisy, t, cb, tk_b)
                if mask_t is not None:
                    m = mask_t[idx]
                    loss = ((pred - noise) ** 2 * m).sum() / m.sum().clamp(min=1)
                else:
                    loss = F.mse_loss(pred, noise)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                ep_loss += loss.item(); nb += 1
            sched.step()
            with torch.no_grad():
                net_sd = self.net.state_dict()
                for k in self.ema_state:
                    self.ema_state[k] = self.ema_decay * self.ema_state[k] + (1 - self.ema_decay) * net_sd[k]
            avg = ep_loss / max(nb, 1)
            losses.append(avg)
            if verbose and (ep + 1) % max(1, epochs // 10) == 0:
                cur_lr = sched.get_last_lr()[0]
                print(f"  epoch {ep+1:5d}/{epochs}  loss={avg:.6f}  lr={cur_lr:.2e}")
        self.net.load_state_dict(self.ema_state)
        self.net.eval()
        return losses

    @torch.no_grad()
    def sample(self, cond, ticker_ids, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        self.net.eval()
        cond = self._normalize_cond(cond)
        n = len(cond)
        cond_t = torch.tensor(cond, dtype=torch.float32, device=DEVICE)
        tk_t = torch.tensor(ticker_ids, dtype=torch.long, device=DEVICE)
        x = torch.randn(n, self.data_dim, device=DEVICE)
        for i in reversed(range(self.n_timesteps)):
            t = torch.full((n,), i, device=DEVICE, dtype=torch.long)
            pred = self.net(x, t, cond_t, tk_t)
            x0_pred = (x - self._ext(self.sqrt_1m_ab, t, x.shape) * pred) / self._ext(self.sqrt_ab, t, x.shape)
            x0_pred = x0_pred.clamp(*self.clip_range)
            mean = self._ext(self.post_coef1, t, x.shape) * x0_pred + self._ext(self.post_coef2, t, x.shape) * x
            if i > 0:
                x = mean + torch.sqrt(self._ext(self.post_var, t, x.shape)) * torch.randn_like(x)
            else:
                x = mean
        return x.cpu().numpy()
