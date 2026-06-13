"""Central configuration. Edit values here, not in scripts."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


# ---- Hardware ----
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    # Ampere+ (RTX 3090 etc.): allow TF32 matmuls — faster, and the precision
    # difference is far below the noise floor of these small MLPs.
    torch.set_float32_matmul_precision("high")


# ---- Tickers ----
# Subset used for the published experiment. To regenerate cache for a different
# set, edit and re-run scripts/prepare_data.py.
SECTOR_TICKERS = ["MU", "PLTR", "HOOD", "MSFT", "GOOG", "AMZN", "META"]


# ---- Train / test split ----
TRAIN_FRAC = 0.80
GAP_DAYS = 5   # gap between train end and test start (in trading days, doubled in date space)


# ---- SVI grid ----
M_GRID = np.array([0.90, 0.92, 0.95, 0.97, 0.99, 1.00, 1.01, 1.03, 1.05, 1.08, 1.10])
T_GRID = np.array([21, 42, 63, 90, 120, 180, 270, 360])
N_MONEYNESS = len(M_GRID)
N_DTE = len(T_GRID)
K_GRID = np.log(M_GRID)
MONEYNESS_LO = 0.85   # raw quote filter (wider than M_GRID)
MONEYNESS_HI = 1.15
TRADING_DAYS_PER_YEAR = 365.25
R = 0.04   # risk-free rate
Q = 0.013  # dividend yield


# ---- Risk factors ----
SVI_RISK_FACTOR_NAMES = [
    "atm_6w", "atm_3m", "atm_6m",
    "skew_6w", "skew_3m", "skew_6m",
    "term_slope", "curvature_3m",
]
LOG_DIFF_FEATURES = ["atm_6w", "atm_3m", "atm_6m"]
SIMPLE_DIFF_FEATURES = ["skew_6w", "skew_3m", "skew_6m", "term_slope", "curvature_3m"]
N_RF = len(SVI_RISK_FACTOR_NAMES)


# ---- EWMA conditioning ----
EWMA_SHORT_ALPHA = 0.99
EWMA_LONG_ALPHA = 0.25
COND_DIM_EWMA = 3 + 2 * N_RF   # VIX(level + 2 EWMAs) + (short + long) EWMAs over 8 risk factors  = 19
TICKER_EMBED_DIM = 16


# ---- Training hyperparameters ----
COND_EPOCHS_SOLO = 6000           # per-ticker solo diffusion
COND_EPOCHS_POOLED = 10000        # pooled ticker-conditioned diffusion (solo + 5000)
BATCH_SIZE = 512
N_TIMESTEPS = 500
HIDDEN_DIMS = (128, 256, 128)
T_DIM = 128
C_DIM = 64
DROPOUT = 0.15
LR = 4e-4
LR_MIN = 1e-5
CLIP_RANGE = (-4.0, 4.0)
EMA_DECAY = 0.999


# ---- Paths ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "cache"
RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
