#!/usr/bin/env bash
# Full ticker-conditioned pipeline for a fresh GPU box (e.g. vast.ai).
#
# Runs ONLY the pooled ticker-conditioned diffusion (tc_diff) against the
# classical baselines — no per-ticker solo diffusion. Stages:
#
#   1. verify the data cache (no network fetch; copy your local cache/ first)
#   2. cross-seed experiment: matrix metrics for tc_diff / NW / t-copula
#   3. paired bootstrap significance test (TC-Diff vs NW, vs t-copula)
#   4. VaR + expected-shortfall backtests and mean-reversion strategy
#   5. all figures
#
# Usage, after `git clone` and uploading cache/ into the repo root:
#
#   bash scripts/run_all_tc.sh
#   # or detached, surviving SSH disconnect:
#   nohup bash scripts/run_all_tc.sh > /dev/null 2>&1 &
#
# Knobs (env vars):
#   N_SEEDS=5  N_BOOT=2000  N_SAMPLES=128  BW_SCALE=1.0  EPOCHS_POOLED=11000
#
# BW_SCALE > 1 widens the NW / t-copula kernel bandwidths; at 1.0 their
# conditional distributions are near point masses (see run_risk_analysis.py
# --help), which is faithful to the baselines as published but makes their
# VaR numbers look terrible. Run with e.g. BW_SCALE=3 for a second opinion.
#
# run_experiment.py checkpoints per-seed parquets, so if the box dies you can
# re-run this script and it resumes where it left off.

set -euo pipefail
cd "$(dirname "$0")/.."

N_SEEDS="${N_SEEDS:-5}"
N_BOOT="${N_BOOT:-2000}"
N_SAMPLES="${N_SAMPLES:-128}"
BW_SCALE="${BW_SCALE:-1.0}"
EPOCHS_POOLED="${EPOCHS_POOLED:-11000}"
METHODS=(tc_diff solo_nw t_copula)

# ---- Environment ----
if [ ! -d .venv ]; then
    echo "Creating venv and installing requirements..."
    python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q -r requirements.txt

mkdir -p results/logs
LOG="results/logs/run_all_tc_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "Logging to $LOG"
python -c "import torch; print('torch', torch.__version__, '| device:', 'cuda' if torch.cuda.is_available() else 'CPU (warning: full run will be very slow)')"
echo "Config: N_SEEDS=$N_SEEDS N_BOOT=$N_BOOT N_SAMPLES=$N_SAMPLES BW_SCALE=$BW_SCALE EPOCHS_POOLED=$EPOCHS_POOLED"

# ---- 1. Verify cache (exits nonzero with a clear message if parquets missing) ----
echo; echo "===== [1/5] verify data cache ====="
python scripts/prepare_data.py --use-cache

# ---- 2. Cross-seed experiment: matrix metrics, tc_diff + baselines ----
echo; echo "===== [2/5] cross-seed experiment (${METHODS[*]}) ====="
python scripts/run_experiment.py \
    --n-seeds "$N_SEEDS" \
    --epochs-pooled "$EPOCHS_POOLED" \
    --methods "${METHODS[@]}"

# ---- 3. Paired bootstrap (trains one TC-Diff, varies test-set resampling) ----
echo; echo "===== [3/5] paired bootstrap (n_boot=$N_BOOT) ====="
python scripts/run_bootstrap.py \
    --n-boot "$N_BOOT" \
    --epochs-pooled "$EPOCHS_POOLED"

# ---- 4. VaR + expected shortfall + mean-reversion strategy ----
echo; echo "===== [4/5] risk analysis (n_samples=$N_SAMPLES, bw_scale=$BW_SCALE) ====="
python scripts/run_risk_analysis.py \
    --n-samples "$N_SAMPLES" \
    --epochs-pooled "$EPOCHS_POOLED" \
    --bw-scale "$BW_SCALE" \
    --methods tc_diff solo_nw t_copula uncond_hist

# ---- 5. Figures ----
echo; echo "===== [5/5] figures ====="
python scripts/plot_results.py
python scripts/plot_risk_analysis.py

echo; echo "Done. Collect: results/experiment, results/bootstrap, results/risk, results/figures"
