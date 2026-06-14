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
#   6. VaR-scale diagnostic readout (confirms calibration held)
#
# Usage, after `git clone` and uploading cache/ into the repo root:
#
#   bash scripts/run_all_tc.sh
#   # or detached, surviving SSH disconnect:
#   nohup bash scripts/run_all_tc.sh > /dev/null 2>&1 &
#
# Knobs (env vars):
#   N_SEEDS=5  N_BOOT=2000  N_SAMPLES=128  BW_SCALE=1.0  EPOCHS_POOLED=11000
#   VAR_SCALE_MODE=auto  VAR_SCALE=4
#
# VAR_SCALE_MODE recalibrates the predictive spread before the VaR/ES/mean-
# reversion metrics (the conditioned diffusion is over-confident and otherwise
# under-estimates VaR badly):
#   auto   (default) per-method inflation factor estimated on train, no leakage
#   manual fixed factor VAR_SCALE (~4 calibrates TC-Diff) on the diffusion only
#   off    no recalibration (reproduces the raw, under-dispersed numbers)
# Stage 6 runs diagnose_var_scale.py on the result: a coverage-optimal S near 1
# means calibration held; S still ~2-4 means auto under-inflated (conditional
# mean over-fit) and a retrain with more regularization is the real fix.
#
# BW_SCALE > 1 widens the NW / t-copula kernel bandwidths; at 1.0 their
# conditional distributions are near point masses (see run_risk_analysis.py
# --help), which is faithful to the baselines as published but makes their
# VaR numbers look terrible. Run with e.g. BW_SCALE=3 for a second opinion.
#
# run_experiment.py checkpoints per-seed parquets, so if the box dies you can
# re-run this script and it resumes where it left off.
#
# Cache note: the SVI risk-factor parquets must be the de-leaked refit (run
# `python scripts/prepare_data.py --force-svi` once locally before uploading if
# unsure). Stage 1 only verifies presence, it does not refit.

set -euo pipefail
cd "$(dirname "$0")/.."

N_SEEDS="${N_SEEDS:-5}"
N_BOOT="${N_BOOT:-2000}"
N_SAMPLES="${N_SAMPLES:-128}"
BW_SCALE="${BW_SCALE:-1.0}"
EPOCHS_POOLED="${EPOCHS_POOLED:-11000}"
VAR_SCALE_MODE="${VAR_SCALE_MODE:-auto}"
VAR_SCALE="${VAR_SCALE:-4}"
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
python -c "import torch; print('torch', torch.__version__, '| device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU — WARNING: no CUDA, full run will be 10-50x slower')"
echo "Config: N_SEEDS=$N_SEEDS N_BOOT=$N_BOOT N_SAMPLES=$N_SAMPLES BW_SCALE=$BW_SCALE EPOCHS_POOLED=$EPOCHS_POOLED VAR_SCALE_MODE=$VAR_SCALE_MODE VAR_SCALE=$VAR_SCALE"

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
echo; echo "===== [4/6] risk analysis (n_samples=$N_SAMPLES, bw_scale=$BW_SCALE, var_scale_mode=$VAR_SCALE_MODE) ====="
RISK_VARSCALE_ARGS=(--var-scale-mode "$VAR_SCALE_MODE")
[ "$VAR_SCALE_MODE" = manual ] && RISK_VARSCALE_ARGS+=(--var-scale "$VAR_SCALE")
python scripts/run_risk_analysis.py \
    --n-samples "$N_SAMPLES" \
    --epochs-pooled "$EPOCHS_POOLED" \
    --bw-scale "$BW_SCALE" \
    --methods tc_diff solo_nw t_copula uncond_hist \
    "${RISK_VARSCALE_ARGS[@]}"

# ---- 5. Figures ----
echo; echo "===== [5/6] figures ====="
python scripts/plot_results.py
python scripts/plot_risk_analysis.py

# ---- 6. VaR-scale diagnostic (post-recalibration sanity check) ----
echo; echo "===== [6/6] VaR-scale diagnostic ====="
python scripts/diagnose_var_scale.py --series results/risk/series.parquet || true

echo; echo "Done. Collect: results/experiment, results/bootstrap, results/risk, results/figures"
