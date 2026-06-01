#!/usr/bin/env python
"""Paired-bootstrap significance test on a SINGLE trained TC-Diffusion run.

This script complements run_experiment.py by measuring noise from the test set,
not from the training seed. It trains one TC-Diff (with the given --train-seed),
runs each baseline once on the fixed test set, then bootstrap-resamples
(real, generated) test-set indices together and compares metrics pairwise.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from options_diffusion.config import (
    CACHE_DIR, COND_DIM_EWMA, COND_EPOCHS_POOLED, DEVICE, N_TIMESTEPS,
    RESULTS_DIR, SECTOR_TICKERS, TICKER_EMBED_DIM, BATCH_SIZE,
)
from options_diffusion.data.preprocess import prepare_data
from options_diffusion.eval.bootstrap import run_paired_bootstrap, summarize_bootstrap
from options_diffusion.eval.experiment import set_global_seed
from options_diffusion.models.diffusion import TickerCondDDPM
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR / "bootstrap")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--boot-seed", type=int, default=42)
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--epochs-pooled", type=int, default=COND_EPOCHS_POOLED)
    parser.add_argument("--tickers", nargs="*", default=SECTOR_TICKERS)
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    prep = prepare_data(args.tickers, args.cache_dir)

    print(f"\nTraining TC-Diffusion (train-seed={args.train_seed}, "
          f"epochs={args.epochs_pooled})...")
    set_global_seed(args.train_seed)
    tc_diff = TickerCondDDPM(
        data_dim=prep.X_train.shape[1],
        n_tickers=len(prep.tickers),
        cond_input_dim=COND_DIM_EWMA,
        n_timesteps=N_TIMESTEPS,
        ticker_embed_dim=TICKER_EMBED_DIM,
    )
    tc_diff.train_model(
        prep.X_train.astype(np.float32),
        prep.tc_cond_ewma_train.astype(np.float32),
        prep.ticker_ids_train,
        mask=prep.mask_train,
        epochs=args.epochs_pooled, batch_size=BATCH_SIZE,
    )

    diff_stats = run_paired_bootstrap(
        prep, tc_diff, args.out_dir,
        n_boot=args.n_boot, boot_seed=args.boot_seed,
    )

    summary = summarize_bootstrap(diff_stats)
    summary.to_parquet(args.out_dir / "summary.parquet")
    print("\n" + "=" * 80)
    print("PAIRED BOOTSTRAP SUMMARY (negative diff = TC-Diff better)")
    print("=" * 80)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
