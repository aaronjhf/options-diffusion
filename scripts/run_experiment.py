#!/usr/bin/env python
"""Run the cross-seed experiment.

For each training seed in --seeds, retrain the pooled ticker-conditioned
diffusion and the per-ticker solo diffusion + baselines, then evaluate on the
fixed test set. Saves per-seed parquets so you can resume if a vast.ai box
goes down mid-run.

Quick smoke test on CPU (drastically reduced epochs):

    python scripts/run_experiment.py --n-seeds 1 --epochs-solo 50 --epochs-pooled 100

Full GPU run (default — matches the notebook hyperparameters):

    python scripts/run_experiment.py --n-seeds 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from options_diffusion.config import (
    CACHE_DIR, COND_EPOCHS_POOLED, COND_EPOCHS_SOLO, DEVICE, RESULTS_DIR,
    SECTOR_TICKERS,
)
from options_diffusion.data.preprocess import prepare_data
from options_diffusion.eval.experiment import (
    METHODS, run_experiment, summarize, summarize_pooled,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR / "experiment")
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=0,
                        help="Seeds used are [seed-start, seed-start+n-seeds).")
    parser.add_argument("--seeds", type=int, nargs="*", default=None,
                        help="Override seed list explicitly.")
    parser.add_argument("--epochs-solo", type=int, default=COND_EPOCHS_SOLO)
    parser.add_argument("--epochs-pooled", type=int, default=COND_EPOCHS_POOLED)
    parser.add_argument("--methods", nargs="*", default=list(METHODS), choices=METHODS)
    parser.add_argument("--tickers", nargs="*", default=SECTOR_TICKERS)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print(f"Tickers: {args.tickers}")

    prep = prepare_data(args.tickers, args.cache_dir)
    print(f"X_train: {prep.X_train.shape}, X_test: {prep.X_test.shape}")
    print(f"Split: {prep.split_date.date()} -> gap end {prep.gap_end_date.date()}")

    seeds = args.seeds if args.seeds is not None else list(
        range(args.seed_start, args.seed_start + args.n_seeds)
    )
    print(f"\nSeeds: {seeds}")
    print(f"Methods: {args.methods}")
    print(f"Epochs: solo={args.epochs_solo}, pooled={args.epochs_pooled}")
    print(f"Out dir: {args.out_dir}\n")

    df = run_experiment(
        prep, seeds, args.out_dir,
        epochs_solo=args.epochs_solo,
        epochs_pooled=args.epochs_pooled,
        methods=tuple(args.methods),
        verbose=not args.quiet,
    )

    print("\n" + "=" * 80)
    print("PER-TICKER SUMMARY (mean +/- std across seeds)")
    print("=" * 80)
    summary = summarize(df)
    summary.to_parquet(args.out_dir / "summary_per_ticker.parquet")
    print(summary.to_string(index=False))

    print("\n" + "=" * 80)
    print("POOLED SUMMARY (mean +/- std across seeds x tickers)")
    print("=" * 80)
    pooled = summarize_pooled(df)
    pooled.to_parquet(args.out_dir / "summary_pooled.parquet")
    print(pooled.to_string(index=False))


if __name__ == "__main__":
    main()
