#!/usr/bin/env python
"""Build the cache that the experiment scripts consume.

Two modes:

  --use-cache (default)
      Validate that the parquets in --cache-dir already cover every ticker in
      `options_diffusion.config.SECTOR_TICKERS`. This is the GitHub-friendly
      path: no API calls, no proprietary data fetched. Use this when running
      the experiment against a cache you copied to the machine yourself
      (vast.ai, etc.).

  --fetch
      Hit Polygon (for option contracts/bars) and yfinance (for spot/VIX) to
      build / augment the cache from scratch. Requires POLYGON_API_KEY in the
      env. This path produces proprietary data and should NOT be run on a
      shared machine without authorization.

After fetching, this script always runs the SVI fit step, which is fast and
deterministic.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from options_diffusion.config import CACHE_DIR, SECTOR_TICKERS
from options_diffusion.data.svi import fit_svi_from_quotes


def _verify_cache(cache_dir: Path, tickers: list[str]) -> int:
    missing = []
    for t in tickers:
        rf = cache_dir / f"{t}_svi_risk_factors.parquet"
        if not rf.exists():
            missing.append(rf)
    if missing:
        print("Missing cache files:")
        for p in missing:
            print(f"  {p}")
        print("\nRun with --fetch to build cache from Polygon (requires API key),")
        print("or copy the cache files into the cache directory.")
        return 1
    print(f"Cache OK — all {len(tickers)} risk-factor parquets present in {cache_dir}.")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR,
                        help=f"Cache directory (default: {CACHE_DIR})")
    parser.add_argument("--use-cache", action="store_true", default=True,
                        help="Validate existing cache (default).")
    parser.add_argument("--fetch", action="store_true",
                        help="Fetch from Polygon + yfinance. Overrides --use-cache. "
                             "Requires POLYGON_API_KEY in env. Proprietary data.")
    parser.add_argument("--augment", action="store_true",
                        help="With --fetch, incrementally extend cache to today.")
    parser.add_argument("--force-svi", action="store_true",
                        help="Re-fit SVI even if cached risk factors exist.")
    parser.add_argument("--tickers", nargs="*", default=SECTOR_TICKERS,
                        help="Subset of tickers (default: SECTOR_TICKERS).")
    args = parser.parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)

    if not args.fetch:
        if args.force_svi:
            # Refit SVI from the cached raw quotes — no network needed. Use
            # this after changes to svi.py to rebuild the risk-factor parquets.
            import pandas as pd
            for t in args.tickers:
                quotes_path = args.cache_dir / f"{t}_raw_quotes.parquet"
                if not quotes_path.exists():
                    print(f"{quotes_path} missing — cannot refit SVI for {t}.")
                    sys.exit(1)
                fit_svi_from_quotes(t, pd.read_parquet(quotes_path),
                                    args.cache_dir, force=True)
        sys.exit(_verify_cache(args.cache_dir, args.tickers))

    # Proprietary path: fetch + SVI
    from options_diffusion.data.fetch import (
        augment_ticker_data, fetch_ticker_data, get_polygon_client,
    )

    client = get_polygon_client()

    for i, t in enumerate(args.tickers):
        print(f"\n[{i+1}/{len(args.tickers)}] {t}")
        if args.augment:
            augment_ticker_data(t, client, args.cache_dir)
        else:
            fetch_ticker_data(t, client, args.cache_dir)
        if i < len(args.tickers) - 1:
            print("    Sleeping 5.1s for rate limit...")
            time.sleep(5.1)

    # SVI fit produces ticker_svi_risk_factors.parquet which downstream consumes.
    import pandas as pd
    for t in args.tickers:
        quotes = pd.read_parquet(args.cache_dir / f"{t}_raw_quotes.parquet")
        fit_svi_from_quotes(t, quotes, args.cache_dir, force=args.force_svi)

    print("\nDone. Cache ready for scripts/run_experiment.py")


if __name__ == "__main__":
    main()
