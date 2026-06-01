"""Polygon + yfinance data fetching.

Requires POLYGON_API_KEY in the environment. This module is only invoked by
scripts/prepare_data.py with --fetch. When running from a published cache
(the GitHub-friendly path), it is never imported.
"""
from __future__ import annotations

import math
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from scipy.optimize import brentq
from scipy.stats import norm

from ..config import (
    MONEYNESS_HI, MONEYNESS_LO, Q, R,
)


BASE_URL = "https://api.polygon.io"


# ---- Black-Scholes IV inversion ----
def bs_call_price(S, K, T, r, q, sigma):
    if sigma <= 0 or T <= 0:
        return max(S * math.exp(-q * T) - K * math.exp(-r * T), 0.0)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def calc_iv(price, S, K, T, r=R, q=Q):
    intrinsic = max(S * math.exp(-q * T) - K * math.exp(-r * T), 0.0)
    if price <= intrinsic + 0.01 or price <= 0 or S <= 0 or K <= 0 or T <= 0:
        return None
    try:
        iv = brentq(lambda sig: bs_call_price(S, K, T, r, q, sig) - price,
                    1e-4, 5.0, xtol=1e-8, maxiter=200)
        return iv if 0.01 <= iv <= 3.0 else None
    except (ValueError, RuntimeError):
        return None


def third_fridays(start_year, start_month, end_year, end_month):
    results = []
    y, m = start_year, start_month
    while (y, m) <= (end_year, end_month):
        first_day = date(y, m, 1)
        dow = first_day.weekday()
        first_friday = first_day + timedelta(days=(4 - dow) % 7)
        results.append(first_friday + timedelta(weeks=2))
        if m == 12:
            y += 1; m = 1
        else:
            m += 1
    return results


class PolygonClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.session = requests.Session()

    def _get(self, url, params=None):
        if params is None:
            params = {}
        params["apiKey"] = self.api_key
        for attempt in range(5):
            resp = self.session.get(url, params=params, timeout=30)
            if resp.status_code == 429:
                wait = 2 ** attempt * 5
                print(f"    Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        resp.raise_for_status()

    def fetch_contracts(self, ticker, expiration_date, strike_min, strike_max):
        base_url = f"{BASE_URL}/v3/reference/options/contracts"
        base_params = {
            "underlying_ticker": ticker, "contract_type": "call",
            "expiration_date.gte": (pd.Timestamp(expiration_date) - timedelta(days=5)).strftime("%Y-%m-%d"),
            "expiration_date.lte": (pd.Timestamp(expiration_date) + timedelta(days=5)).strftime("%Y-%m-%d"),
            "strike_price.gte": strike_min, "strike_price.lte": strike_max,
            "limit": 1000, "order": "asc", "sort": "strike_price",
        }
        all_results = []
        for exp_flag in [True, False]:
            params = {**base_params, "expired": exp_flag}
            url = base_url
            while True:
                data = self._get(url, params)
                all_results.extend(data.get("results", []))
                next_url = data.get("next_url")
                if not next_url:
                    break
                url = next_url
                params = {}
        seen, deduped = set(), []
        for c in all_results:
            if c["ticker"] not in seen:
                seen.add(c["ticker"])
                deduped.append(c)
        return deduped

    def fetch_aggs(self, ticker, from_date, to_date):
        url = f"{BASE_URL}/v2/aggs/ticker/{ticker}/range/1/day/{from_date}/{to_date}"
        data = self._get(url, {"adjusted": "true", "sort": "asc", "limit": 50000})
        return data.get("results", []) or []


def _bars_to_quote_records(bars_df, contract_info, opt_ticker, spot):
    """Convert Polygon bar rows for one contract into IV-tagged quote records."""
    if opt_ticker not in contract_info:
        return []
    K, exp_date = contract_info[opt_ticker]
    out = []
    for _, bar in bars_df.iterrows():
        obs_date = bar["date"]
        S = spot.get(obs_date)
        if S is None or (isinstance(S, float) and np.isnan(S)):
            continue
        S = float(S)
        dte_days = (exp_date - obs_date).days
        if dte_days < 3:
            continue
        T_yrs = dte_days / 365.0
        moneyness = K / S
        if moneyness < MONEYNESS_LO or moneyness > MONEYNESS_HI:
            continue
        call_price = float(bar["c"])
        iv = calc_iv(call_price, S, K, T_yrs)
        if iv is None:
            continue
        out.append({
            "date": obs_date, "strike": K, "expiration": exp_date,
            "call_price": call_price, "spot": S, "dte_days": dte_days,
            "moneyness": moneyness, "T_years": T_yrs, "iv": iv,
        })
    return out


def _build_contract_info(contracts_df):
    info = {}
    for _, row in contracts_df.iterrows():
        t = row["ticker"]
        k = float(row["strike_price"])
        exp = row["expiration_date"]
        if isinstance(exp, str):
            exp = datetime.strptime(exp[:10], "%Y-%m-%d").date()
        info[t] = (k, exp)
    return info


def _fetch_spot(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        close = df[("Close", ticker)]
    else:
        close = df["Close"]
    close.index = pd.to_datetime(close.index).date
    close.name = "close"
    return close


def fetch_ticker_data(
    ticker: str,
    client: PolygonClient,
    cache_dir: Path,
    start: str = "2022-02-01",
    end: str | None = None,
    force_refetch: bool = False,
):
    """Fetch spot + raw option quotes for one ticker. Returns (quotes_df, spot_df).

    Quotes are returned as raw records with IV computed but NOT placed on a grid.
    SVI fitting (`svi.fit_svi_from_quotes`) handles interpolation.
    """
    if end is None:
        end = str(date.today())

    spot_cache = cache_dir / f"{ticker}_spot.parquet"
    quotes_cache = cache_dir / f"{ticker}_raw_quotes.parquet"

    if not force_refetch and quotes_cache.exists() and spot_cache.exists():
        print(f"  Loading {ticker} from cache...")
        return pd.read_parquet(quotes_cache), pd.read_parquet(spot_cache)

    print(f"  Fetching {ticker} from Polygon...")
    spot = _fetch_spot(ticker, start, end)
    print(f"    Spot: {len(spot)} days")

    expirations = third_fridays(2022, 3, 2028, 1)
    all_contracts = []
    for exp in expirations:
        t0 = max(list(spot.index)[0], (pd.Timestamp(exp) - timedelta(days=400)).date())
        t1 = min(list(spot.index)[-1], pd.Timestamp(exp).date())
        relevant = spot[(spot.index >= t0) & (spot.index <= t1)]
        if len(relevant) == 0:
            continue
        s_min = int(float(relevant.min()) * MONEYNESS_LO)
        s_max = int(float(relevant.max()) * MONEYNESS_HI) + 1
        try:
            results = client.fetch_contracts(ticker, str(exp), s_min, s_max)
        except Exception:
            results = []
        if not results:
            time.sleep(0.05); continue
        exp_dates = pd.Series([c["expiration_date"] for c in results])
        actual_exp = exp_dates.mode()[0]
        results = [c for c in results if c["expiration_date"] == actual_exp]
        all_contracts.extend(results)
        time.sleep(0.05)

    contracts_df = pd.DataFrame(all_contracts).drop_duplicates("ticker")
    print(f"    Contracts: {len(contracts_df)}")

    price_data = {}
    opt_tickers = contracts_df["ticker"].unique().tolist()
    for idx, opt_t in enumerate(opt_tickers):
        try:
            bars = client.fetch_aggs(opt_t, start, end)
            if bars:
                bdf = pd.DataFrame(bars)
                bdf["date"] = pd.to_datetime(bdf["t"], unit="ms").dt.date
                price_data[opt_t] = bdf
        except Exception:
            pass
        time.sleep(0.06)
        if (idx + 1) % 100 == 0:
            print(f"    Price progress: {idx+1}/{len(opt_tickers)}")
    print(f"    Price data for {len(price_data)}/{len(opt_tickers)} contracts")

    contract_info = _build_contract_info(contracts_df)

    quote_records = []
    for opt_ticker, bars_df in price_data.items():
        quote_records.extend(_bars_to_quote_records(bars_df, contract_info, opt_ticker, spot))

    quotes_df = pd.DataFrame(quote_records)
    print(f'    Raw quotes: {len(quotes_df)} observations across {quotes_df["date"].nunique()} days')

    spot_out = spot.to_frame(name="close")
    spot_out.index = pd.to_datetime(spot_out.index)
    spot_out.to_parquet(spot_cache)
    quotes_df.to_parquet(quotes_cache)
    return quotes_df, spot_out


def _augment_fetch_window(ticker, client, spot, exp_from_date, start_str, end_str):
    """Fetch new option quote records in a date window for one ticker."""
    expirations = [e for e in third_fridays(2022, 3, 2028, 1) if e >= exp_from_date]
    all_contracts = []
    for exp in expirations:
        t0 = max(list(spot.index)[0], (pd.Timestamp(exp) - timedelta(days=400)).date())
        t1 = min(list(spot.index)[-1], pd.Timestamp(exp).date())
        relevant = spot[(spot.index >= t0) & (spot.index <= t1)]
        if len(relevant) == 0:
            continue
        s_min = int(float(relevant.min()) * MONEYNESS_LO)
        s_max = int(float(relevant.max()) * MONEYNESS_HI) + 1
        try:
            results = client.fetch_contracts(ticker, str(exp), s_min, s_max)
        except Exception:
            results = []
        if not results:
            time.sleep(0.05); continue
        exp_dates = pd.Series([c["expiration_date"] for c in results])
        actual_exp = exp_dates.mode()[0]
        results = [c for c in results if c["expiration_date"] == actual_exp]
        all_contracts.extend(results)
        time.sleep(0.05)

    if not all_contracts:
        print("    No contracts in augmentation window.")
        return pd.DataFrame()

    contracts_df = pd.DataFrame(all_contracts).drop_duplicates("ticker")
    print(f"    Contracts touched: {len(contracts_df)}")

    price_data = {}
    opt_tickers = contracts_df["ticker"].unique().tolist()
    for idx, opt_t in enumerate(opt_tickers):
        try:
            bars = client.fetch_aggs(opt_t, start_str, end_str)
            if bars:
                bdf = pd.DataFrame(bars)
                bdf["date"] = pd.to_datetime(bdf["t"], unit="ms").dt.date
                price_data[opt_t] = bdf
        except Exception:
            pass
        time.sleep(0.06)
        if (idx + 1) % 200 == 0:
            print(f"    Bars progress: {idx+1}/{len(opt_tickers)}")
    print(f"    Bars returned for {len(price_data)}/{len(opt_tickers)} contracts")

    contract_info = _build_contract_info(contracts_df)
    quote_records = []
    for opt_ticker, bars_df in price_data.items():
        quote_records.extend(_bars_to_quote_records(bars_df, contract_info, opt_ticker, spot))
    return pd.DataFrame(quote_records)


def augment_ticker_data(
    ticker: str,
    client: PolygonClient,
    cache_dir: Path,
    end: str | None = None,
    overlap_days: int = 5,
):
    """Incrementally extend cached raw_quotes + spot with new data.

    Downstream SVI / IV / mask caches are deleted for any ticker that received
    new rows so the SVI step refits on the enlarged dataset.
    """
    if end is None:
        end = str(date.today())
    end_d = pd.Timestamp(end).date()

    spot_cache = cache_dir / f"{ticker}_spot.parquet"
    quotes_cache = cache_dir / f"{ticker}_raw_quotes.parquet"

    if not (spot_cache.exists() and quotes_cache.exists()):
        print(f"  {ticker}: no existing cache - run fetch_ticker_data first.")
        return None, None

    existing_quotes = pd.read_parquet(quotes_cache).copy()
    existing_spot = pd.read_parquet(spot_cache)
    existing_quotes["date"] = pd.to_datetime(existing_quotes["date"]).dt.date
    existing_quotes["expiration"] = pd.to_datetime(existing_quotes["expiration"]).dt.date

    last_d = pd.Timestamp(existing_quotes["date"].max()).date()
    if last_d >= end_d - timedelta(days=1):
        print(f"  {ticker}: already up to date through {last_d}.")
        return existing_quotes, existing_spot

    aug_start_d = last_d - timedelta(days=overlap_days)
    aug_start_s = aug_start_d.strftime("%Y-%m-%d")
    print(f"  {ticker}: cache through {last_d}; augmenting {aug_start_s} -> {end}")

    spot_full = _fetch_spot(ticker, "2022-02-01", end)

    new_quotes = _augment_fetch_window(
        ticker, client, spot_full,
        exp_from_date=aug_start_d, start_str=aug_start_s, end_str=end,
    )
    print(f"    New quote records fetched: {len(new_quotes)}")

    if len(new_quotes) == 0:
        merged = existing_quotes
    else:
        kept = existing_quotes[existing_quotes["date"] < aug_start_d]
        merged = (
            pd.concat([kept, new_quotes], ignore_index=True)
              .drop_duplicates(subset=["date", "strike", "expiration"], keep="last")
              .sort_values(["date", "expiration", "strike"])
              .reset_index(drop=True)
        )
        print(f"    Kept from old cache: {len(kept)}   merged total: {len(merged)}")

    print(f'    Coverage now: {merged["date"].min()} -> {merged["date"].max()}'
          f'  ({merged["date"].nunique()} days)')

    spot_out = spot_full.to_frame(name="close")
    spot_out.index = pd.to_datetime(spot_out.index)
    spot_out.to_parquet(spot_cache)
    merged.to_parquet(quotes_cache)

    for stale in [
        cache_dir / f"{ticker}_svi_risk_factors.parquet",
        cache_dir / f"{ticker}_iv_surface.parquet",
        cache_dir / f"{ticker}_obs_mask.npy",
    ]:
        if stale.exists():
            stale.unlink()
            print(f"    Invalidated {stale.name}")

    return merged, spot_out


def get_polygon_client() -> PolygonClient:
    api_key = os.environ.get("POLYGON_API_KEY")
    if not api_key:
        raise RuntimeError(
            "POLYGON_API_KEY not set. Run:\n"
            "  export POLYGON_API_KEY='your_key_here'\n"
            "before running data fetching."
        )
    return PolygonClient(api_key)
