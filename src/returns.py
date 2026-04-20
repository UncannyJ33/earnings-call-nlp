"""
src/returns.py — Cumulative Abnormal Return (CAR) calculation.

Earnings window timing decision:
  Calls before 4:00 p.m. ET (morning / midday) → day 1 = earnings_date.
    The market is open and can react immediately; using the same day
    maximises the signal window and avoids losing one full day of reaction.
  Calls at or after 4:00 p.m. ET (after close) → day 1 = next trading day.
    The market is closed; the earliest reaction is the following morning.
    Using earnings_date here would capture pre-call returns — noise, not signal.
  Time unknown → next trading day (conservative fallback).

Primary CAR method: market-adjusted (stock return − SPY return).
Robustness check:  beta-adjusted CAR (stock return − beta × SPY return),
                   where beta is estimated on the 120 trading days prior to
                   the earnings event.

Pipeline:
  1. fetch_price_data()          — download & cache adjusted close prices
  2. identify_earnings_window()  — map each call to its 1/3/5-day window
  3. compute_car()               — market-adjusted CAR
  4. compute_beta_adjusted_car() — beta-adjusted CAR (robustness)
  5. build_returns_matrix()      — assemble analysis-ready parquet
"""

import logging
import re
import time as time_module
import warnings
from datetime import datetime, time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

import config

logger = logging.getLogger(__name__)

# Market close time in ET — calls at or after this are treated as after-hours
_MARKET_CLOSE = time(16, 0)  # 4:00 p.m. ET


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _parse_call_time(raw_date: object) -> Optional[time]:
    """Extract the call time from the raw date field.

    Raw formats seen in dataset:
      "Aug 27, 2020, 9:00 p.m. ET"
      "Aug 5, 2020, 8:30 a.m. ET"
      ['Company Q2 2020', 'Jul 30, 2020, 4:30 p.m. ET']  (list variant)

    Args:
        raw_date: Raw value from the dataset 'date' column.

    Returns:
        datetime.time in ET, or None if unparseable.
    """
    if isinstance(raw_date, list):
        combined = " ".join(str(p) for p in raw_date)
    elif isinstance(raw_date, str):
        combined = raw_date
    else:
        return None

    # Match "H:MM a.m." or "H:MM p.m." (with or without leading zero)
    match = re.search(r"(\d{1,2}):(\d{2})\s+(a\.m\.|p\.m\.)", combined, re.IGNORECASE)
    if not match:
        return None

    hour = int(match.group(1))
    minute = int(match.group(2))
    period = match.group(3).lower().replace(".", "")  # "am" or "pm"

    if period == "pm" and hour != 12:
        hour += 12
    elif period == "am" and hour == 12:
        hour = 0

    try:
        return time(hour, minute)
    except ValueError:
        return None


def _build_timing_map(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Build a (ticker, date, after_close) lookup from the raw dataset.

    after_close is True when the call time is at or after 4:00 p.m. ET,
    or when the time is unknown (conservative fallback).

    Args:
        raw_df: Raw transcript DataFrame with 'ticker', 'date', columns.

    Returns:
        DataFrame with columns: ticker, call_date, after_close.
    """
    rows = []
    for _, row in raw_df.iterrows():
        call_time = _parse_call_time(row["date"])
        if call_time is None:
            after_close = True  # unknown → conservative
        else:
            after_close = call_time >= _MARKET_CLOSE

        # Extract date only (same logic as data_ingestion)
        raw_date_str = row["date"]
        if isinstance(raw_date_str, list):
            raw_date_str = " ".join(str(p) for p in raw_date_str)

        date_match = re.search(
            r"(Jan(?:uary)?\.?|Feb(?:ruary)?\.?|Mar(?:ch)?\.?|Apr(?:il)?\.?|"
            r"May\.?|Jun(?:e)?\.?|Jul(?:y)?\.?|Aug(?:ust)?\.?|Sep(?:tember)?\.?|"
            r"Oct(?:ober)?\.?|Nov(?:ember)?\.?|Dec(?:ember)?\.?)"
            r"\s+\d{1,2},?\s+\d{4}",
            raw_date_str, re.IGNORECASE,
        )
        if not date_match:
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            call_date = pd.to_datetime(date_match.group(0), format="mixed", errors="coerce")
        if pd.isna(call_date):
            continue

        rows.append({
            "ticker": str(row["ticker"]).strip().upper(),
            "call_date": call_date.normalize(),
            "after_close": after_close,
        })

    result = pd.DataFrame(rows).drop_duplicates(subset=["ticker", "call_date"])
    logger.info(
        "_build_timing_map: %d events, %d after close (%.1f%%), %d before/during",
        len(result),
        result["after_close"].sum(),
        result["after_close"].mean() * 100,
        (~result["after_close"]).sum(),
    )
    return result


# ---------------------------------------------------------------------------
# 1. Price data fetching and caching
# ---------------------------------------------------------------------------

def fetch_price_data(
    tickers: list[str],
    start_date: str,
    end_date: str,
    cache_path: Optional[Path] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download adjusted close prices for all tickers plus SPY benchmark.

    Results are cached to parquet. If the cache exists it is loaded directly
    without re-downloading. Delete the cache file to force a fresh download.

    Args:
        tickers: List of ticker symbols to download.
        start_date: Download start date string ("YYYY-MM-DD"). Should be at
            least 180 days before the first earnings event to allow for the
            120-day beta lookback.
        end_date: Download end date string ("YYYY-MM-DD").
        cache_path: Path for the prices parquet cache. Defaults to
            config.CACHE_DIR / "prices.parquet".

    Returns:
        Tuple of (stock_prices_df, spy_df):
          - stock_prices_df: DataFrame with dates as index and tickers as
            columns (adjusted close prices).
          - spy_df: Single-column DataFrame of SPY adjusted close.
    """
    if cache_path is None:
        cache_path = config.CACHE_DIR / "prices.parquet"

    if cache_path.exists():
        logger.info("Loading price cache from %s", cache_path)
        combined = pd.read_parquet(cache_path)
        spy_df = combined[["SPY"]].copy()
        stock_df = combined.drop(columns=["SPY"])
        logger.info(
            "Cache loaded: %d tickers, %d trading days",
            stock_df.shape[1], len(stock_df),
        )
        return stock_df, spy_df

    all_tickers = list(set(tickers) | {"SPY"})
    logger.info(
        "Downloading prices for %d tickers from %s to %s...",
        len(all_tickers), start_date, end_date,
    )

    # Download in batches — yfinance rate-limits and silently drops tickers
    # when given too many at once. Batches of 100 with a short inter-batch
    # delay are reliable. Failed batches are retried once after a longer pause.
    BATCH_SIZE = 100
    BATCH_DELAY = 2     # seconds between batches (avoids rate limiting)
    RETRY_DELAY = 15    # seconds before retrying a failed batch
    all_frames: list[pd.DataFrame] = []

    # Ensure SPY is always in the first batch so it's downloaded early
    sorted_tickers = ["SPY"] + [t for t in all_tickers if t != "SPY"]
    batches = [sorted_tickers[i : i + BATCH_SIZE] for i in range(0, len(sorted_tickers), BATCH_SIZE)]
    n_batches = len(batches)

    for batch_num, batch in enumerate(batches, 1):
        print(f"  Batch {batch_num}/{n_batches} ({len(batch)} tickers)...", end="\r")

        for attempt in range(2):  # try twice before giving up
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = yf.download(
                    batch,
                    start=start_date,
                    end=end_date,
                    auto_adjust=True,
                    progress=False,
                    threads=False,  # serial avoids hammering the API
                )

            if not raw.empty:
                break
            if attempt == 0:
                logger.debug("Batch %d returned empty, retrying after %ds...", batch_num, RETRY_DELAY)
                time_module.sleep(RETRY_DELAY)

        if raw.empty:
            logger.warning("Batch %d returned empty after retry — skipping", batch_num)
            time_module.sleep(BATCH_DELAY)
            continue

        # Extract Close prices — yfinance always returns MultiIndex now
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"]
        else:
            close = raw[["Close"]].rename(columns={"Close": batch[0]})

        all_frames.append(close)
        time_module.sleep(BATCH_DELAY)

    print()  # newline after \r progress

    if not all_frames:
        raise RuntimeError("Price download returned no data for any batch.")

    # Align on the union of all trading days, forward-fill within each ticker
    prices = pd.concat(all_frames, axis=1)
    prices = prices.loc[~prices.index.duplicated()]
    prices = prices.sort_index()
    prices = prices.dropna(how="all")

    # Report coverage
    spy_present = "SPY" in prices.columns and prices["SPY"].notna().any()
    non_spy = [t for t in all_tickers if t != "SPY"]
    succeeded = [t for t in non_spy if t in prices.columns and prices[t].notna().sum() > 0]
    failed = [t for t in non_spy if t not in succeeded]

    print(f"\n{'='*55}")
    print(f"  Price download summary")
    print(f"{'='*55}")
    print(f"  Tickers requested : {len(non_spy):,}")
    print(f"  Tickers succeeded : {len(succeeded):,}  ({100*len(succeeded)/len(non_spy):.1f}%)")
    print(f"  Tickers failed    : {len(failed):,}")
    print(f"  SPY downloaded    : {'yes' if spy_present else 'NO — CRITICAL ERROR'}")
    print(f"  Trading days      : {len(prices):,}")
    print(f"  Date range        : {prices.index[0].date()} → {prices.index[-1].date()}")
    if failed:
        print(f"  Sample failures   : {failed[:10]}")
    print(f"{'='*55}\n")

    # Cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(cache_path)
    logger.info("Price cache saved → %s", cache_path)

    spy_df = prices[["SPY"]].copy()
    stock_df = prices.drop(columns=["SPY"], errors="ignore")

    return stock_df, spy_df


# ---------------------------------------------------------------------------
# 2. Earnings window identification
# ---------------------------------------------------------------------------

def identify_earnings_window(
    earnings_date: pd.Timestamp,
    after_close: bool,
    trading_days: pd.DatetimeIndex,
    windows: list[int] = None,
) -> dict[int, list[pd.Timestamp]]:
    """Map an earnings date to its CAR trading day windows.

    Timing rule (see module docstring for rationale):
      - after_close=False (before/during market): window starts on earnings_date
      - after_close=True  (post-close or unknown): window starts next trading day

    Args:
        earnings_date: The date of the earnings call.
        after_close: True if call was at or after 4:00 p.m. ET.
        trading_days: DatetimeIndex of all valid trading days (from price data).
        windows: CAR window lengths in trading days. Defaults to config.CAR_WINDOWS.

    Returns:
        Dict mapping window length → list of Timestamps for that window.
        Empty dict if the earnings date cannot be mapped to a trading day.
    """
    if windows is None:
        windows = config.CAR_WINDOWS

    # Find the first trading day on or after earnings_date
    future_days = trading_days[trading_days >= earnings_date]
    if len(future_days) == 0:
        return {}

    if after_close:
        # Start on the NEXT trading day after earnings_date
        next_days = trading_days[trading_days > earnings_date]
        if len(next_days) == 0:
            return {}
        day1 = next_days[0]
    else:
        # Start on earnings_date itself (or next trading day if it's a holiday)
        day1 = future_days[0]

    # Find day1's position in the trading day index
    try:
        start_pos = trading_days.get_loc(day1)
    except KeyError:
        return {}

    result = {}
    for w in windows:
        end_pos = start_pos + w
        if end_pos > len(trading_days):
            result[w] = []  # not enough trading days remaining
        else:
            result[w] = list(trading_days[start_pos:end_pos])

    return result


# ---------------------------------------------------------------------------
# 3. Market-adjusted CAR
# ---------------------------------------------------------------------------

def compute_car(
    ticker: str,
    window_dates: dict[int, list[pd.Timestamp]],
    stock_prices: pd.DataFrame,
    spy_prices: pd.DataFrame,
) -> dict[str, float]:
    """Compute market-adjusted CAR for 1, 3, and 5-day windows.

    Abnormal return for each day = stock_return − SPY_return.
    CAR = cumulative sum of daily abnormal returns over the window.

    Args:
        ticker: Stock ticker symbol.
        window_dates: Output of identify_earnings_window() — maps window
            length to list of trading day Timestamps.
        stock_prices: DataFrame of adjusted close prices (dates × tickers).
        spy_prices: Single-column DataFrame of SPY adjusted close.

    Returns:
        Dict with keys CAR_1d, CAR_3d, CAR_5d. NaN when price data is
        missing or the window extends beyond available data.
    """
    result = {f"CAR_{w}d": np.nan for w in config.CAR_WINDOWS}

    if ticker not in stock_prices.columns:
        logger.debug("compute_car: ticker %s not in price data", ticker)
        return result

    stock_col = stock_prices[ticker].dropna()
    spy_col = spy_prices["SPY"].dropna()

    for w in config.CAR_WINDOWS:
        days = window_dates.get(w, [])
        if not days:
            continue

        # Need the day before the window to compute day-1 return
        all_trading = stock_prices.index
        window_start_pos = all_trading.get_loc(days[0]) if days[0] in all_trading else None
        if window_start_pos is None or window_start_pos == 0:
            continue

        abnormal_returns = []
        valid = True

        for day in days:
            try:
                day_pos = all_trading.get_loc(day)
                prev_day = all_trading[day_pos - 1]

                stock_ret = (stock_col.get(day, np.nan) / stock_col.get(prev_day, np.nan)) - 1
                spy_ret = (spy_col.get(day, np.nan) / spy_col.get(prev_day, np.nan)) - 1

                if np.isnan(stock_ret) or np.isnan(spy_ret):
                    valid = False
                    break

                abnormal_returns.append(stock_ret - spy_ret)
            except (KeyError, ZeroDivisionError):
                valid = False
                break

        if valid and len(abnormal_returns) == w:
            result[f"CAR_{w}d"] = float(np.sum(abnormal_returns))

    return result


# ---------------------------------------------------------------------------
# 4. Beta-adjusted CAR (robustness check)
# ---------------------------------------------------------------------------

def compute_beta_adjusted_car(
    ticker: str,
    window_dates: dict[int, list[pd.Timestamp]],
    stock_prices: pd.DataFrame,
    spy_prices: pd.DataFrame,
    lookback: int = 120,
) -> dict[str, float]:
    """Compute beta-adjusted CAR for robustness checking.

    Beta is estimated via OLS on the lookback window of daily returns
    immediately BEFORE the earnings window (non-overlapping). The expected
    return for each event day is beta × SPY_return. The beta-adjusted
    abnormal return is stock_return − expected_return.

    This is the robustness check only — primary analysis uses market-adjusted
    CAR (compute_car).

    Args:
        ticker: Stock ticker symbol.
        window_dates: Output of identify_earnings_window().
        stock_prices: Adjusted close prices DataFrame.
        spy_prices: SPY adjusted close DataFrame.
        lookback: Number of trading days prior to window for beta estimation.

    Returns:
        Dict with keys beta_CAR_1d, beta_CAR_3d, beta_CAR_5d, and beta.
        NaN when insufficient data for beta estimation or missing prices.
    """
    result = {f"beta_CAR_{w}d": np.nan for w in config.CAR_WINDOWS}
    result["beta"] = np.nan

    if ticker not in stock_prices.columns:
        return result

    all_days = stock_prices.index

    # Identify the start of the event window
    min_window_days = [d for w in config.CAR_WINDOWS for d in window_dates.get(w, [])]
    if not min_window_days:
        return result

    window_start = min(min_window_days)
    try:
        start_pos = all_days.get_loc(window_start)
    except KeyError:
        return result

    # Beta estimation window: lookback days before the event window
    beta_end_pos = start_pos - 1
    beta_start_pos = beta_end_pos - lookback

    if beta_start_pos < 1:
        logger.debug("compute_beta_adjusted_car: insufficient lookback for %s", ticker)
        return result

    beta_days = all_days[beta_start_pos : beta_end_pos + 1]

    stock_px = stock_prices[ticker].reindex(beta_days).dropna()
    spy_px = spy_prices["SPY"].reindex(beta_days).dropna()
    common = stock_px.index.intersection(spy_px.index)

    if len(common) < 30:  # need at least 30 days for a meaningful beta
        return result

    stock_rets = stock_px.reindex(common).pct_change().dropna()
    spy_rets = spy_px.reindex(common).pct_change().dropna()
    common2 = stock_rets.index.intersection(spy_rets.index)
    stock_rets = stock_rets.reindex(common2)
    spy_rets = spy_rets.reindex(common2)

    # OLS beta = cov(stock, spy) / var(spy)
    cov_matrix = np.cov(stock_rets.values, spy_rets.values)
    spy_var = cov_matrix[1, 1]
    if spy_var == 0:
        return result
    beta = cov_matrix[0, 1] / spy_var
    result["beta"] = round(float(beta), 4)

    stock_col = stock_prices[ticker].dropna()
    spy_col = spy_prices["SPY"].dropna()

    for w in config.CAR_WINDOWS:
        days = window_dates.get(w, [])
        if not days:
            continue

        abnormal_returns = []
        valid = True

        for day in days:
            try:
                day_pos = all_days.get_loc(day)
                prev_day = all_days[day_pos - 1]

                stock_ret = (stock_col.get(day, np.nan) / stock_col.get(prev_day, np.nan)) - 1
                spy_ret = (spy_col.get(day, np.nan) / spy_col.get(prev_day, np.nan)) - 1

                if np.isnan(stock_ret) or np.isnan(spy_ret):
                    valid = False
                    break

                expected_ret = beta * spy_ret
                abnormal_returns.append(stock_ret - expected_ret)
            except (KeyError, ZeroDivisionError):
                valid = False
                break

        if valid and len(abnormal_returns) == w:
            result[f"beta_CAR_{w}d"] = float(np.sum(abnormal_returns))

    return result


# ---------------------------------------------------------------------------
# 5. Returns matrix assembly
# ---------------------------------------------------------------------------

def build_returns_matrix(
    features_df: pd.DataFrame,
    price_cache_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Compute CARs for all earnings events and merge with feature matrix.

    Fetches (or loads cached) price data, maps each earnings event to its
    trading day window, computes market-adjusted and beta-adjusted CARs,
    and merges the result with the sentiment features.

    Saves to data/processed/analysis_ready.parquet.

    Args:
        features_df: Output of build_feature_matrix() — one row per
            (ticker, date) earnings event.
        price_cache_path: Path to prices.parquet cache. Defaults to
            config.CACHE_DIR / "prices.parquet".

    Returns:
        Merged DataFrame with all feature columns plus:
            CAR_1d, CAR_3d, CAR_5d,
            beta_CAR_1d, beta_CAR_3d, beta_CAR_5d,
            beta, market_cap, sector.
    """
    if price_cache_path is None:
        price_cache_path = config.CACHE_DIR / "prices.parquet"

    tickers = features_df["ticker"].unique().tolist()
    dates = pd.to_datetime(features_df["date"])

    # Buffer start by 180 days to cover 120-day beta lookback + margin
    start_date = (dates.min() - pd.Timedelta(days=180)).strftime("%Y-%m-%d")
    end_date = (dates.max() + pd.Timedelta(days=10)).strftime("%Y-%m-%d")

    stock_prices, spy_prices = fetch_price_data(
        tickers, start_date, end_date, cache_path=price_cache_path
    )
    trading_days = stock_prices.index

    # Build timing map from raw data
    print("Building earnings timing map from raw call times...")
    raw_df = pd.read_pickle(config.RAW_DIR / "motley-fool-data.pkl")
    timing_map = _build_timing_map(raw_df)
    timing_map = timing_map.rename(columns={"call_date": "date"})
    timing_map["date"] = pd.to_datetime(timing_map["date"]).dt.normalize()

    features = features_df.copy()
    features["date"] = pd.to_datetime(features["date"]).dt.normalize()
    features = features.merge(timing_map, on=["ticker", "date"], how="left")
    # Unknown timing → conservative (after_close=True)
    features["after_close"] = features["after_close"].fillna(True)

    # --- Compute CARs ---
    car_records = []
    n_total = len(features)
    print(f"Computing CARs for {n_total:,} earnings events...")

    for i, row in features.iterrows():
        window_dates = identify_earnings_window(
            pd.Timestamp(row["date"]),
            bool(row["after_close"]),
            trading_days,
        )
        car = compute_car(row["ticker"], window_dates, stock_prices, spy_prices)
        beta_car = compute_beta_adjusted_car(row["ticker"], window_dates, stock_prices, spy_prices)

        car_records.append({
            "ticker": row["ticker"],
            "date": row["date"],
            **car,
            **beta_car,
        })

        if (i + 1) % 1000 == 0:
            print(f"  {i+1:,} / {n_total:,} processed...")

    car_df = pd.DataFrame(car_records)

    # --- Fetch market cap and sector (best-effort from yfinance) ---
    print("\nFetching market cap and sector data...")
    meta_records = []
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            meta_records.append({
                "ticker": ticker,
                "market_cap": info.get("marketCap", np.nan),
                "sector": info.get("sector", np.nan),
            })
        except Exception:
            meta_records.append({"ticker": ticker, "market_cap": np.nan, "sector": np.nan})

    meta_df = pd.DataFrame(meta_records)

    # --- Merge everything ---
    analysis = features.merge(car_df, on=["ticker", "date"], how="left")
    analysis = analysis.merge(meta_df, on="ticker", how="left")
    analysis = analysis.drop(columns=["after_close"])

    # --- Report ---
    car_cols = ["CAR_1d", "CAR_3d", "CAR_5d"]
    complete = analysis[car_cols].notna().all(axis=1).sum()
    dropped = n_total - complete

    print(f"\n{'='*55}")
    print(f"  Returns matrix summary")
    print(f"{'='*55}")
    print(f"  Total earnings events   : {n_total:,}")
    print(f"  Complete (all 3 CARs)   : {complete:,}  ({100*complete/n_total:.1f}%)")
    print(f"  Dropped (missing prices): {dropped:,}  ({100*dropped/n_total:.1f}%)")
    for col in car_cols:
        s = analysis[col].dropna()
        print(f"\n  {col}:")
        print(f"    mean={s.mean():.4f}  std={s.std():.4f}  "
              f"p5={s.quantile(0.05):.4f}  p25={s.quantile(0.25):.4f}  "
              f"p50={s.quantile(0.50):.4f}  p75={s.quantile(0.75):.4f}  "
              f"p95={s.quantile(0.95):.4f}")
    print(f"{'='*55}\n")

    out_path = config.PROCESSED_DIR / "analysis_ready.parquet"
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    analysis.to_parquet(out_path, index=False)
    logger.info("analysis_ready.parquet saved: %d rows → %s", len(analysis), out_path)

    return analysis
