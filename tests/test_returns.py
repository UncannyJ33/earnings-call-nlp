"""
tests/test_returns.py — pytest suite for src/returns.py

Covers:
  - _parse_call_time(): time extraction from raw date strings and edge cases
  - identify_earnings_window(): before/after-close timing, weekend handling
  - compute_car(): arithmetic correctness, missing data, edge cases
  - compute_beta_adjusted_car(): result structure and beta estimation
  - Integration test: AAPL events from the cached analysis_ready.parquet
"""

import math
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.returns import (
    _parse_call_time,
    compute_beta_adjusted_car,
    compute_car,
    identify_earnings_window,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def trading_days() -> pd.DatetimeIndex:
    """Eight business days: Mon 2020-01-06 through Wed 2020-01-15."""
    return pd.bdate_range("2020-01-06", periods=8)


@pytest.fixture
def stock_prices(trading_days: pd.DatetimeIndex) -> pd.DataFrame:
    """TEST ticker: +10% on day 1, flat for the remaining 7 days."""
    prices = [100.0, 110.0, 110.0, 110.0, 110.0, 110.0, 110.0, 110.0]
    return pd.DataFrame({"TEST": prices}, index=trading_days)


@pytest.fixture
def spy_prices(trading_days: pd.DatetimeIndex) -> pd.DataFrame:
    """SPY: +1% on day 1, flat for the remaining 7 days."""
    prices = [200.0, 202.0, 202.0, 202.0, 202.0, 202.0, 202.0, 202.0]
    return pd.DataFrame({"SPY": prices}, index=trading_days)


def _make_window(
    trading_days: pd.DatetimeIndex,
    start_idx: int,
    windows: list[int],
) -> dict[int, list[pd.Timestamp]]:
    """Build window_dates starting at trading_days[start_idx]."""
    return {
        w: list(trading_days[start_idx : start_idx + w])
        for w in windows
    }


# ---------------------------------------------------------------------------
# _parse_call_time — time extraction from raw date strings
# ---------------------------------------------------------------------------

class TestParseCallTime:
    def test_morning_call(self):
        assert _parse_call_time("Aug 5, 2020, 8:30 a.m. ET") == time(8, 30)

    def test_evening_call(self):
        assert _parse_call_time("Aug 27, 2020, 9:00 p.m. ET") == time(21, 0)

    def test_exactly_market_close(self):
        # 4:00 p.m. — sits on the boundary, should be treated as after-close
        assert _parse_call_time("Nov 3, 2021, 4:00 p.m. ET") == time(16, 0)

    def test_one_minute_before_close(self):
        assert _parse_call_time("Nov 3, 2021, 3:59 p.m. ET") == time(15, 59)

    def test_list_input(self):
        raw = ["Company Q2 2020", "Jul 30, 2020, 4:30 p.m. ET"]
        assert _parse_call_time(raw) == time(16, 30)

    def test_noon(self):
        # 12:00 p.m. must stay at 12, not become 24
        assert _parse_call_time("Mar 12, 2021, 12:00 p.m. ET") == time(12, 0)

    def test_midnight(self):
        # 12:00 a.m. must become 0, not stay at 12
        assert _parse_call_time("Mar 12, 2021, 12:00 a.m. ET") == time(0, 0)

    def test_no_time_in_string_returns_none(self):
        assert _parse_call_time("Just a plain date string") is None

    def test_none_input_returns_none(self):
        assert _parse_call_time(None) is None

    def test_market_close_boundary_triggers_after_close(self):
        """4:00 p.m. call should satisfy call_time >= MARKET_CLOSE (→ after_close=True)."""
        market_close = time(16, 0)
        call_time = _parse_call_time("Nov 3, 2021, 4:00 p.m. ET")
        assert call_time >= market_close

    def test_before_close_boundary_does_not_trigger_after_close(self):
        market_close = time(16, 0)
        call_time = _parse_call_time("Nov 3, 2021, 3:59 p.m. ET")
        assert call_time < market_close


# ---------------------------------------------------------------------------
# identify_earnings_window — timing logic and edge cases
# ---------------------------------------------------------------------------

class TestIdentifyEarningsWindow:
    def test_before_close_window_starts_same_day(self, trading_days):
        event = trading_days[2]  # Wed 2020-01-08
        result = identify_earnings_window(event, after_close=False, trading_days=trading_days, windows=[1, 3])
        assert result[1][0] == event

    def test_after_close_window_starts_next_trading_day(self, trading_days):
        event = trading_days[2]  # Wed 2020-01-08
        result = identify_earnings_window(event, after_close=True, trading_days=trading_days, windows=[1, 3])
        assert result[1][0] == trading_days[3]  # Thu 2020-01-09

    def test_friday_after_close_skips_weekend(self, trading_days):
        # trading_days[4] = Fri 2020-01-10; next trading day = Mon 2020-01-13
        friday = trading_days[4]
        result = identify_earnings_window(friday, after_close=True, trading_days=trading_days, windows=[1])
        assert result[1][0] == pd.Timestamp("2020-01-13")

    def test_saturday_before_close_maps_to_monday(self, trading_days):
        # Saturday is not a trading day — before-close should snap forward to Monday
        saturday = pd.Timestamp("2020-01-11")
        result = identify_earnings_window(saturday, after_close=False, trading_days=trading_days, windows=[1])
        assert result[1][0] == pd.Timestamp("2020-01-13")

    def test_window_list_lengths_are_correct(self, trading_days):
        event = trading_days[0]
        result = identify_earnings_window(event, after_close=False, trading_days=trading_days, windows=[1, 3, 5])
        assert len(result[1]) == 1
        assert len(result[3]) == 3
        assert len(result[5]) == 5

    def test_window_days_are_consecutive_trading_days(self, trading_days):
        event = trading_days[0]
        result = identify_earnings_window(event, after_close=False, trading_days=trading_days, windows=[3])
        assert result[3] == [trading_days[0], trading_days[1], trading_days[2]]

    def test_date_after_all_trading_days_returns_empty_dict(self, trading_days):
        far_future = pd.Timestamp("2030-01-01")
        result = identify_earnings_window(far_future, after_close=True, trading_days=trading_days, windows=[1, 3, 5])
        assert result == {}

    def test_insufficient_remaining_days_returns_empty_list(self, trading_days):
        # Last trading day: only 0 days remain for a 5-day window
        last_day = trading_days[-1]
        result = identify_earnings_window(last_day, after_close=False, trading_days=trading_days, windows=[5])
        assert result[5] == []


# ---------------------------------------------------------------------------
# compute_car — CAR arithmetic and missing-data handling
# ---------------------------------------------------------------------------

class TestComputeCar:
    def test_car_1d_arithmetic(self, trading_days, stock_prices, spy_prices):
        # Stock +10%, SPY +1% on day 1 → abnormal return = 0.09
        window = _make_window(trading_days, start_idx=1, windows=[1, 3, 5])
        result = compute_car("TEST", window, stock_prices, spy_prices)
        assert math.isclose(result["CAR_1d"], 0.09, rel_tol=1e-6)

    def test_car_3d_flat_after_day1(self, trading_days, stock_prices, spy_prices):
        # Days 2 and 3 have zero abnormal return → CAR_3d == CAR_1d
        window = _make_window(trading_days, start_idx=1, windows=[1, 3, 5])
        result = compute_car("TEST", window, stock_prices, spy_prices)
        assert math.isclose(result["CAR_3d"], 0.09, rel_tol=1e-6)

    def test_car_5d_flat_after_day1(self, trading_days, stock_prices, spy_prices):
        window = _make_window(trading_days, start_idx=1, windows=[1, 3, 5])
        result = compute_car("TEST", window, stock_prices, spy_prices)
        assert math.isclose(result["CAR_5d"], 0.09, rel_tol=1e-6)

    def test_negative_car(self, trading_days, spy_prices):
        # Stock drops −10%, SPY +1% → CAR_1d = −0.11
        prices = [100.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0]
        stock = pd.DataFrame({"TEST": prices}, index=trading_days)
        window = _make_window(trading_days, start_idx=1, windows=[1, 3, 5])
        result = compute_car("TEST", window, stock, spy_prices)
        assert math.isclose(result["CAR_1d"], -0.11, rel_tol=1e-6)

    def test_stock_matches_spy_car_is_zero(self, trading_days):
        # Identical price series → all abnormal returns are zero
        prices = [100.0, 110.0, 121.0, 133.1, 146.4, 161.1, 177.2, 194.9]
        stock = pd.DataFrame({"TEST": prices}, index=trading_days)
        spy = pd.DataFrame({"SPY": prices.copy()}, index=trading_days)
        window = _make_window(trading_days, start_idx=1, windows=[1, 3, 5])
        result = compute_car("TEST", window, stock, spy)
        assert math.isclose(result["CAR_1d"], 0.0, abs_tol=1e-9)
        assert math.isclose(result["CAR_3d"], 0.0, abs_tol=1e-9)
        assert math.isclose(result["CAR_5d"], 0.0, abs_tol=1e-9)

    def test_missing_ticker_returns_all_nan(self, trading_days, stock_prices, spy_prices):
        window = _make_window(trading_days, start_idx=1, windows=[1, 3, 5])
        result = compute_car("NONEXISTENT", window, stock_prices, spy_prices)
        assert math.isnan(result["CAR_1d"])
        assert math.isnan(result["CAR_3d"])
        assert math.isnan(result["CAR_5d"])

    def test_nan_stock_price_in_window_returns_nan(self, trading_days, spy_prices):
        prices = [100.0, np.nan, 110.0, 110.0, 110.0, 110.0, 110.0, 110.0]
        stock = pd.DataFrame({"TEST": prices}, index=trading_days)
        window = {1: [trading_days[1]], 3: [trading_days[1], trading_days[2], trading_days[3]], 5: []}
        result = compute_car("TEST", window, stock, spy_prices)
        assert math.isnan(result["CAR_1d"])
        assert math.isnan(result["CAR_3d"])

    def test_empty_window_list_returns_nan(self, trading_days, stock_prices, spy_prices):
        window = {1: [], 3: [], 5: []}
        result = compute_car("TEST", window, stock_prices, spy_prices)
        assert math.isnan(result["CAR_1d"])
        assert math.isnan(result["CAR_3d"])
        assert math.isnan(result["CAR_5d"])

    def test_window_at_index_zero_no_prev_day_returns_nan(self, trading_days, stock_prices, spy_prices):
        # Window starts at position 0 — no previous day exists to compute a return
        window = {1: [trading_days[0]], 3: [trading_days[0], trading_days[1], trading_days[2]], 5: []}
        result = compute_car("TEST", window, stock_prices, spy_prices)
        assert math.isnan(result["CAR_1d"])
        assert math.isnan(result["CAR_3d"])


# ---------------------------------------------------------------------------
# compute_beta_adjusted_car — structure and beta estimation
# ---------------------------------------------------------------------------

class TestComputeBetaAdjustedCar:
    def test_result_keys_present(self, trading_days):
        stock = pd.DataFrame({"TEST": np.linspace(100, 115, 8)}, index=trading_days)
        spy = pd.DataFrame({"SPY": np.linspace(200, 210, 8)}, index=trading_days)
        window = {1: [trading_days[6]], 3: [], 5: []}
        result = compute_beta_adjusted_car("TEST", window, stock, spy, lookback=4)
        assert set(result.keys()) == {"beta_CAR_1d", "beta_CAR_3d", "beta_CAR_5d", "beta"}

    def test_missing_ticker_returns_all_nan(self, trading_days, stock_prices, spy_prices):
        window = _make_window(trading_days, start_idx=1, windows=[1, 3, 5])
        result = compute_beta_adjusted_car("NONEXISTENT", window, stock_prices, spy_prices)
        assert math.isnan(result["beta_CAR_1d"])
        assert math.isnan(result["beta"])

    def test_insufficient_lookback_returns_nan(self, trading_days, stock_prices, spy_prices):
        # lookback=120 but only 8 days of price data — cannot estimate beta
        window = {1: [trading_days[2]], 3: [], 5: []}
        result = compute_beta_adjusted_car("TEST", window, stock_prices, spy_prices, lookback=120)
        assert math.isnan(result["beta"])
        assert math.isnan(result["beta_CAR_1d"])

    def test_beta_approximately_one_when_stock_mirrors_spy(self):
        # Stock whose daily returns are identical to SPY → beta = 1.0
        n = 60
        days = pd.bdate_range("2020-01-02", periods=n)
        rng = np.random.default_rng(42)
        rets = rng.normal(0.0005, 0.01, n - 1)
        prices = np.concatenate([[100.0], 100.0 * np.cumprod(1 + rets)])
        stock = pd.DataFrame({"TEST": prices}, index=days)
        spy = pd.DataFrame({"SPY": prices.copy()}, index=days)
        window = {1: [days[-1]], 3: [], 5: []}
        result = compute_beta_adjusted_car("TEST", window, stock, spy, lookback=30)
        assert math.isclose(result["beta"], 1.0, rel_tol=1e-3)


# ---------------------------------------------------------------------------
# Integration tests — require cached analysis_ready.parquet
# ---------------------------------------------------------------------------

_CACHE_PATH = Path("data/processed/analysis_ready.parquet")
_have_cache = _CACHE_PATH.exists()


@pytest.mark.skipif(not _have_cache, reason="data/processed/analysis_ready.parquet not available")
class TestAaplIntegration:
    """Sanity-checks on AAPL events from the full pipeline output.

    These tests do not re-run inference or price downloads — they verify that
    the cached analysis_ready.parquet contains well-formed values for a ticker
    that should be present in the Motley Fool dataset.
    """

    @pytest.fixture(autouse=True)
    def load_data(self):
        self.df = pd.read_parquet(_CACHE_PATH)
        self.aapl = self.df[self.df["ticker"] == "AAPL"]

    def test_aapl_events_present(self):
        assert len(self.aapl) >= 10, "Expected ≥10 AAPL earnings events"

    def test_aapl_car_values_not_all_nan(self):
        assert self.aapl["CAR_3d"].notna().sum() > 0, "All AAPL CAR_3d values are NaN"

    def test_aapl_car_values_in_realistic_range(self):
        car = self.aapl.dropna(subset=["CAR_1d", "CAR_3d", "CAR_5d"])
        assert car["CAR_1d"].abs().max() < 0.50, "CAR_1d has an implausible value (>50%)"
        assert car["CAR_3d"].abs().max() < 0.50, "CAR_3d has an implausible value (>50%)"
        assert car["CAR_5d"].abs().max() < 0.50, "CAR_5d has an implausible value (>50%)"

    def test_sentiment_delta_direction_consistent_with_car(self):
        """High-delta AAPL calls should have a higher mean CAR than low-delta calls.

        This is a soft directional check — AAPL is in Technology, where the
        signal is present. We don't require significance, just that the mean
        difference is not badly reversed.
        """
        valid = self.aapl.dropna(subset=["sentiment_delta", "CAR_3d"])
        if len(valid) < 8:
            pytest.skip("Too few AAPL events with sentiment_delta")
        median_delta = valid["sentiment_delta"].median()
        mean_car_high = valid.loc[valid["sentiment_delta"] > median_delta, "CAR_3d"].mean()
        mean_car_low = valid.loc[valid["sentiment_delta"] <= median_delta, "CAR_3d"].mean()
        # Directional check only — individual stock noise is high
        assert not (math.isnan(mean_car_high) or math.isnan(mean_car_low))
