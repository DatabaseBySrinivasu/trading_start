import pytest
import pandas as pd
import numpy as np
from datetime import datetime, time as dtime
import pytz

from app.services.institutional_filter_engine import institutional_filter_engine, InstitutionalFilterEngine
from app.services.strategy_engine import strategy_engine


@pytest.fixture
def sample_trending_df():
    """Generates synthetic trending bullish OHLCV dataframe (50 candles)."""
    np.random.seed(42)
    n = 60
    base_price = 22000.0
    # Consistent upward trend with volume expansion
    trend = np.linspace(0, 500, n)
    noise = np.random.normal(0, 10, n)
    closes = base_price + trend + noise
    opens = closes - np.random.uniform(5, 20, n)
    highs = np.maximum(opens, closes) + np.random.uniform(5, 25, n)
    lows = np.minimum(opens, closes) - np.random.uniform(5, 20, n)
    volumes = np.random.uniform(50000, 150000, n)
    # Last 5 bars have volume surge
    volumes[-5:] = volumes[-5:] * 2.5

    df = pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })
    return df


@pytest.fixture
def sample_choppy_df():
    """Generates synthetic flat choppy sideways OHLCV dataframe (50 candles)."""
    np.random.seed(42)
    n = 60
    base_price = 22000.0
    # Tight range oscillation (consolidation chop)
    noise = np.sin(np.linspace(0, 20, n)) * 30
    closes = base_price + noise
    opens = base_price - noise * 0.5
    highs = np.maximum(opens, closes) + np.random.uniform(2, 10, n)
    lows = np.minimum(opens, closes) - np.random.uniform(2, 10, n)
    volumes = np.random.uniform(10000, 30000, n)

    df = pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })
    return df


def test_choppiness_index_calculation(sample_trending_df, sample_choppy_df):
    """Test 11. Choppiness Index (CHOP) formula and gate validation."""
    # Trending dataframe should have low CHOP (< 50) and pass gate
    chop_trend = institutional_filter_engine.calculate_choppiness_index(sample_trending_df, period=14)
    assert "chop" in chop_trend
    assert "gate_passed" in chop_trend
    assert chop_trend["chop"] < 61.8
    assert chop_trend["gate_passed"] is True

    # Choppy dataframe should have elevated CHOP
    chop_sideways = institutional_filter_engine.calculate_choppiness_index(sample_choppy_df, period=14)
    assert "chop" in chop_sideways
    assert chop_sideways["chop"] > chop_trend["chop"]


def test_ema_slopes_and_vector_angles(sample_trending_df):
    """Test 4. Normalized EMA Slope & Vector Angle calculation."""
    atr_val = 150.0
    slopes = institutional_filter_engine.calculate_ema_slopes(sample_trending_df, atr_val=atr_val, lookback=3)

    assert "slope_ema9" in slopes
    assert "slope_ema21" in slopes
    assert "angle_ema9" in slopes
    assert "is_steep" in slopes
    assert slopes["slope_ema9"] > 0.0  # Positive upward slope on bullish trend
    assert slopes["angle_ema9"] > 0.0


def test_institutional_order_flow_delta_and_cvd(sample_trending_df):
    """Test 8. Institutional Order Flow Proxy (Bar Delta + Cumulative Volume Delta CVD)."""
    flow = institutional_filter_engine.calculate_institutional_flow(sample_trending_df, lookback=20)

    assert "cvd" in flow
    assert "recent_delta" in flow
    assert "flow_direction" in flow
    assert "delta_bias" in flow
    # Trending bullish bars with close > open should yield positive institutional CVD
    assert flow["cvd"] > 0
    assert "BULLISH" in flow["flow_direction"]


def test_retest_confirmation_engine(sample_trending_df):
    """Test 14. Retest Confirmation Engine with tolerance and rejection wicks."""
    broken_level = float(sample_trending_df["high"].iloc[-15])

    # Test bullish retest confirmation
    retest_res = institutional_filter_engine.check_retest_confirmation(
        df=sample_trending_df,
        broken_level=broken_level,
        breakout_direction="BUY CALL (CE)",
        tolerance_pct=0.01,
    )
    assert "retest_stage" in retest_res
    assert "retest_confirmed" in retest_res
    assert "broken_level" in retest_res
    assert retest_res["broken_level"] == broken_level


def test_risk_reward_validation():
    """Test 13. Mandatory Risk-Reward Gate (minimum 1:1.5 required)."""
    # Valid trade: 200 entry, 215 T1 (+15 reward), 190 SL (-10 risk) -> R:R = 1.5
    valid_rr = institutional_filter_engine.evaluate_risk_reward(
        entry=200.0,
        target_1=215.0,
        stop_loss=190.0,
        direction="BUY CALL (CE)",
        min_rr=1.5,
    )
    assert valid_rr["is_valid"] is True
    assert valid_rr["rr_ratio"] >= 1.5
    assert "1:" in valid_rr["rr_label"]

    # Invalid trade: 200 entry, 205 T1 (+5 reward), 190 SL (-10 risk) -> R:R = 0.5 (Rejected)
    invalid_rr = institutional_filter_engine.evaluate_risk_reward(
        entry=200.0,
        target_1=205.0,
        stop_loss=190.0,
        direction="BUY CALL (CE)",
        min_rr=1.5,
    )
    assert invalid_rr["is_valid"] is False
    assert invalid_rr["rr_ratio"] < 1.5


def test_market_regime_classification():
    """Test 9. Market Regime Classifier (Expansion, Chop, Volatility, Mean Reversion)."""
    # 1. Trending expansion: high ADX, low CHOP
    reg_exp = institutional_filter_engine.detect_market_regime(
        adx=32.0, chop=32.0, atr_pct=1.2, vix=14.5, cpr_type="AVERAGE"
    )
    assert reg_exp["regime"] == "TRENDING_EXPANSION"
    assert "ACTIVE_TREND_FOLLOWING" in reg_exp["action_directive"]

    # 2. Chop regime: low ADX, high CHOP
    reg_chop = institutional_filter_engine.detect_market_regime(
        adx=16.0, chop=65.0, atr_pct=0.5, vix=12.0, cpr_type="WIDE_CPR"
    )
    assert reg_chop["regime"] == "RANGE_BOUND_CHOP"
    assert "ALERT_SUPPRESSION" in reg_chop["action_directive"]

    # 3. High Volatility Expansion
    reg_vol = institutional_filter_engine.detect_market_regime(
        adx=28.0, chop=45.0, atr_pct=2.5, vix=24.0, cpr_type="AVERAGE"
    )
    assert reg_vol["regime"] == "VOLATILITY_EXPANSION"


def test_killzone_timing_filter():
    """Test 21. Killzone Timing Filter for Indian Equity & MCX Commodity sessions."""
    ist_tz = pytz.timezone("Asia/Kolkata")

    # 1. Opening Bell (09:30 IST on Wednesday)
    wed_open = ist_tz.localize(datetime(2026, 10, 7, 9, 30, 0))
    kz_open = institutional_filter_engine.evaluate_killzone_timing(now_dt=wed_open, is_commodity=False)
    assert kz_open["killzone"] == "OPENING_BELL_EXPANSION"
    assert kz_open["allow_new_entries"] is True

    # 2. Midday Lunch Chop (12:30 IST on Wednesday)
    wed_lunch = ist_tz.localize(datetime(2026, 10, 7, 12, 30, 0))
    kz_lunch = institutional_filter_engine.evaluate_killzone_timing(now_dt=wed_lunch, is_commodity=False)
    assert kz_lunch["killzone"] == "MIDDAY_LUNCH_CHOP"
    assert kz_lunch["score_modifier"] == 10.0  # Enforces +10% score selectivity

    # 3. Power Hour (14:45 IST on Wednesday)
    wed_power = ist_tz.localize(datetime(2026, 10, 7, 14, 45, 0))
    kz_power = institutional_filter_engine.evaluate_killzone_timing(now_dt=wed_power, is_commodity=False)
    assert kz_power["killzone"] == "POWER_HOUR"
    assert kz_power["allow_new_entries"] is True

    # 4. MCX Evening Prime Session (19:30 IST on Wednesday)
    kz_mcx = institutional_filter_engine.evaluate_killzone_timing(now_dt=ist_tz.localize(datetime(2026, 10, 7, 19, 30, 0)), is_commodity=True)
    assert kz_mcx["killzone"] == "MCX_EVENING_SESSION"
    assert kz_mcx["allow_new_entries"] is True

    # 5. Weekend Closed (Sunday)
    sun_dt = ist_tz.localize(datetime(2026, 10, 4, 11, 0, 0))
    kz_sun = institutional_filter_engine.evaluate_killzone_timing(now_dt=sun_dt, is_commodity=False)
    assert kz_sun["killzone"] == "WEEKEND_CLOSED"
    assert kz_sun["allow_new_entries"] is False


def test_htf_liquidity_levels(sample_trending_df):
    """Test 15. Higher Timeframe (HTF) Liquidity Levels Map."""
    cur_price = float(sample_trending_df["close"].iloc[-1])
    htf = institutional_filter_engine.calculate_htf_liquidity_levels(sample_trending_df, cur_price)

    assert "htf_support" in htf
    assert "htf_resistance" in htf
    assert "pwh" in htf
    assert "pwl" in htf
    assert htf["htf_support"] <= cur_price
    assert htf["htf_resistance"] >= cur_price


def test_evaluate_21_factors_comprehensive(sample_trending_df):
    """Test full 21-Factor Institutional Matrix evaluation and scoring."""
    cur_price = float(sample_trending_df["close"].iloc[-1])

    matrix = institutional_filter_engine.evaluate_21_factors(
        symbol="^NSEI",
        df=sample_trending_df,
        cur_price=cur_price,
        indicators={
            "rsi": 62.5,
            "ema_9": cur_price - 10,
            "ema_21": cur_price - 30,
            "ema_50": cur_price - 70,
            "ema_200": cur_price - 200,
            "supertrend_direction": 1,
            "adx": 28.5,
            "atr": 120.0,
        },
        cpr_data={"pdh": cur_price + 50, "pdl": cur_price - 150, "status": "ABOVE_PDH_BULLISH_BREAKOUT"},
        pattern_data={
            "structure_breakout": "BULLISH_BOS",
            "liquidity_sweeps": {"detected": False},
            "candlestick_patterns": [{"name": "Bullish Engulfing", "pattern": "Bullish Engulfing", "type": "BULLISH"}],
        },
        vix_data={"vix": 13.8, "india_vix": {"regime": "LOW_VOLATILITY"}, "gift_nifty": {"gap_points": 45.0, "bias": "MODERATE_GAP_UP"}},
        pcr_data={"pcr_oi": 1.25, "sentiment": "BULLISH"},
        proposed_trade={
            "recommendation": "BUY CALL (CE)",
            "spot_price": cur_price,
            "spot_target_1": cur_price + 100.0,
            "spot_stop_loss": cur_price - 50.0,
            "confidence_score": 82.0,
            "bullish_score": 85.0,
            "bearish_score": 15.0,
            "expected_gain_pts": 15.0,
        },
    )

    assert matrix["total_factors"] == 21
    assert len(matrix["factors"]) == 21
    assert matrix["passed_count"] >= 15
    assert matrix["confluence_percentage"] > 70.0
    assert "regime" in matrix
    assert "chop_index" in matrix
    assert "cvd" in matrix
    assert "killzone" in matrix
    assert "risk_reward" in matrix


def test_strategy_engine_integration():
    """Verify that strategy_engine generates full 21-factor institutional_matrix."""
    sig = strategy_engine.generate_options_call_put_signal("^NSEI")

    assert "institutional_matrix" in sig
    matrix = sig["institutional_matrix"]
    assert matrix["total_factors"] == 21
    assert len(matrix["factors"]) == 21
    assert "passed_count" in matrix
    assert "confluence_percentage" in matrix
    assert "regime" in matrix
    assert "chop_index" in matrix
    assert "cvd" in matrix
    assert "killzone" in matrix

    # Check that each of the 21 factors contains expected keys
    for f in matrix["factors"]:
        assert "name" in f
        assert "status" in f
        assert f["status"] in ["PASS", "WATCH", "FAIL"]
        assert "passed" in f
        assert "value" in f
        assert "description" in f
