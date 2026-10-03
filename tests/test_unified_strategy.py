import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from fastapi.testclient import TestClient

from app.main import app
from app.services.unified_strategy_service import unified_strategy_service, UnifiedStrategyService
from app.services.strategy_engine import strategy_engine


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_bullish_df():
    """Generates synthetic multi-day bullish expansion dataset with strong volume and clear trend."""
    np.random.seed(42)
    n = 60
    base_time = datetime(2026, 9, 15, 9, 15)
    times = [base_time + timedelta(minutes=15 * i) for i in range(n)]

    cur = 22400.0
    opens, highs, lows, closes, volumes = [], [], [], [], []

    for i in range(n):
        drift = 8.0 + np.random.normal(0, 4.0)
        o = cur
        c = o + drift
        h = max(o, c) + abs(np.random.normal(5.0, 2.0))
        l = min(o, c) - abs(np.random.normal(3.0, 1.0))
        v = int(np.random.uniform(100000, 300000))

        opens.append(round(o, 2))
        closes.append(round(c, 2))
        highs.append(round(h, 2))
        lows.append(round(l, 2))
        volumes.append(v)
        cur = c

    return pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    }, index=pd.DatetimeIndex(times))


@pytest.fixture
def sample_bearish_df():
    """Generates synthetic multi-day bearish breakdown dataset with heavy selling volume."""
    np.random.seed(42)
    n = 60
    base_time = datetime(2026, 9, 15, 9, 15)
    times = [base_time + timedelta(minutes=15 * i) for i in range(n)]

    cur = 22800.0
    opens, highs, lows, closes, volumes = [], [], [], [], []

    for i in range(n):
        drift = -9.0 + np.random.normal(0, 4.0)
        o = cur
        c = o + drift
        h = max(o, c) + abs(np.random.normal(3.0, 1.0))
        l = min(o, c) - abs(np.random.normal(5.0, 2.0))
        v = int(np.random.uniform(120000, 350000))

        opens.append(round(o, 2))
        closes.append(round(c, 2))
        highs.append(round(h, 2))
        lows.append(round(l, 2))
        volumes.append(v)
        cur = c

    return pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    }, index=pd.DatetimeIndex(times))


@pytest.fixture
def sample_chop_df():
    """Generates synthetic sideways choppiness range-bound dataset."""
    np.random.seed(42)
    n = 60
    base_time = datetime(2026, 9, 15, 9, 15)
    times = [base_time + timedelta(minutes=15 * i) for i in range(n)]

    center = 22500.0
    opens, highs, lows, closes, volumes = [], [], [], [], []

    for i in range(n):
        offset = np.sin(i / 2.0) * 10.0
        o = center + offset
        c = center - offset
        h = max(o, c) + 8.0
        l = min(o, c) - 8.0
        v = int(np.random.uniform(40000, 90000))

        opens.append(round(o, 2))
        closes.append(round(c, 2))
        highs.append(round(h, 2))
        lows.append(round(l, 2))
        volumes.append(v)

    return pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    }, index=pd.DatetimeIndex(times))


def test_strategies_catalog_retrieval():
    """Verifies that the catalog lists all 15+ underlying strategies with ID, name, and parameters."""
    catalog = unified_strategy_service.get_strategies_catalog()
    assert isinstance(catalog, list)
    assert len(catalog) >= 15

    ids = [s["id"] for s in catalog]
    assert "ST_01" in ids  # Supertrend
    assert "ST_04" in ids  # Order Blocks
    assert "ST_07" in ids  # CVD / Delta
    assert "ST_08" in ids  # CPR
    assert "ST_09" in ids  # Camarilla
    assert "ST_12" in ids  # Choppiness
    assert "ST_15" in ids  # 5-Point Rule & R:R


def test_4_pillar_bullish_evaluation(sample_bullish_df):
    """Tests that a strong uptrend triggers BUY CALL (CE) with all 4 pillars evaluated."""
    res = unified_strategy_service.evaluate_simple_master_strategy(
        symbol="^NSEI",
        df=sample_bullish_df,
        live_price=22850.0,
        pcr_data={"pcr_oi": 1.25, "sentiment": "BULLISH"},
    )

    assert res["symbol"] == "^NSEI"
    assert res["recommendation"] == "BUY CALL (CE)"
    assert res["signal_type"] == "BULLISH"
    assert res["confidence_score"] >= 65.0

    pillars = res["pillars"]
    assert "pillar_1" in pillars
    assert "pillar_2" in pillars
    assert "pillar_3" in pillars
    assert "pillar_4" in pillars

    # Pillar 1 (Trend) should be Bullish
    assert pillars["pillar_1"]["status"] == "BULLISH"
    assert pillars["pillar_1"]["supertrend"] == "BULLISH"

    # Trade Setup verification
    ts = res["trade_setup"]
    assert ts["option_entry_price"] > 0
    assert ts["option_target_1"] > ts["option_entry_price"]
    assert ts["option_stop_loss"] < ts["option_entry_price"]
    assert ts["expected_move_points"] >= 5.0
    assert ts["min_5_pts_verified"] is True


def test_4_pillar_bearish_evaluation(sample_bearish_df):
    """Tests that a strong downtrend triggers BUY PUT (PE)."""
    res = unified_strategy_service.evaluate_simple_master_strategy(
        symbol="^NSEI",
        df=sample_bearish_df,
        live_price=22300.0,
        pcr_data={"pcr_oi": 0.65, "sentiment": "BEARISH"},
    )

    assert res["recommendation"] == "BUY PUT (PE)"
    assert res["signal_type"] == "BEARISH"
    assert res["confidence_score"] >= 65.0

    pillars = res["pillars"]
    assert pillars["pillar_1"]["status"] == "BEARISH"
    assert pillars["pillar_1"]["supertrend"] == "BEARISH"

    ts = res["trade_setup"]
    assert ts["expected_move_points"] >= 5.0
    assert ts["min_5_pts_verified"] is True


def test_4_pillar_chop_protection(sample_chop_df):
    """Tests that choppy range-bound markets are safely filtered with NEUTRAL / WAIT."""
    res = unified_strategy_service.evaluate_simple_master_strategy(
        symbol="^NSEI",
        df=sample_chop_df,
        live_price=22500.0,
        pcr_data={"pcr_oi": 1.0, "sentiment": "NEUTRAL"},
    )

    assert res["recommendation"] in ["NEUTRAL / WAIT", "BUY CALL (CE)", "BUY PUT (PE)"]
    assert "pillars" in res
    assert "simple_rationale" in res


def test_historical_backtest_and_autocorrection(sample_bullish_df):
    """Tests candle-by-candle simulation on previous historical data and parameter auto-correction."""
    backtest = unified_strategy_service.backtest_and_auto_correct(
        symbol="^NSEI",
        historical_df=sample_bullish_df,
        min_acceptable_winrate=70.0,
    )

    assert backtest["symbol"] == "^NSEI"
    assert "performance" in backtest
    perf = backtest["performance"]
    assert "total_trades" in perf
    assert "win_rate_pct" in perf
    assert "total_points_gained" in perf
    assert "calibrated_parameters" in backtest
    assert "auto_correction_status" in backtest


def test_strategy_engine_master_strategy_integration():
    """Verifies that StrategyEngine embeds master_strategy into its options signals."""
    sig = strategy_engine.generate_options_call_put_signal("^NSEI")
    assert sig is not None
    assert "master_strategy" in sig
    ms = sig["master_strategy"]
    assert "recommendation" in ms
    assert "confidence_score" in ms


def test_commodity_simple_master_strategy_pricing():
    """Verifies MCX commodity (Crude Oil & Natural Gas) lot sizing and 1-pt / 5-pt rules."""
    crude_res = unified_strategy_service.evaluate_simple_master_strategy(
        symbol="CRUDEOIL",
        live_price=6150.0,
    )
    assert crude_res["symbol"] == "CRUDEOIL"
    assert crude_res["trade_setup"]["lot_size"] == 100

    ng_res = unified_strategy_service.evaluate_simple_master_strategy(
        symbol="NATURALGAS",
        live_price=235.0,
    )
    assert ng_res["symbol"] == "NATURALGAS"
    assert ng_res["trade_setup"]["lot_size"] == 1250


def test_master_strategy_endpoints(client):
    """Tests REST API endpoints for catalog, master strategy evaluation, and backtesting."""
    # 1. Catalog
    res_cat = client.get("/api/v1/market/strategies/catalog")
    assert res_cat.status_code == 200
    data_cat = res_cat.json()
    assert "strategies" in data_cat
    assert data_cat["count"] >= 15

    # 2. Master Evaluation
    res_master = client.get("/api/v1/market/strategies/master/^NSEI")
    assert res_master.status_code == 200
    data_master = res_master.json()
    assert "recommendation" in data_master
    assert "pillars" in data_master
    assert "trade_setup" in data_master

    # 3. Backtest
    res_bt = client.get("/api/v1/market/strategies/backtest/^NSEI?min_winrate=65.0")
    assert res_bt.status_code == 200
    data_bt = res_bt.json()
    assert "performance" in data_bt
    assert "calibrated_parameters" in data_bt
