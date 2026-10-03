import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from fastapi.testclient import TestClient

from app.main import app
from app.services.institutional_order_flow_service import institutional_order_flow_service, InstitutionalOrderFlowService
from app.services.strategy_engine import strategy_engine
from app.services.telegram_service import telegram_service
from app.services.instagram_service import instagram_service


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_ohlcv_df():
    """Generates synthetic OHLCV DataFrame with distinct accumulation, breakout, and distribution patterns."""
    np.random.seed(42)
    n = 60
    base_time = datetime(2026, 10, 2, 9, 15)
    times = [base_time + timedelta(minutes=15 * i) for i in range(n)]

    # Create price action with a down-candle then explosive up displacement (Bullish OB)
    closes = [22400.0]
    highs = [22410.0]
    lows = [22390.0]
    opens = [22400.0]
    volumes = [100000]

    for i in range(1, n):
        if i == 20:
            # Down candle (Bullish Order Block base)
            o = 22420.0
            c = 22380.0
            h = 22425.0
            l = 22375.0
            v = 250000
        elif i in [21, 22]:
            # Explosive upward displacement breaking structure
            o = closes[-1]
            c = o + 60.0
            h = c + 10.0
            l = o - 5.0
            v = 400000
        elif i == 40:
            # Up candle (Bearish Order Block base)
            o = 22600.0
            c = 22650.0
            h = 22660.0
            l = 22595.0
            v = 220000
        elif i in [41, 42]:
            # Explosive downward displacement breaking structure
            o = closes[-1]
            c = o - 70.0
            h = o + 5.0
            l = c - 10.0
            v = 450000
        else:
            delta = np.random.normal(2.0, 15.0)
            o = closes[-1]
            c = o + delta
            h = max(o, c) + abs(np.random.normal(5.0, 3.0))
            l = min(o, c) - abs(np.random.normal(5.0, 3.0))
            v = int(np.random.uniform(80000, 180000))

        opens.append(round(o, 2))
        closes.append(round(c, 2))
        highs.append(round(h, 2))
        lows.append(round(l, 2))
        volumes.append(v)

    df = pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    }, index=pd.DatetimeIndex(times))
    return df


def test_bar_delta_and_cvd_calculation(sample_ohlcv_df):
    """Verifies bar delta formula and Cumulative Volume Delta (CVD) calculation."""
    delta_s, cvd_s = institutional_order_flow_service.calculate_bar_delta_and_cvd(sample_ohlcv_df)

    assert len(delta_s) == len(sample_ohlcv_df)
    assert len(cvd_s) == len(sample_ohlcv_df)
    assert isinstance(delta_s.iloc[0], (int, float, np.floating))
    assert isinstance(cvd_s.iloc[-1], (int, float, np.floating))

    # Up candle should have positive delta, down candle negative delta
    up_idx = 21  # Explosive up displacement
    assert delta_s.iloc[up_idx] > 0

    down_idx = 41  # Explosive down displacement
    assert delta_s.iloc[down_idx] < 0


def test_order_block_detection_and_mitigation(sample_ohlcv_df):
    """Tests identification of Bullish & Bearish Order Blocks, 50% Mean Threshold, and mitigation lifecycle."""
    obs = institutional_order_flow_service.detect_order_blocks(sample_ohlcv_df, lookback=50)

    assert "bullish_order_blocks" in obs
    assert "bearish_order_blocks" in obs
    assert "smc_bias" in obs

    # Check Bullish OB structure
    if obs["bullish_order_blocks"]:
        bob = obs["bullish_order_blocks"][0]
        assert bob["type"] == "BULLISH_ORDER_BLOCK"
        assert bob["zone_top"] >= bob["zone_bottom"]
        assert bob["mean_threshold"] == round((bob["zone_top"] + bob["zone_bottom"]) / 2.0, 2)
        assert bob["mitigation_status"] in [
            "UNMITIGATED_FRESH", "TESTED_REJECTED", "REACTING_IN_ZONE",
            "LIQUIDITY_SWEPT_HELD", "BREACHED_INVALIDATED"
        ]

    # Check Bearish OB structure
    if obs["bearish_order_blocks"]:
        sob = obs["bearish_order_blocks"][0]
        assert sob["type"] == "BEARISH_ORDER_BLOCK"
        assert sob["zone_top"] >= sob["zone_bottom"]
        assert sob["mean_threshold"] == round((sob["zone_top"] + sob["zone_bottom"]) / 2.0, 2)


def test_fair_value_gap_detection(sample_ohlcv_df):
    """Tests 3-bar Fair Value Gap (FVG) detection and CE midpoint."""
    fvgs = institutional_order_flow_service.detect_fair_value_gaps(sample_ohlcv_df, lookback=50)
    assert isinstance(fvgs, list)

    for gap in fvgs:
        assert gap["type"] in ["BULLISH_FVG", "BEARISH_FVG"]
        assert gap["gap_top"] >= gap["gap_bottom"]
        assert gap["midpoint_ce"] == round((gap["gap_top"] + gap["gap_bottom"]) / 2.0, 2)
        assert gap["fill_status"] in ["UNFILLED_OPEN", "PARTIALLY_FILLED", "FULLY_MITIGATED"]


def test_institutional_flow_and_activity_timeline(sample_ohlcv_df):
    """Tests institutional phase classification, delta divergence, and timestamped buying/selling timeline."""
    flow = institutional_order_flow_service.analyze_institutional_flow_and_timeline(
        sample_ohlcv_df, symbol="^NSEI"
    )

    assert "institutional_phase" in flow
    assert "phase_badge" in flow
    assert flow["institutional_phase"] in [
        "INSTITUTIONAL_ACCUMULATION",
        "INSTITUTIONAL_MARKUP",
        "INSTITUTIONAL_DISTRIBUTION",
        "INSTITUTIONAL_MARKDOWN",
        "CONSOLIDATION_EQUILIBRIUM",
    ]

    assert "buyer_dominance_pct" in flow
    assert 0.0 <= flow["buyer_dominance_pct"] <= 100.0
    assert "cumulative_volume_delta" in flow
    assert "timeline_events" in flow
    assert isinstance(flow["timeline_events"], list)

    if flow["timeline_events"]:
        evt = flow["timeline_events"][0]
        assert "action" in evt
        assert "actor" in evt
        assert "price_level" in evt
        assert "narrative" in evt


def test_comprehensive_snapshot_generation(sample_ohlcv_df):
    """Tests the full Smart Money & Institutional snapshot generator."""
    snap = institutional_order_flow_service.get_comprehensive_institutional_snapshot(
        symbol="^NSEI",
        df=sample_ohlcv_df,
        live_price=22500.0,
        pcr_data={"pcr_oi": 1.15, "sentiment": "BULLISH"},
    )

    assert snap["symbol"] == "^NSEI"
    assert "order_blocks" in snap
    assert "fair_value_gaps" in snap
    assert "order_flow" in snap
    assert "smart_money_summary" in snap
    assert isinstance(snap["smart_money_summary"], str)


def test_order_block_endpoints(client):
    """Tests FastAPI REST endpoints for order blocks, institutional flow, and smart money snapshot."""
    res_ob = client.get("/api/v1/market/order-blocks/^NSEI")
    assert res_ob.status_code == 200
    data_ob = res_ob.json()
    assert "bullish_order_blocks" in data_ob
    assert "bearish_order_blocks" in data_ob

    res_flow = client.get("/api/v1/market/institutional-flow/^NSEI")
    assert res_flow.status_code == 200
    data_flow = res_flow.json()
    assert "institutional_phase" in data_flow
    assert "timeline_events" in data_flow

    res_sm = client.get("/api/v1/market/smart-money/^NSEI")
    assert res_sm.status_code == 200
    data_sm = res_sm.json()
    assert "order_blocks" in data_sm
    assert "order_flow" in data_sm
    assert "smart_money_summary" in data_sm


def test_strategy_engine_order_block_integration():
    """Tests that StrategyEngine embeds institutional order flow and adjusts stops/targets accordingly."""
    sig = strategy_engine.generate_options_call_put_signal("^NSEI")
    assert sig is not None
    assert "institutional_order_flow" in sig
    iof = sig["institutional_order_flow"]
    assert "phase_badge" in iof
    assert "cumulative_volume_delta" in iof
    assert "buyer_dominance_pct" in iof


def test_telegram_and_instagram_alert_payload_with_order_blocks():
    """Tests that Telegram and Instagram alert messages format Order Blocks and Institutional Flow cleanly."""
    mock_signal = {
        "symbol": "^NSEI",
        "instrument": "NIFTY 50",
        "recommendation": "BUY CALL (CE)",
        "signal_type": "BULLISH",
        "suggested_strike": "22500 CE",
        "option_entry_price": 140.0,
        "option_target_1": 180.0,
        "option_target_2": 220.0,
        "option_stop_loss": 115.0,
        "spot_price": 22480.0,
        "spot_target_1": 22560.0,
        "spot_stop_loss": 22430.0,
        "confidence_score": 85.0,
        "risk_reward_ratio": "1:2.0",
        "lot_size": 65,
        "capital_required_per_lot": 9100.0,
        "est_profit_per_lot_t1": 2600.0,
        "est_profit_per_lot_t2": 5200.0,
        "est_risk_per_lot": 1625.0,
        "confluence_reasons": ["Institutional Demand Zone bounce verified", "Positive CVD accumulation"],
        "institutional_order_flow": {
            "phase_badge": "INSTITUTIONAL_ACCUMULATION",
            "buyer_dominance_pct": 68.5,
            "cumulative_volume_delta": 45000.0,
            "delta_divergence": "BULLISH_ABSORPTION_DIVERGENCE",
            "nearest_bullish_ob": {
                "zone_top": 22450.0,
                "zone_bottom": 22420.0,
                "mitigation_status": "UNMITIGATED_FRESH",
            },
            "nearest_bearish_ob": {
                "zone_top": 22620.0,
                "zone_bottom": 22590.0,
                "mitigation_status": "UNMITIGATED_FRESH",
            }
        },
        "institutional_matrix": {
            "passed_count": 18,
            "total_factors": 21,
            "confluence_percentage": 85.7,
            "regime": "TRENDING_EXPANSION",
        },
    }

    # Telegram format verification
    tg_msg = telegram_service.format_signal_message(mock_signal)
    assert "Demand" in tg_msg
    assert "22420.0" in tg_msg
    assert "INSTITUTIONAL ACCUMULATION" in tg_msg
    assert "CVD: +45,000" in tg_msg

    # Instagram format verification
    ig_msg = instagram_service.format_signal_message(mock_signal)
    assert "Demand OB: ₹22420.0 - ₹22450.0" in ig_msg
    assert "Flow: INSTITUTIONAL ACCUMULATION (CVD: +45,000)" in ig_msg
