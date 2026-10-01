import pytest
import os
import json
import time
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from app.main import app
from app.services.trade_audit_service import TradeAuditService, trade_audit_service


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def audit_service(tmp_path):
    """Creates an isolated TradeAuditService instance for testing with a temp logs dir."""
    svc = TradeAuditService()
    svc.logs_dir = tmp_path / "logs"
    svc.logs_dir.mkdir(parents=True, exist_ok=True)
    svc.alerts_file = svc.logs_dir / "trade_alerts.jsonl"
    svc.corrections_file = svc.logs_dir / "trade_corrections.jsonl"
    svc.audit_log_file = svc.logs_dir / "trade_audit.log"
    svc.stats_file = svc.logs_dir / "audit_stats.json"
    svc.active_alerts = {}
    svc.completed_alerts = []
    svc.corrections_history = []
    svc.symbol_loss_streak = {}
    svc.symbol_confidence_boost = {}
    return svc


def test_log_trade_alert_persists_to_disk(audit_service):
    """Verifies that trade alerts are recorded to local JSONL and human-readable text logs."""
    sample_signal = {
        "symbol": "^NSEI",
        "instrument": "NIFTY 50",
        "recommendation": "BUY CALL (CE)",
        "signal_type": "BULLISH",
        "suggested_strike": "22500 CE",
        "spot_price": 22450.0,
        "option_entry_price": 140.0,
        "option_target_1": 155.0,
        "option_target_2": 180.0,
        "option_stop_loss": 125.0,
        "spot_levels": {
            "spot_entry": 22450.0,
            "spot_target_1": 22520.0,
            "spot_target_2": 22600.0,
            "spot_stop_loss": 22390.0,
        },
        "confidence_score": 75.0,
        "confluence_reasons": ["Supertrend Bullish", "RSI Momentum +62"],
        "lot_size": 75,
    }

    record = audit_service.log_trade_alert(
        signal=sample_signal,
        channels=["TELEGRAM", "INSTAGRAM"],
        delivery_status="SENT",
    )

    assert record["status"] == "ACTIVE"
    assert record["symbol"] == "^NSEI"
    assert record["suggested_strike"] == "22500 CE"
    assert record["expected_option_gain_pts"] == 15.0

    # Verify JSONL file exists and contains record
    assert audit_service.alerts_file.exists()
    with open(audit_service.alerts_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["alert_id"] == record["alert_id"]
        assert data["recommendation"] == "BUY CALL (CE)"

    # Verify text log exists
    assert audit_service.audit_log_file.exists()
    logs = audit_service.get_recent_logs()
    assert len(logs) > 0
    assert "TRADE ALERT LOGGED" in logs[-1]


def test_audit_detects_target_1_and_trails_stop_loss(audit_service):
    """Verifies that achieving Target 1 (+5 pts) triggers profit booking and trails SL to cost."""
    sample_signal = {
        "symbol": "^NSEI",
        "instrument": "NIFTY 50",
        "recommendation": "BUY CALL (CE)",
        "suggested_strike": "22500 CE",
        "spot_price": 22450.0,
        "option_entry_price": 140.0,
        "option_target_1": 150.0,
        "option_target_2": 190.0,
        "option_stop_loss": 125.0,
        "spot_levels": {
            "spot_entry": 22450.0,
            "spot_target_1": 22470.0,
            "spot_target_2": 22580.0,
            "spot_stop_loss": 22400.0,
        },
        "confidence_score": 80.0,
    }

    record = audit_service.log_trade_alert(sample_signal)
    alert_id = record["alert_id"]

    # Mock market data moving up to Target 1 spot price (22470 >= 22470, option moves 140 -> 150, below T2 190)
    with patch("app.services.market_data.market_data_service.get_live_price", return_value={"price": 22470.0}):
        with patch("app.services.telegram_service.telegram_service.send_message") as mock_tg:
            with patch("app.services.instagram_service.instagram_service.send_message") as mock_ig:
                corrections = audit_service.check_and_self_correct(force_dispatch=True)

                assert len(corrections) == 1
                corr = corrections[0]
                assert corr["type"] == "TARGET_1_HIT_TRAIL_SL"
                assert corr["is_success"] is True
                assert "TARGET 1" in corr["reason"].upper()
                assert "TRAIL STOP LOSS" in corr["action_directive"].upper()
                assert audit_service.active_alerts[alert_id]["target_1_hit"] is True
                # Verify SL was trailed to breakeven entry
                assert audit_service.active_alerts[alert_id]["option_stop_loss"] == 140.0

                # Verify dispatch occurred
                mock_tg.assert_called_once()
                mock_ig.assert_called_once()


def test_audit_detects_stop_loss_breach_and_triggers_exit_correction(audit_service):
    """Verifies that an adverse move breaching Stop Loss triggers an immediate exit self-correction alert."""
    sample_signal = {
        "symbol": "^NSEI",
        "instrument": "NIFTY 50",
        "recommendation": "BUY CALL (CE)",
        "suggested_strike": "22500 CE",
        "spot_price": 22450.0,
        "option_entry_price": 140.0,
        "option_target_1": 150.0,
        "option_target_2": 170.0,
        "option_stop_loss": 125.0,
        "spot_levels": {
            "spot_entry": 22450.0,
            "spot_target_1": 22500.0,
            "spot_target_2": 22580.0,
            "spot_stop_loss": 22400.0,
        },
        "confidence_score": 70.0,
    }

    record = audit_service.log_trade_alert(sample_signal)
    alert_id = record["alert_id"]

    # Mock market data dropping below Stop Loss (22380 <= 22400)
    with patch("app.services.market_data.market_data_service.get_live_price", return_value={"price": 22380.0}):
        with patch("app.services.telegram_service.telegram_service.send_message") as mock_tg:
            with patch("app.services.instagram_service.instagram_service.send_message") as mock_ig:
                corrections = audit_service.check_and_self_correct(force_dispatch=True)

                assert len(corrections) == 1
                corr = corrections[0]
                assert corr["type"] == "STOP_LOSS_EXIT"
                assert corr["is_success"] is False
                assert "EXIT POSITION IMMEDIATELY" in corr["action_directive"]

                # Alert should now be closed and moved from active to completed
                assert alert_id not in audit_service.active_alerts
                assert len(audit_service.completed_alerts) == 1
                assert audit_service.completed_alerts[0]["status"] == "STOP_LOSS_HIT"

                # Corrections JSONL should have the event recorded
                assert audit_service.corrections_file.exists()


def test_self_learning_recalibration_streak(audit_service):
    """Verifies that consecutive losses dynamically raise the confluence threshold for that symbol."""
    sym = "^NSEBANK"
    assert audit_service.get_symbol_confidence_threshold(sym) == 60.0

    # Record 1 loss
    audit_service._update_self_learning_feedback(sym, is_success=False)
    assert audit_service.symbol_loss_streak[sym] == 1

    # Record 2nd consecutive loss -> should add +15% penalty
    audit_service._update_self_learning_feedback(sym, is_success=False)
    assert audit_service.symbol_loss_streak[sym] == 2
    assert audit_service.get_symbol_confidence_threshold(sym) == 75.0

    # Win resets streak to 0
    audit_service._update_self_learning_feedback(sym, is_success=True)
    assert audit_service.symbol_loss_streak[sym] == 0
    assert audit_service.get_symbol_confidence_threshold(sym) == 60.0


def test_audit_api_endpoints(client):
    """Tests all FastAPI endpoints under /api/v1/audit/."""
    # 1. Summary
    res = client.get("/api/v1/audit/summary")
    assert res.status_code == 200
    data = res.json()
    assert "total_alerts_logged" in data
    assert "accuracy_win_rate_pct" in data
    assert "log_files" in data
    assert "technical_factors_log" in data["log_files"]

    # 2. Alerts
    res = client.get("/api/v1/audit/alerts")
    assert res.status_code == 200
    data = res.json()
    assert "alerts" in data

    # 3. Corrections
    res = client.get("/api/v1/audit/corrections")
    assert res.status_code == 200
    data = res.json()
    assert "corrections" in data

    # 4. Logs
    res = client.get("/api/v1/audit/logs")
    assert res.status_code == 200
    data = res.json()
    assert "logs" in data

    # 5. Technical Factors Logs
    res_tf = client.get("/api/v1/audit/technical-factors")
    assert res_tf.status_code == 200
    data_tf = res_tf.json()
    assert "logs" in data_tf
    assert "log_file" in data_tf

    # 6. Check-Now trigger
    res = client.post("/api/v1/audit/check-now")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "COMPLETED"
    assert "corrections_triggered_count" in data


def test_telegram_quantpulse_message_formatting():
    """Verifies that format_signal_message conforms exactly to QuantPulse Pro template."""
    from app.services.telegram_service import telegram_service

    # 1. Test NIFTY Put Signal
    nifty_sig = {
        "symbol": "^NSEI",
        "instrument": "NIFTY 50",
        "recommendation": "BUY PUT (PE)",
        "suggested_strike": "22400 PE",
        "strike_price": 22400,
        "spot_price": 22421.95,
        "option_entry_price": 101.0,
        "option_target_1": 146.45,
        "option_target_2": 186.85,
        "option_stop_loss": 72.72,
        "expected_option_gain_pts": 45.45,
        "risk_reward_ratio": "1:1.3",
        "lot_size": 65,
        "capital_required_per_lot": 6565.0,
        "est_profit_per_lot_t1": 2954.25,
        "est_profit_per_lot_t2": 5580.25,
        "est_risk_per_lot": 1838.2,
        "spot_levels": {
            "spot_target_1": 22305.36,
            "spot_stop_loss": 22511.64,
        },
        "confidence_score": 110.0,
        "price_action_momentum": {
            "momentum_score": 24.0,
            "momentum_regime": "MODERATE_BULLISH_MOMENTUM",
        },
        "pcr": {
            "pcr_oi": 1.16,
            "sentiment": "MODERATELY BULLISH",
        },
        "order_blocks": {
            "smc_bias": "NEUTRAL RANGE",
        },
        "confluence_reasons": [
            "Fibonacci: Price below 50% retracement, testing Golden Pocket support (₹22443.66)",
            "Supertrend: Bearish trend active (Trailing Resistance: ₹22475.03)",
            "EMA: Perfect Bearish Alignment (9 < 21 < 50 EMA: ₹22386.0 < ₹22432.1 < ₹22539.6)",
            "RSI (14): Bearish Momentum (38.5)",
        ],
        "institutional_matrix": {
            "passed_count": 9,
            "total_factors": 21,
            "confluence_percentage": 42.9,
            "regime": "MEAN REVERSION",
        },
    }
    msg_nifty = telegram_service.format_signal_message(nifty_sig)
    assert "⚡️ <b>QUANTPULSE PRO: LIVE MARKET ALERT</b> ⚡️" in msg_nifty
    assert "📊 <b>Instrument:</b> NIFTY 50 (^NSEI)" in msg_nifty
    assert "🎯 <b>Action:</b> 🔴 BUY PUT (PE)" in msg_nifty
    assert "🏷️ <b>Suggested Strike:</b> 22400 PE" in msg_nifty
    assert "🚀 <b>Expected Move:</b> +45.45 Pts (Min 5 Pts Verified)" in msg_nifty
    assert "💰 <b>OPTION CONTRACT PREMIUM (BUY):</b>" in msg_nifty
    assert "▶️ <b>Buy Entry Price:</b> ₹101.00" in msg_nifty
    assert "🎯 <b>Target 1:</b> ₹146.45 (+45.45 pts)" in msg_nifty
    assert "🎯 <b>Target 2:</b> ₹186.85" in msg_nifty
    assert "🛑 <b>Stop Loss:</b> ₹72.72" in msg_nifty
    assert "⚖️ <b>Risk-Reward:</b> 1:1.3" in msg_nifty
    assert "📦 <b>LOT SIZING & CAPITAL METRICS:</b>" in msg_nifty
    assert "• <b>Lot Sizing:</b> 65 Qty / Lot" in msg_nifty
    assert "• <b>Capital / Lot:</b> ₹6,565.00" in msg_nifty
    assert "• <b>Projected Gain (T1):</b> +₹2,954.25 / lot" in msg_nifty
    assert "• <b>Projected Gain (T2):</b> +₹5,580.25 / lot" in msg_nifty
    assert "• <b>Max Risk (SL):</b> -₹1,838.20 / lot" in msg_nifty
    assert "📍 <b>UNDERLYING SPOT REFERENCE:</b>" in msg_nifty
    assert "• <b>Spot LTP:</b> ₹22,421.95 ₹" in msg_nifty
    assert "• <b>Spot Target 1:</b> ₹22,305.36 | <b>Stop:</b> ₹22,511.64" in msg_nifty
    assert "🔥 <b>CONFLUENCE SIGNALS (Score: 110.0%):</b>" in msg_nifty
    assert "⚡️ <b>Momentum:</b> MODERATE BULLISH MOMENTUM (+24.0)" in msg_nifty
    assert "📊 <b>PCR (OI):</b> 1.16 (MODERATELY BULLISH)" in msg_nifty
    assert "🏛️ <b>SMC Bias:</b> NEUTRAL RANGE" in msg_nifty
    assert "💡 <b>Key Technical Factors:</b>" in msg_nifty
    assert "• Fibonacci: Price below 50% retracement, testing Golden Pocket support" in msg_nifty
    assert "• Supertrend: Bearish trend active" in msg_nifty
    assert "• EMA: Perfect Bearish Alignment (9 &lt; 21 &lt; 50 EMA: ₹22386.0 &lt; ₹22432.1 &lt; ₹22539.6)" in msg_nifty
    assert "• RSI (14): Bearish Momentum (38.5)" in msg_nifty
    assert "🏛️ <b>21-FACTOR INSTITUTIONAL MATRIX:</b>" in msg_nifty
    assert "• <b>Filter Confluence:</b> 9/21 PASS (42.9%)" in msg_nifty
    assert "• <b>Market Regime:</b> MEAN REVERSION" in msg_nifty
    assert "🤖 <b>QuantPulse India Pro Terminal</b>" in msg_nifty


def test_technical_factors_logging_persistence(audit_service):
    """Verifies that full technical factors diagnostic report is written to trade_technical_factors.log."""
    audit_service.technical_factors_file = audit_service.logs_dir / "trade_technical_factors.log"

    sample_signal = {
        "symbol": "^NSEI",
        "instrument": "NIFTY 50",
        "recommendation": "BUY CALL (CE)",
        "suggested_strike": "23400 CE",
        "strike_price": 23400,
        "spot_price": 23380.0,
        "spot_target_1": 23460.0,
        "spot_target_2": 23550.0,
        "spot_stop_loss": 23300.0,
        "option_entry_price": 180.0,
        "option_target_1": 234.0,
        "option_target_2": 288.0,
        "option_target_3": 360.0,
        "option_long_target": 450.0,
        "option_stop_loss": 135.0,
        "lot_size": 65,
        "risk_reward_ratio": "1:2.0",
        "confidence_score": 85.0,
        "institutional_matrix": {
            "symbol": "^NSEI",
            "confluence_percentage": 85.7,
            "passed_count": 18,
            "watch_count": 2,
            "fail_count": 1,
            "factors": [
                {"id": 1, "name": "Volume Surge Filter", "status": "PASS", "value": "1.85x", "description": "Volume surge verified"},
                {"id": 11, "name": "Choppiness Index", "status": "PASS", "value": "34.2", "description": "Directional expansion confirmed"},
            ],
        },
        "indicators": {
            "rsi_14": 62.5,
            "ema_9": 23350.0,
            "ema_21": 23300.0,
            "supertrend": 23280.0,
            "supertrend_signal": "BULLISH",
            "macd": 14.5,
            "macd_signal": 8.2,
            "atr": 95.0,
        },
        "pcr": {
            "pcr_oi": 1.35,
            "sentiment": "BULLISH",
            "max_pain_strike": 23400,
            "total_put_oi": 14500000,
            "total_call_oi": 10740000,
        },
        "price_action_momentum": {
            "momentum_score": 45.0,
            "momentum_regime": "STRONG_BULLISH_IMPULSE",
            "trend_structure": "HIGHER_HIGHS_HIGHER_LOWS",
            "bos_status": "BOS_BULLISH_BREAKOUT",
            "wick_rejection": "BULLISH_LOWER_WICK_REJECTION",
        },
        "day_levels": {
            "pdh": 23420.0,
            "pdl": 23250.0,
            "pdc": 23340.0,
            "cpr": {"pivot": 23336.7, "tc": 23380.0, "bc": 23293.4},
        },
        "order_blocks": {
            "smc_bias": "BULLISH",
            "nearest_bullish_ob": 23320.0,
            "nearest_bearish_ob": 23500.0,
        },
        "vix_intel": {
            "current_vix": 12.4,
            "regime": "IDEAL_VOLATILITY",
            "trend_bias": "VIX_FALLING_COOLING",
        },
        "gift_nifty_intel": {
            "gift_nifty_price": 23440.0,
            "projected_gap_pts": 35.0,
            "opening_bias": "GAP_UP_STRONG",
        },
    }

    record = audit_service.log_trade_alert(sample_signal)
    assert record["status"] == "ACTIVE"

    # Verify that trade_technical_factors.log was created and populated
    assert audit_service.technical_factors_file.exists()
    tf_logs = audit_service.get_recent_technical_factor_logs(limit=100)
    assert len(tf_logs) > 0

    log_content = "\n".join(tf_logs)
    assert "TRADE TECHNICAL FACTORS AUDIT REPORT" in log_content
    assert "NIFTY 50 (^NSEI)" in log_content
    assert "BUY CALL (CE) | Strike: 23400 CE" in log_content
    assert "21-FACTOR INSTITUTIONAL MATRIX" in log_content
    assert "Volume Surge Filter" in log_content
    assert "Choppiness Index" in log_content
    assert "CORE TECHNICAL INDICATORS" in log_content
    assert "RSI (14)" in log_content
    assert "DERIVATIVES & OPTIONS FLOW (PCR & MAX PAIN)" in log_content
    assert "PRICE ACTION & SMART MONEY CONCEPTS" in log_content
    assert "MACRO SENTIMENT & VOLATILITY CONTEXT" in log_content

