import os
import json
import logging
import pytest
from pathlib import Path
from app.services.market_data import market_data_service
from app.services.strategy_engine import strategy_engine
from app.services.pcr_service import pcr_service
from app.services.paper_trader import PaperTrader

logger = logging.getLogger("system_audit")


# -------------------------------------------------------------
# 1. CORE SYSTEM & DASHBOARD HEALTH
# -------------------------------------------------------------
def test_dashboard_and_health(client):
    """Verify frontend dashboard HTML, health check, and OpenAPI schema."""
    # Health check
    res_health = client.get("/health")
    assert res_health.status_code == 200
    data_health = res_health.json()
    assert data_health["status"] == "healthy"
    assert data_health["version"] == "0.1.0"

    # Dashboard HTML
    res_dash = client.get("/")
    assert res_dash.status_code == 200
    assert "<!DOCTYPE html>" in res_dash.text
    assert "QuantPulse India" in res_dash.text
    assert "PRO" in res_dash.text

    # OpenAPI JSON
    res_open = client.get("/api/v1/openapi.json")
    assert res_open.status_code == 200
    openapi_doc = res_open.json()
    assert "paths" in openapi_doc
    assert "/api/v1/market/indices" in openapi_doc["paths"]
    assert "/api/v1/paper/portfolio" in openapi_doc["paths"]


# -------------------------------------------------------------
# 2. MARKET DATA & INDICES VERIFICATION
# -------------------------------------------------------------
def test_market_indices_and_watchlist(client):
    """Verify live quotes for indices and bluechip stocks."""
    # Indices
    res_ind = client.get("/api/v1/market/indices")
    assert res_ind.status_code == 200
    indices = res_ind.json()
    assert len(indices) == 3
    symbols = [idx["symbol"] for idx in indices]
    assert "^NSEI" in symbols
    assert "^BSESN" in symbols
    assert "^NSEBANK" in symbols
    for item in indices:
        assert item["price"] > 0
        assert "change" in item
        assert "change_percent" in item

    # Watchlist
    res_watch = client.get("/api/v1/market/watchlist")
    assert res_watch.status_code == 200
    watchlist = res_watch.json()
    assert len(watchlist) == 8
    watch_symbols = [s["symbol"] for s in watchlist]
    assert "RELIANCE" in watch_symbols
    assert "TCS" in watch_symbols
    assert "SBIN" in watch_symbols
    assert "HDFCBANK" in watch_symbols


# -------------------------------------------------------------
# 3. MCX COMMODITIES DATA & SIGNALS
# -------------------------------------------------------------
def test_mcx_commodities_data(client):
    """Verify MCX commodities prices, units, and options signals."""
    res = client.get("/api/v1/market/commodities")
    assert res.status_code == 200
    commodities = res.json()
    assert len(commodities) == 5

    comm_map = {item["symbol"]: item for item in commodities}
    assert "CRUDEOIL" in comm_map
    assert "GOLD" in comm_map
    assert "SILVER" in comm_map
    assert "NATURALGAS" in comm_map
    assert "COPPER" in comm_map

    # Unit checks
    assert comm_map["CRUDEOIL"]["unit"] == "₹/bbl"
    assert comm_map["GOLD"]["unit"] == "₹/10g"
    assert comm_map["SILVER"]["unit"] == "₹/kg"
    assert comm_map["NATURALGAS"]["unit"] == "₹/mmBtu"
    assert comm_map["COPPER"]["unit"] == "₹/kg"

    # Strike and lot size sanity
    assert comm_map["GOLD"]["spot_price"] > 50000.0
    assert comm_map["GOLD"]["lot_size"] == 100
    assert comm_map["CRUDEOIL"]["spot_price"] > 3000.0
    assert comm_map["CRUDEOIL"]["lot_size"] == 100


# -------------------------------------------------------------
# 4. OPTIONS SIGNALS & GREEKS MATHEMATICAL INTEGRITY
# -------------------------------------------------------------
def test_options_signals_mathematical_bounds(client):
    """Verify Black-Scholes strike calculations, Delta, targets, and stop loss bounds."""
    # NIFTY 50
    res = client.get("/api/v1/market/signals/options/^NSEI")
    assert res.status_code == 200
    sig = res.json()

    spot = sig["spot_price"]
    strike = sig["strike_price"]
    entry = sig["option_entry_price"]
    t1 = sig["option_target_1"]
    t2 = sig["option_target_2"]
    sl = sig["option_stop_loss"]

    assert spot > 15000.0
    # Strike must be divisible by 50 for NIFTY
    assert strike % 50 == 0
    assert sig["lot_size"] == 65
    assert entry > 0

    # Risk/Reward integrity
    assert t1 > entry, f"Target 1 ({t1}) must be greater than Entry ({entry})"
    assert t2 > t1, f"Target 2 ({t2}) must be greater than Target 1 ({t1})"
    assert sl < entry, f"Stop Loss ({sl}) must be less than Entry ({entry})"

    # Target 1 should represent a realistic gain (+25% to +45%)
    t1_gain_pct = (t1 - entry) / entry * 100
    assert 20.0 <= t1_gain_pct <= 50.0, f"Target 1 gain ({t1_gain_pct:.1f}%) outside expected 20-50% range"

    # Stop Loss should represent a bounded risk (-15% to -35%)
    sl_loss_pct = (entry - sl) / entry * 100
    assert 15.0 <= sl_loss_pct <= 35.0, f"Stop Loss drawdown ({sl_loss_pct:.1f}%) outside expected 15-35% range"

    # Delta Greek check (ATM should be ~0.45 - 0.60)
    delta = sig.get("delta", 0.50)
    assert 0.30 <= delta <= 0.75, f"ATM Delta ({delta}) outside reasonable bound"


# -------------------------------------------------------------
# 5. TICKER ALIASING & STRIKE STEP ACCURACY
# -------------------------------------------------------------
def test_all_symbol_aliases_and_strike_steps():
    """Verify that normalize_symbol and strike step resolution correctly handles all variations."""
    test_cases = [
        ("NIFTY", "^NSEI", 50, 65),
        ("NIFTY 50", "^NSEI", 50, 65),
        ("NIFTY50", "^NSEI", 50, 65),
        ("BANKNIFTY", "^NSEBANK", 100, 30),
        ("BANK NIFTY", "^NSEBANK", 100, 30),
        ("SENSEX", "^BSESN", 100, 20),
        ("BSE SENSEX", "^BSESN", 100, 20),
        ("FINNIFTY", "NIFTY_FIN_SERVICE.NS", 50, 65),
        ("MIDCPNIFTY", "^NSEMDCP50", 25, 120),
        ("CRUDEOIL", "CRUDEOIL", 50, 100),
        ("BRENT", "CRUDEOIL", 50, 100),
        ("GOLD", "GOLD", 100, 100),
        ("SILVER", "SILVER", 500, 30),
        ("NATURALGAS", "NATURALGAS", 5, 1250),
        ("HENRYHUB", "NATURALGAS", 5, 1250),
        ("COPPER", "COPPER", 5, 2500),
    ]

    for alias, expected_canonical, expected_step, expected_lot in test_cases:
        canonical = market_data_service.normalize_symbol(alias)
        assert canonical == expected_canonical, f"Alias '{alias}' resolved to '{canonical}', expected '{expected_canonical}'"

        step = pcr_service._get_strike_step(alias, spot_price=50000.0)
        assert step == expected_step, f"Step for '{alias}' is {step}, expected {expected_step}"


# -------------------------------------------------------------
# 6. TECHNICAL INDICATORS & SMC PATTERNS
# -------------------------------------------------------------
def test_technical_analysis_and_smc(client):
    """Verify indicators, Fibonacci Golden Pocket, SMC Order Blocks, and Day Levels."""
    # Fibonacci
    fib_res = client.get("/api/v1/market/fibonacci/SBIN")
    assert fib_res.status_code == 200
    fib_data = fib_res.json()
    assert "levels" in fib_data
    assert "fib_61.8" in fib_data["levels"]
    assert "fib_50.0" in fib_data["levels"]
    assert "fib_38.2" in fib_data["levels"]

    # Day Levels & CPR
    dl_res = client.get("/api/v1/market/daylevels/SBIN")
    assert dl_res.status_code == 200
    dl_data = dl_res.json()
    assert "pdh" in dl_data
    assert "pdl" in dl_data
    assert "cpr" in dl_data
    assert "pivot" in dl_data["cpr"]

    # SMC Order Blocks
    ob_res = client.get("/api/v1/market/orderblocks/SBIN")
    assert ob_res.status_code == 200
    ob_data = ob_res.json()
    assert "smc_bias" in ob_data
    assert "nearest_bullish_ob" in ob_data
    assert "nearest_bearish_ob" in ob_data

    # Momentum Diagnostics
    mom_res = client.get("/api/v1/market/momentum/SBIN")
    assert mom_res.status_code == 200
    mom_data = mom_res.json()
    assert "momentum_score" in mom_data
    assert "momentum_regime" in mom_data
    assert "trend_structure" in mom_data


# -------------------------------------------------------------
# 7. MACRO SENTIMENT, VIX, AND GIFT NIFTY
# -------------------------------------------------------------
def test_macro_vix_gift_nifty_and_news(client):
    """Verify India VIX regime, GIFT NIFTY opening cues, and news sentiment."""
    # VIX
    vix_res = client.get("/api/v1/market/vix")
    assert vix_res.status_code == 200
    vix_data = vix_res.json()
    assert "current_vix" in vix_data
    assert "regime" in vix_data
    assert vix_data["current_vix"] > 0

    # GIFT NIFTY
    gift_res = client.get("/api/v1/market/giftnifty")
    assert gift_res.status_code == 200
    gift_data = gift_res.json()
    assert "gift_nifty_price" in gift_data
    assert "projected_gap_pts" in gift_data
    assert "opening_bias" in gift_data

    # News
    news_res = client.get("/api/v1/market/news?limit=5")
    assert news_res.status_code == 200
    news_data = news_res.json()
    assert "overall_sentiment" in news_data
    assert "articles" in news_data
    assert len(news_data["articles"]) > 0


# -------------------------------------------------------------
# 8. PAPER TRADING FULL LIFECYCLE & INPUT VALIDATION
# -------------------------------------------------------------
def test_paper_trading_lifecycle_and_validation(client):
    """Verify complete paper trading lifecycle and robust edge case validations."""
    # 1. Invalid input validation (quantity <= 0)
    invalid_buy = client.post("/api/v1/paper/buy", json={"symbol": "SBIN", "quantity": 0, "price": 800.0})
    assert invalid_buy.status_code == 422  # Pydantic validation error

    invalid_neg = client.post("/api/v1/paper/buy", json={"symbol": "SBIN", "quantity": -5, "price": 800.0})
    assert invalid_neg.status_code == 422

    # 2. Buy with excessive funds (insufficient virtual cash)
    huge_buy = client.post("/api/v1/paper/buy", json={"symbol": "SBIN", "quantity": 10000000, "price": 800.0})
    assert huge_buy.status_code == 400
    assert "Insufficient funds" in huge_buy.json()["detail"]

    # 3. Sell shares not owned
    unheld_sell = client.post("/api/v1/paper/sell", json={"symbol": "UNHELD_STOCK_XYZ", "quantity": 10, "price": 100.0})
    assert unheld_sell.status_code == 400
    assert "Insufficient shares" in unheld_sell.json()["detail"]

    # 4. Successful Buy -> Hold -> Sell Lifecycle with isolated trader instance
    trader = PaperTrader(initial_balance=50000.0)
    # Buy 10 SBIN @ 800 (Cost = 8,000)
    buy1 = trader.buy("SBIN", quantity=10, price=800.0)
    assert buy1["success"] is True
    assert trader.cash == 42000.0
    assert trader.positions["SBIN"].quantity == 10
    assert trader.positions["SBIN"].average_price == 800.0

    # Buy 10 more SBIN @ 820 (Avg Price = 810)
    buy2 = trader.buy("SBIN", quantity=10, price=820.0)
    assert buy2["success"] is True
    assert trader.cash == 33800.0
    assert trader.positions["SBIN"].quantity == 20
    assert trader.positions["SBIN"].average_price == 810.0

    # Sell 10 SBIN @ 850 (Revenue = 8,500; Realized PnL = 8,500 - 8,100 = +400)
    sell1 = trader.sell("SBIN", quantity=10, price=850.0)
    assert sell1["success"] is True
    assert trader.cash == 42300.0
    assert trader.realized_pnl == 400.0
    assert trader.positions["SBIN"].quantity == 10

    # Sell remaining 10 SBIN @ 860 (Revenue = 8,600; Realized PnL = 400 + 500 = +900)
    sell2 = trader.sell("SBIN", quantity=10, price=860.0)
    assert sell2["success"] is True
    assert trader.cash == 50900.0
    assert trader.realized_pnl == 900.0
    assert "SBIN" not in trader.positions

    summary = trader.get_portfolio_summary()
    assert summary["available_cash"] == 50900.0
    assert summary["realized_pnl"] == 900.0
    assert summary["total_pnl"] == 900.0
    assert summary["open_positions_count"] == 0


# -------------------------------------------------------------
# 9. TELEGRAM ALERTING & ANGEL ONE INTEGRATIONS
# -------------------------------------------------------------
def test_telegram_and_angel_endpoints(client):
    """Verify Telegram webhook alerting and Angel One status."""
    # Angel Status
    angel_res = client.get("/api/v1/market/angel/status")
    assert angel_res.status_code == 200
    angel_data = angel_res.json()
    assert "is_configured" in angel_data
    assert "is_connected" in angel_data
    assert "supported_instruments" in angel_data

    # Telegram Status
    tg_res = client.get("/api/v1/telegram/status")
    assert tg_res.status_code == 200
    tg_data = tg_res.json()
    assert "target_chat_id" in tg_data
    assert tg_data["target_chat_id"] in ["9100040008", "6817447645"]

    # Telegram Broadcast
    bcast = client.post("/api/v1/telegram/broadcast-all?include_commodities=true")
    assert bcast.status_code == 200
    bcast_data = bcast.json()
    assert bcast_data["total_scanned"] == 8  # 3 indices + 5 commodities
    assert "dispatched" in bcast_data

    # Instagram Status & Alerting
    ig_res = client.get("/api/v1/instagram/status")
    assert ig_res.status_code == 200
    ig_data = ig_res.json()
    assert "target_recipient" in ig_data
    assert ig_data["target_recipient"] == "9100040008"

    ig_test = client.post("/api/v1/instagram/test?recipient_id=9100040008")
    assert ig_test.status_code == 200
    assert ig_test.json()["success"] is True

    ig_bcast = client.post("/api/v1/instagram/broadcast-all?recipient_id=9100040008")
    assert ig_bcast.status_code == 200
    assert ig_bcast.json()["total_scanned"] == 8


# -------------------------------------------------------------
# 10. REAL-TIME 1-SECOND CONSOLIDATED LIVE TICK ENDPOINT
# -------------------------------------------------------------
def test_live_tick_consolidated_snapshot(client):
    """Verify high-frequency consolidated 1-second live tick payload."""
    res = client.get("/api/v1/market/live-tick")
    assert res.status_code == 200
    data = res.json()

    assert "timestamp" in data
    assert "iso_time" in data
    assert "angel_status" in data
    assert "indices" in data
    assert len(data["indices"]) == 3
    assert "watchlist" in data
    assert len(data["watchlist"]) == 8
    assert "options_signals" in data
    assert len(data["options_signals"]) == 3
    assert "commodities" in data
    assert len(data["commodities"]) == 5
    assert "vix" in data
    assert "gift_nifty" in data

    # Verify indices structure
    nifty = next((x for x in data["indices"] if x["symbol"] == "^NSEI"), None)
    assert nifty is not None
    assert nifty["price"] > 15000.0
    assert "tick_direction" in nifty

    # Verify options signal in snapshot
    nifty_sig = next((x for x in data["options_signals"] if x["symbol"] == "^NSEI"), None)
    assert nifty_sig is not None
    assert nifty_sig["option_entry_price"] > 0
    assert nifty_sig["option_target_1"] > nifty_sig["option_entry_price"]
    assert nifty_sig["option_stop_loss"] < nifty_sig["option_entry_price"]
