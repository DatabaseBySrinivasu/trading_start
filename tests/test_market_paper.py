def test_market_price(client):
    response = client.get("/api/v1/market/price/SBIN")
    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "SBIN"
    assert "price" in data
    assert data["currency"] == "INR"


def test_strategy_ema(client):
    response = client.get("/api/v1/market/strategy/ema/SBIN")
    assert response.status_code == 200
    data = response.json()
    assert "signal" in data
    assert data["signal"] in ["BUY", "SELL", "HOLD"]


def test_paper_portfolio(client):
    response = client.get("/api/v1/paper/portfolio")
    assert response.status_code == 200
    data = response.json()
    assert "available_cash" in data
    assert "total_portfolio_value" in data


def test_options_signals_endpoint(client):
    response = client.get("/api/v1/market/signals/options")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 3
    symbols = [item["symbol"] for item in data]
    assert "^NSEI" in symbols
    assert "^BSESN" in symbols
    assert "recommendation" in data[0]
    assert "suggested_strike" in data[0]
    assert "target_1" in data[0]
    assert "stop_loss" in data[0]
    assert "option_entry_price" in data[0]
    assert "lot_size" in data[0]


def test_single_options_signal_and_fibonacci(client):
    # Options signal for single stock
    response = client.get("/api/v1/market/signals/options/RELIANCE")
    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "RELIANCE"
    assert "recommendation" in data
    assert "fibonacci" in data
    assert "pcr" in data
    assert "pcr_oi" in data["pcr"]
    assert "price_action_momentum" in data

    # Fibonacci levels endpoint
    fib_res = client.get("/api/v1/market/fibonacci/RELIANCE")
    assert fib_res.status_code == 200
    fib_data = fib_res.json()
    assert "levels" in fib_data
    assert "fib_61.8" in fib_data["levels"]
    assert "fib_50.0" in fib_data["levels"]


def test_commodities_endpoint(client):
    response = client.get("/api/v1/market/commodities")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 5
    symbols = [item["symbol"] for item in data]
    assert "CRUDEOIL" in symbols
    assert "GOLD" in symbols
    assert "SILVER" in symbols
    assert "NATURALGAS" in symbols
    assert "COPPER" in symbols
    assert "option_entry_price" in data[0]
    assert "lot_size" in data[0]
    assert "unit" in data[0]


def test_momentum_endpoint(client):
    response = client.get("/api/v1/market/momentum/SBIN")
    assert response.status_code == 200
    data = response.json()
    assert "momentum_score" in data
    assert "momentum_regime" in data
    assert "trend_structure" in data
    assert "body_expansion_ratio" in data
    assert "bos_status" in data
    assert "wick_rejection" in data


def test_pcr_endpoints(client):
    # PCR for major indices
    res = client.get("/api/v1/market/pcr")
    assert res.status_code == 200
    pcr_list = res.json()
    assert isinstance(pcr_list, list)
    assert len(pcr_list) == 3
    assert "pcr_oi" in pcr_list[0]
    assert "max_pain_strike" in pcr_list[0]

    # PCR for single stock
    stock_res = client.get("/api/v1/market/pcr/SBIN")
    assert stock_res.status_code == 200
    stock_pcr = stock_res.json()
    assert stock_pcr["symbol"] == "SBIN"
    assert "pcr_oi" in stock_pcr
    assert "sentiment" in stock_pcr
    assert "total_put_oi" in stock_pcr
    assert "total_call_oi" in stock_pcr


def test_paper_buy_and_sell(client):
    # Place simulated buy
    buy_res = client.post("/api/v1/paper/buy", json={"symbol": "SBIN", "quantity": 2, "price": 800.0})
    assert buy_res.status_code == 200
    buy_data = buy_res.json()
    assert buy_data["success"] is True

    # Place simulated sell
    sell_res = client.post("/api/v1/paper/sell", json={"symbol": "SBIN", "quantity": 1, "price": 850.0})
    assert sell_res.status_code == 200
    sell_data = sell_res.json()
    assert sell_data["success"] is True


def test_news_endpoints(client):
    # Macro market news & sentiment endpoint
    res = client.get("/api/v1/market/news?limit=5")
    assert res.status_code == 200
    news_data = res.json()
    assert "overall_sentiment" in news_data
    assert "suggestion" in news_data
    assert "articles" in news_data
    assert isinstance(news_data["articles"], list)

    # Stock-specific news endpoint
    stock_news_res = client.get("/api/v1/market/news/RELIANCE?limit=3")
    assert stock_news_res.status_code == 200
    stock_news = stock_news_res.json()
    assert stock_news["symbol"] == "RELIANCE"
    assert "overall_sentiment" in stock_news
    assert "suggestion" in stock_news


def test_pattern_and_smc_endpoints(client):
    # Full patterns scan
    pat_res = client.get("/api/v1/market/patterns/SBIN")
    assert pat_res.status_code == 200
    p_data = pat_res.json()
    assert "candle_patterns" in p_data
    assert "day_levels" in p_data
    assert "volume_analysis" in p_data
    assert "order_blocks" in p_data
    assert "wm_patterns" in p_data

    # Day levels & CPR
    dl_res = client.get("/api/v1/market/daylevels/SBIN")
    assert dl_res.status_code == 200
    dl_data = dl_res.json()
    assert "pdh" in dl_data
    assert "pdl" in dl_data
    assert "cpr" in dl_data
    assert "tc" in dl_data["cpr"]
    assert "pivot" in dl_data["cpr"]
    assert "bc" in dl_data["cpr"]

    # SMC Order blocks
    ob_res = client.get("/api/v1/market/orderblocks/SBIN")
    assert ob_res.status_code == 200
    ob_data = ob_res.json()
    assert "smc_bias" in ob_data

    # W/M Patterns
    wm_res = client.get("/api/v1/market/wm/SBIN")
    assert wm_res.status_code == 200
    wm_data = wm_res.json()
    assert "pattern" in wm_data
    assert "pattern_type" in wm_data

    # Volume & VWAP
    vol_res = client.get("/api/v1/market/volume/SBIN")
    assert vol_res.status_code == 200
    vol_data = vol_res.json()
    assert "spike_status" in vol_data
    assert "volume_ratio" in vol_data
    assert "vwap" in vol_data


def test_vix_and_gift_nifty_endpoints(client):
    # India VIX endpoint
    vix_res = client.get("/api/v1/market/vix")
    assert vix_res.status_code == 200
    vix_data = vix_res.json()
    assert "current_vix" in vix_data
    assert "regime" in vix_data
    assert "regime_label" in vix_data
    assert "options_strategy" in vix_data

    # GIFT NIFTY endpoint
    gift_res = client.get("/api/v1/market/giftnifty")
    assert gift_res.status_code == 200
    gift_data = gift_res.json()
    assert "gift_nifty_price" in gift_data
    assert "nifty_spot_price" in gift_data
    assert "projected_gap_pts" in gift_data
    assert "opening_bias" in gift_data


def test_angel_status_and_connect_endpoints(client):
    # Angel status check
    status_res = client.get("/api/v1/market/angel/status")
    assert status_res.status_code == 200
    status_data = status_res.json()
    assert "is_configured" in status_data
    assert "is_connected" in status_data
    assert "supported_instruments" in status_data
    assert len(status_data["supported_instruments"]) > 0


def test_telegram_alerting_endpoints(client):
    # 1. Telegram Status Endpoint
    status_res = client.get("/api/v1/telegram/status")
    assert status_res.status_code == 200
    status_data = status_res.json()
    assert "configured" in status_data
    assert status_data["target_chat_id"] in ["9100040008", "6817447645"]
    assert "guide" in status_data

    # 2. Telegram Send Signal Endpoint
    sig_res = client.post("/api/v1/telegram/send-signal/^NSEI")
    assert sig_res.status_code == 200
    sig_data = sig_res.json()
    assert "delivery_result" in sig_data
    assert "signal_sent" in sig_data
    assert sig_data["signal_sent"]["symbol"] == "^NSEI"
    assert "suggested_strike" in sig_data["signal_sent"]
    assert "option_entry_price" in sig_data["signal_sent"]

    # 3. Telegram Test Ping Endpoint
    test_res = client.post("/api/v1/telegram/test")
    assert test_res.status_code == 200
    test_data = test_res.json()
    assert "status" in test_data
    assert test_data["chat_id"] in ["9100040008", "6817447645"]

    # 4. Telegram Broadcast Endpoint
    bcast_res = client.post("/api/v1/telegram/broadcast-all?include_commodities=false")
    assert bcast_res.status_code == 200
    bcast_data = bcast_res.json()
    assert "total_scanned" in bcast_data
    assert bcast_data["total_scanned"] == 3
    assert "dispatched" in bcast_data

    # 5. Telegram Dynamic Config Endpoint
    cfg_res = client.post("/api/v1/telegram/config", json={"bot_token": "TEST_TOKEN_123", "chat_id": "9100040008"})
    assert cfg_res.status_code == 200
    cfg_data = cfg_res.json()
    assert cfg_data["status"] == "success"


def test_strike_price_accuracy_and_alias_resolution(client):
    """Verify alias normalization and strike price accuracy for indices, stocks, and commodities."""
    # 1. NIFTY alias
    res_nifty = client.get("/api/v1/market/signals/options/NIFTY")
    assert res_nifty.status_code == 200
    d_nifty = res_nifty.json()
    assert d_nifty["spot_price"] > 20000.0
    assert d_nifty["strike_price"] > 20000
    assert d_nifty["lot_size"] == 65
    assert d_nifty["suggested_strike"].endswith("CE") or d_nifty["suggested_strike"].endswith("PE")

    # 2. BANK NIFTY alias
    res_bank = client.get("/api/v1/market/signals/options/BANKNIFTY")
    assert res_bank.status_code == 200
    d_bank = res_bank.json()
    assert d_bank["spot_price"] > 45000.0
    assert d_bank["strike_price"] > 45000
    assert d_bank["lot_size"] == 30

    # 3. SENSEX alias
    res_sensex = client.get("/api/v1/market/signals/options/SENSEX")
    assert res_sensex.status_code == 200
    d_sensex = res_sensex.json()
    assert d_sensex["spot_price"] > 65000.0
    assert d_sensex["strike_price"] > 65000
    assert d_sensex["lot_size"] == 20

    # 4. Commodities (GOLD & CRUDEOIL)
    res_gold = client.get("/api/v1/market/signals/options/GOLD")
    assert res_gold.status_code == 200
    d_gold = res_gold.json()
    assert d_gold["spot_price"] > 60000.0
    assert d_gold["strike_price"] > 60000
    assert d_gold["lot_size"] == 100

    res_crude = client.get("/api/v1/market/signals/options/CRUDEOIL")
    assert res_crude.status_code == 200
    d_crude = res_crude.json()
    assert d_crude["spot_price"] > 4000.0
    assert d_crude["lot_size"] == 100


def test_index_level_paper_trading_execution(client):
    """Test full multi-asset Index-level paper trading with Long/Short, lot multipliers and margin."""
    # 1. Reset portfolio first with ₹5,00,000 capital
    reset_res = client.post("/api/v1/paper/reset", json={"initial_balance": 500000.0})
    assert reset_res.status_code == 200
    r_data = reset_res.json()
    assert r_data["success"] is True
    assert r_data["available_cash"] == 500000.0

    # 2. Place LONG order on NIFTY 50 (1 lot = 65 qty)
    long_res = client.post("/api/v1/paper/order", json={
        "symbol": "^NSEI",
        "side": "LONG",
        "asset_type": "INDEX",
        "lots": 1,
        "price": 24500.0,
        "target_1": 24550.0,
        "target_2": 24600.0,
        "stop_loss": 24450.0,
        "notes": "Test NIFTY Long Level"
    })
    assert long_res.status_code == 200
    long_data = long_res.json()
    assert long_data["success"] is True
    assert long_data["position"]["symbol"] == "^NSEI"
    assert long_data["position"]["quantity"] == 65
    assert long_data["position"]["side"] == "LONG"
    assert long_data["position"]["margin_used"] == (65 * 24500.0) * 0.10

    # 3. Place SHORT order on SENSEX (1 lot = 20 qty)
    short_res = client.post("/api/v1/paper/order", json={
        "symbol": "^BSESN",
        "side": "SHORT",
        "asset_type": "INDEX",
        "lots": 1,
        "price": 80000.0,
        "target_1": 79800.0,
        "target_2": 79600.0,
        "stop_loss": 80150.0,
        "notes": "Test SENSEX Short Level"
    })
    assert short_res.status_code == 200
    short_data = short_res.json()
    assert short_data["success"] is True
    assert short_data["position"]["quantity"] == 20
    assert short_data["position"]["side"] == "SHORT"

    # 4. Check portfolio summary with open positions
    port_res = client.get("/api/v1/paper/portfolio")
    assert port_res.status_code == 200
    port = port_res.json()
    assert len(port["positions"]) == 2
    assert port["margin_used"] > 0
    assert port["available_cash"] < 500000.0

    # 5. Reverse NIFTY position (Long -> Short)
    rev_res = client.post("/api/v1/paper/reverse", json={"symbol": "^NSEI"})
    assert rev_res.status_code == 200
    rev_data = rev_res.json()
    assert rev_data["success"] is True
    assert rev_data["new_position"]["side"] == "SHORT"

    # 6. Close SENSEX position
    close_res = client.post("/api/v1/paper/close", json={"symbol": "^BSESN", "price": 79900.0})
    assert close_res.status_code == 200
    close_data = close_res.json()
    assert close_data["success"] is True
    assert close_data["realized_pnl"] == (80000.0 - 79900.0) * 20  # +2000 INR on short profit

    # 7. Check lot sizes endpoint
    lots_res = client.get("/api/v1/paper/lot-sizes")
    assert lots_res.status_code == 200
    lots_data = lots_res.json()
    assert lots_data["indices"]["NIFTY 50"] == 65
    assert lots_data["indices"]["SENSEX"] == 20
    assert lots_data["indices"]["BANK NIFTY"] == 30
    assert lots_data["commodities"]["CRUDE OIL (Brent)"]["lot_size"] == 100
    assert lots_data["commodities"]["NATURAL GAS (HH)"]["lot_size"] == 1250



