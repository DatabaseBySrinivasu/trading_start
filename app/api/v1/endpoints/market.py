from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List, Dict, Any
from app.services.market_data import market_data_service
from app.services.strategy_engine import strategy_engine
from app.services.pcr_service import pcr_service
from app.services.news_service import news_service
from app.services.pattern_service import pattern_service
from app.services.volatility_service import volatility_service
from app.services.angel_service import angel_client
from app.services.local_data_service import local_data_service
from app.services.institutional_order_flow_service import institutional_order_flow_service
from app.services.unified_strategy_service import unified_strategy_service
from app.services.instagram_service import instagram_service

router = APIRouter()

POPULAR_INDICES = [
    {"name": "NIFTY 50", "symbol": "^NSEI"},
    {"name": "BSE SENSEX", "symbol": "^BSESN"},
    {"name": "BANK NIFTY", "symbol": "^NSEBANK"},
]

POPULAR_COMMODITIES = [
    {"name": "Brent Crude Oil (MCX)", "symbol": "CRUDEOIL", "unit": "₹/bbl"},
    {"name": "Gold 999 (MCX)", "symbol": "GOLD", "unit": "₹/10g"},
    {"name": "Silver (MCX)", "symbol": "SILVER", "unit": "₹/kg"},
    {"name": "Henry Hub Natural Gas (MCX)", "symbol": "NATURALGAS", "unit": "₹/mmBtu"},
    {"name": "Copper (MCX)", "symbol": "COPPER", "unit": "₹/kg"},
]

POPULAR_STOCKS = [
    {"name": "Reliance Industries", "symbol": "RELIANCE"},
    {"name": "Tata Consultancy Services", "symbol": "TCS"},
    {"name": "HDFC Bank", "symbol": "HDFCBANK"},
    {"name": "State Bank of India", "symbol": "SBIN"},
    {"name": "Infosys", "symbol": "INFY"},
    {"name": "ICICI Bank", "symbol": "ICICIBANK"},
    {"name": "Larsen & Toubro", "symbol": "LT"},
    {"name": "ITC", "symbol": "ITC"},
]


@router.get("/indices", summary="Get top NSE & BSE indices")
def get_indices():
    """Fetch live data for NIFTY 50, BSE SENSEX, and BANK NIFTY."""
    results = []
    for item in POPULAR_INDICES:
        data = market_data_service.get_live_price(item["symbol"])
        if data:
            results.append({
                "name": item["name"],
                "symbol": item["symbol"],
                "price": data["price"],
                "change": data["change"],
                "change_percent": data["change_percent"],
            })
    return results


@router.get("/commodities", summary="Get MCX / Global Commodities live quotes and signals")
def get_commodities(
    interval: str = Query("15m", description="Candle interval (5m, 15m, 1h, 1d)"),
    period: str = Query("5d", description="Lookback period (5d, 1mo)"),
):
    """
    Fetch real-time MCX Commodity prices, lot sizes, and multi-strategy CALL / PUT signals
    for Crude Oil, Gold, Silver, Natural Gas, and Copper.
    """
    results = []
    for item in POPULAR_COMMODITIES:
        sig = strategy_engine.generate_options_call_put_signal(item["symbol"], interval=interval, period=period)
        live_info = market_data_service.get_live_price(item["symbol"])
        sig["unit"] = item.get("unit", "₹")
        sig["raw_usd_price"] = live_info.get("raw_usd_price") if live_info else None
        results.append(sig)
    return results


@router.get("/watchlist", summary="Get top Indian stocks watchlist")
def get_watchlist():
    """Fetch live prices for popular Indian stocks."""
    results = []
    for stock in POPULAR_STOCKS:
        data = market_data_service.get_live_price(stock["symbol"])
        if data:
            results.append({
                "name": stock["name"],
                "symbol": stock["symbol"],
                "price": data["price"],
                "change": data["change"],
                "change_percent": data["change_percent"],
            })
    return results


@router.get("/price/{symbol}", summary="Get live stock or commodity quote")
def get_stock_price(symbol: str):
    """Retrieve real-time price and day change for any stock, index, or commodity."""
    data = market_data_service.get_live_price(symbol)
    if not data:
        raise HTTPException(status_code=404, detail=f"Data not found for symbol: {symbol}")
    return data


@router.get("/strategy/ema/{symbol}", summary="EMA 9/21 Crossover Strategy Analysis")
def analyze_ema(
    symbol: str,
    interval: str = Query("15m", description="Candle interval (e.g. 5m, 15m, 1h, 1d)"),
    period: str = Query("5d", description="Lookback period (e.g. 5d, 1mo, 3mo)"),
):
    """Get BUY/SELL/HOLD signal based on EMA 9 and 21 crossovers."""
    return strategy_engine.analyze_ema_crossover(symbol, interval=interval, period=period)


@router.get("/signals/options", summary="Get CALL / PUT recommendation for NIFTY 50, SENSEX, & BANK NIFTY")
def get_options_signals(
    interval: str = Query("15m", description="Candle interval (5m, 15m, 1h, 1d)"),
    period: str = Query("5d", description="Lookback period (5d, 1mo)"),
):
    """
    Evaluates multi-strategy confluence (Fibonacci, Supertrend, EMA 9/21/50, RSI, MACD, Bollinger Bands,
    PCR, SMC Order Blocks, Price Action Momentum, India VIX, and GIFT NIFTY)
    and returns actionable CALL (CE) / PUT (PE) recommendations, strikes, targets, and stop-loss levels.
    """
    indices = ["^NSEI", "^BSESN", "^NSEBANK"]
    signals = []
    for sym in indices:
        sig = strategy_engine.generate_options_call_put_signal(sym, interval=interval, period=period)
        signals.append(sig)
    return signals


@router.get("/signals/options/{symbol}", summary="Get CALL / PUT recommendation for any symbol")
def get_single_options_signal(
    symbol: str,
    interval: str = Query("15m", description="Candle interval (5m, 15m, 1h, 1d)"),
    period: str = Query("5d", description="Lookback period (5d, 1mo)"),
):
    """Returns CALL / PUT recommendation and indicators for a single stock, index, or commodity."""
    return strategy_engine.generate_options_call_put_signal(symbol, interval=interval, period=period)


@router.get("/momentum/{symbol}", summary="Get Price Action Momentum & Impulse Diagnostics")
def get_price_action_momentum(
    symbol: str,
    period: str = Query("1mo", description="Historical lookback period"),
    interval: str = Query("15m", description="Candle interval (5m, 15m, 1h, 1d)"),
):
    """
    Evaluates Price Action Momentum, Impulse Velocity, Candle Body Expansion,
    Higher High/Higher Low vs Lower High/Lower Low structure, Break of Structure (BOS),
    and Rejection Wicks.
    """
    df = market_data_service.get_historical_candles(symbol, period=period, interval=interval)
    if df is None:
        raise HTTPException(status_code=404, detail=f"Candle data not found for symbol: {symbol}")
    return pattern_service.analyze_price_action_momentum(df)


@router.get("/fibonacci/{symbol}", summary="Get Fibonacci Retracement Levels")
def get_fibonacci_levels(
    symbol: str,
    period: str = Query("1mo", description="Historical period to calculate swing high/low"),
    interval: str = Query("1d", description="Candle interval"),
):
    """Calculates 0.0%, 23.6%, 38.2%, 50.0%, 61.8% Golden Pocket, 78.6%, 100.0% Fibonacci levels."""
    df = market_data_service.get_historical_candles(symbol, period=period, interval=interval)
    if df is None:
        raise HTTPException(status_code=404, detail=f"No historical candle data for {symbol}")
    return strategy_engine.calculate_fibonacci(df)


@router.get("/pcr", summary="Get Put-Call Ratio (PCR) and OI analysis for top indices")
def get_indices_pcr():
    """Returns Put-Call Ratio (PCR), Open Interest (OI) distribution, and Max Pain for NIFTY 50, SENSEX, & BANK NIFTY."""
    indices = ["^NSEI", "^BSESN", "^NSEBANK"]
    results = []
    for sym in indices:
        data = market_data_service.get_live_price(sym)
        if data:
            pcr_res = pcr_service.analyze_pcr(sym, spot_price=data["price"])
            results.append(pcr_res)
    return results


@router.get("/pcr/{symbol}", summary="Get Put-Call Ratio (PCR) and Open Interest for any symbol")
def get_single_pcr(symbol: str):
    """Calculates live PCR (OI), Max Pain Strike, and Call/Put Open Interest distribution for any stock or index."""
    data = market_data_service.get_live_price(symbol)
    if not data:
        raise HTTPException(status_code=404, detail=f"Price data not found for symbol: {symbol}")
    return pcr_service.analyze_pcr(symbol, spot_price=data["price"])


@router.get("/news", summary="Get latest Indian stock market news, sentiment, and AI trading suggestions")
def get_market_news(
    limit: int = Query(15, description="Number of news articles to retrieve", ge=1, le=50),
    refresh: bool = Query(False, description="Force refresh news feed bypassing cache"),
):
    """
    Fetches real-time financial market news from Google News RSS & Yahoo Finance,
    computes natural language sentiment scores, and generates actionable trading suggestions.
    """
    return news_service.get_market_news(force_refresh=refresh, limit=limit)


@router.get("/news/{symbol}", summary="Get company-specific news and catalyst sentiment for any stock")
def get_stock_news(
    symbol: str,
    limit: int = Query(10, description="Number of news articles to retrieve", ge=1, le=30),
    refresh: bool = Query(False, description="Force refresh news feed bypassing cache"),
):
    """
    Retrieves latest news and sentiment for a specific ticker (e.g. SBIN, RELIANCE, TCS, INFY).
    """
    return news_service.get_stock_news(symbol, limit=limit, force_refresh=refresh)


@router.get("/patterns/{symbol}", summary="Full Pattern Scanner (Candlestick, Day Levels, Volume, SMC Order Blocks, W/M, Momentum)")
def get_all_patterns(
    symbol: str,
    period: str = Query("1mo", description="Historical lookback period"),
    interval: str = Query("15m", description="Candle interval (5m, 15m, 1h, 1d)"),
):
    """
    Scans any index or stock for Candlestick patterns, Previous Day High/Low & CPR,
    Volume Spikes & VWAP, Institutional Order Blocks (SMC), W/M Chart Patterns, and Price Action Momentum.
    """
    df = market_data_service.get_historical_candles(symbol, period=period, interval=interval)
    if df is None:
        raise HTTPException(status_code=404, detail=f"Candle data not found for symbol: {symbol}")
    return pattern_service.run_full_pattern_scan(df)


@router.get("/daylevels/{symbol}", summary="Get Day High, Day Low, and Central Pivot Range (CPR)")
def get_day_levels(
    symbol: str,
    period: str = Query("1mo", description="Historical period"),
    interval: str = Query("1d", description="Interval for daily levels"),
):
    """Calculates PDH, PDL, PDC, CDH, CDL, Central Pivot Range (CPR), R1-R4, S1-S4, and Camarilla levels."""
    df = market_data_service.get_historical_candles(symbol, period=period, interval=interval)
    if df is None:
        raise HTTPException(status_code=404, detail=f"Candle data not found for symbol: {symbol}")
    return pattern_service.calculate_day_levels(df)


@router.get("/support-resistance/{symbol}", summary="Unified Multi-Timeframe Support & Resistance Hierarchy")
def get_multi_timeframe_support_resistance(symbol: str):
    """
    Computes comprehensive Multi-Timeframe Historical Support & Resistance:
    - Multi-Day Extremes (PDH/PDL, 2DH-20DH, PWH/PWL, PMH/PML)
    - Daily & Weekly Central Pivot Range (CPR)
    - Daily & Weekly Classical Pivots (R1-R4, S1-S4)
    - Daily & Weekly Camarilla Pivots (H1-H5, L1-L5)
    - Supply & Demand Fractal Swing Clusters
    - Macro Fibonacci Retracements & Golden Pocket
    - Consolidated Resistance (R_near, R_mid, R_major) & Support (S_near, S_mid, S_major) Hierarchy
    """
    df_intra = market_data_service.get_historical_candles(symbol, period="5d", interval="15m")
    df_daily = market_data_service.get_historical_candles(symbol, period="3mo", interval="1d")
    df_weekly = market_data_service.get_historical_candles(symbol, period="6mo", interval="1wk")
    return pattern_service.calculate_multi_timeframe_sr(
        df_intraday=df_intra,
        df_daily=df_daily,
        df_weekly=df_weekly,
        symbol=symbol,
    )


@router.get("/camarilla/{symbol}", summary="Get Daily & Weekly Camarilla Pivot System")
def get_camarilla_pivots(
    symbol: str,
    period: str = Query("1mo", description="Historical period"),
    interval: str = Query("1d", description="Candle interval"),
):
    """Calculates institutional Camarilla Pivot Levels (H1-H5, L1-L5) with breakout/breakdown status."""
    df = market_data_service.get_historical_candles(symbol, period=period, interval=interval)
    if df is None or len(df) < 2:
        raise HTTPException(status_code=404, detail=f"Insufficient candle history for symbol: {symbol}")
    prev_day = df.iloc[-2] if len(df) >= 2 else df.iloc[-1]
    cur_p = float(df["close"].iloc[-1])
    return pattern_service.calculate_camarilla_pivots(
        high=float(prev_day["high"]),
        low=float(prev_day["low"]),
        close=float(prev_day["close"]),
        cur_price=cur_p,
    )


@router.get("/macro-fibonacci/{symbol}", summary="Get Multi-Day Macro Fibonacci Retracements")
def get_macro_fibonacci(
    symbol: str,
    period: str = Query("3mo", description="Historical period"),
    interval: str = Query("1d", description="Candle interval"),
):
    """Calculates Multi-Day Macro Fibonacci Retracements (23.6%, 38.2%, 50%, 61.8% Golden Pocket, 78.6%, 161.8%)."""
    df = market_data_service.get_historical_candles(symbol, period=period, interval=interval)
    if df is None:
        raise HTTPException(status_code=404, detail=f"Candle data not found for symbol: {symbol}")
    return pattern_service.calculate_macro_fibonacci(df, lookback=45)


@router.get("/orderblocks/{symbol}", summary="Get Smart Money Concepts (SMC) Order Blocks & Fair Value Gaps")
@router.get("/order-blocks/{symbol}", summary="Get Institutional Order Blocks & Mitigation Status")
def get_order_blocks(
    symbol: str,
    period: str = Query("1mo", description="Historical period"),
    interval: str = Query("15m", description="Candle interval"),
):
    """
    Identifies institutional Bullish Demand Zones & Bearish Supply Zones with exact [Low - High] price bounds,
    50% Mean Threshold, Mitigation lifecycle tracking (UNMITIGATED_FRESH, TESTED_REJECTED, BREACHED_BREAKER),
    and Fair Value Gaps (FVG).
    """
    df = market_data_service.get_historical_candles(symbol, period=period, interval=interval)
    if df is None:
        raise HTTPException(status_code=404, detail=f"Candle data not found for symbol: {symbol}")
    return institutional_order_flow_service.detect_order_blocks(df)


@router.get("/institutional-flow/{symbol}", summary="Get Institutional Buyer vs Seller Flow & Timeline")
def get_institutional_flow(
    symbol: str,
    period: str = Query("1mo", description="Historical period"),
    interval: str = Query("15m", description="Candle interval"),
):
    """
    Analyzes institutional buyer vs seller order flow:
    - Cumulative Volume Delta (CVD)
    - Institutional Buying vs Selling Phase (Accumulation, Markup, Distribution, Markdown)
    - Delta Divergences (Bullish Absorption vs Bearish Exhaustion)
    - Chronological Institutional Activity Timeline with timestamps (IST) and price levels.
    """
    df = market_data_service.get_historical_candles(symbol, period=period, interval=interval)
    if df is None:
        raise HTTPException(status_code=404, detail=f"Candle data not found for symbol: {symbol}")
    return institutional_order_flow_service.analyze_institutional_flow_and_timeline(df, symbol=symbol)


@router.get("/smart-money/{symbol}", summary="Complete Institutional Order Blocks & Smart Money Footprint")
def get_smart_money_snapshot(
    symbol: str,
    period: str = Query("1mo", description="Historical period"),
    interval: str = Query("15m", description="Candle interval"),
):
    """
    Synthesizes complete Institutional Order Block & Order Flow Intelligence:
    1. Bullish Demand & Bearish Supply Order Blocks with mitigation status.
    2. Unfilled Fair Value Gaps (FVGs).
    3. Real-time CVD and Institutional Phase (Accumulation / Distribution).
    4. Chronological Timeline of WHEN institutions bought and sold.
    5. Option Chain Institutional Writing Footprint.
    """
    df = market_data_service.get_historical_candles(symbol, period=period, interval=interval)
    if df is None:
        raise HTTPException(status_code=404, detail=f"Candle data not found for symbol: {symbol}")
    live = market_data_service.get_live_price(symbol)
    cur_p = live["price"] if live and live.get("price") else 0.0
    pcr_data = pcr_service.analyze_pcr(symbol, spot_price=cur_p)
    return institutional_order_flow_service.get_comprehensive_institutional_snapshot(
        symbol=symbol,
        df=df,
        live_price=cur_p,
        pcr_data=pcr_data,
    )


@router.get("/wm/{symbol}", summary="Detect W-Pattern (Double Bottom) & M-Pattern (Double Top)")
def get_wm_patterns(
    symbol: str,
    period: str = Query("1mo", description="Historical period"),
    interval: str = Query("15m", description="Candle interval"),
):
    """Detects W and M chart patterns with neckline breakout/breakdown levels and projected targets."""
    df = market_data_service.get_historical_candles(symbol, period=period, interval=interval)
    if df is None:
        raise HTTPException(status_code=404, detail=f"Candle data not found for symbol: {symbol}")
    return pattern_service.detect_wm_patterns(df)


@router.get("/volume/{symbol}", summary="Volume Spikes, Moving Averages, & VWAP Analysis")
def get_volume_analysis(
    symbol: str,
    period: str = Query("5d", description="Historical period"),
    interval: str = Query("15m", description="Candle interval"),
):
    """Analyzes volume spikes (>1.5x/2.0x 20-period SMA), volume bias, and price vs VWAP."""
    df = market_data_service.get_historical_candles(symbol, period=period, interval=interval)
    if df is None:
        raise HTTPException(status_code=404, detail=f"Candle data not found for symbol: {symbol}")
    return pattern_service.analyze_volume(df)


@router.get("/vix", summary="Get live India VIX volatility regime and fear gauge")
def get_india_vix():
    """Returns live India VIX (^INDIAVIX), volatility regime classification, and options trading suggestions."""
    return volatility_service.get_india_vix()


@router.get("/giftnifty", summary="Get live GIFT NIFTY index, gap projection & global market cues")
def get_gift_nifty():
    """Returns live GIFT NIFTY index projection, opening gap prediction vs NIFTY 50 spot, and global cues."""
    return volatility_service.get_gift_nifty_and_global_cues()


@router.get("/giftnifty/multi-source", summary="Multi-source GIFT NIFTY comparing Direct Feed vs Econometric Synthesis")
def get_gift_nifty_multi_source():
    """Returns detailed comparison of Direct NSE IX Ticker vs Econometric Multi-Factor Synthesis Engine."""
    return volatility_service.get_gift_nifty_multi_source()


@router.get("/international-cockpit", summary="Comprehensive International Indices & Macro Cues Cockpit")
def get_international_cockpit():
    """
    Returns classified global market benchmarks across US Markets & Futures (45%),
    Asian Morning Cues (30%), European Session (15%), and Macro Commodities & Currencies (10%)
    with Sector Impact Analysis and FII Institutional Flow Predictions.
    """
    return volatility_service.get_international_indices_cockpit()


@router.get("/session-status", summary="Get Indian market hours and active session status")
def get_market_session_status(symbol: Optional[str] = Query("^NSEI", description="Symbol to query (e.g. ^NSEI, CRUDEOIL)")):
    """Returns real-time session open/close status in Indian Standard Time (IST)."""
    return market_data_service.is_market_open(symbol)


@router.get("/angel/status", summary="Get Angel One SmartAPI connection status")
def get_angel_status():
    """Returns current connection, credential configuration, and supported instruments for Angel One."""
    return angel_client.get_status()


@router.get("/live-tick", summary="Consolidated 1-second real-time terminal snapshot")
def get_live_tick():
    """
    Returns live prices, options signals, watchlist, commodities, VIX, and broker status
    in a single consolidated <5ms response for real-time 1-second UI updates.
    """
    return market_data_service.get_live_dashboard_snapshot()


@router.post("/angel/connect", summary="Trigger Angel One SmartAPI connection")
def connect_angel():
    """Attempts to authenticate and start a live SmartAPI session with configured credentials."""
    try:
        res = angel_client.login()
        return {"status": "success", "message": "Angel One connected successfully", "data": res}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/local-data/status", summary="Local market data storage & persistence status")
def get_local_storage_status():
    """
    Returns statistics on all locally stored historical candle datasets,
    live tick streams, and market snapshots.
    """
    return local_data_service.get_storage_status()


@router.post("/local-data/sync", summary="Synchronize all tracked symbols to local disk")
def sync_all_symbols_to_local(force_refresh: bool = Query(False, description="Force re-fetch from network")):
    """
    Downloads and merges multi-timeframe candles (15m, 1d, 1wk) for all tracked
    Indices, MCX Commodities, and Watchlist Equities directly to local storage.
    """
    return local_data_service.sync_all_tracked_symbols(market_data_service, force_refresh=force_refresh)


@router.get("/strategies/catalog", summary="Get Full Strategies & Indicators Catalog")
def get_strategies_catalog():
    """Returns the comprehensive catalog of all 15+ underlying strategies and indicator parameters."""
    return {
        "count": len(unified_strategy_service.get_strategies_catalog()),
        "strategies": unified_strategy_service.get_strategies_catalog(),
    }


@router.get("/strategies/master/{symbol}", summary="Evaluate 4-Pillar Simple Master Strategy")
def evaluate_master_strategy(
    symbol: str,
    period: str = Query("5d", description="Historical period"),
    interval: str = Query("15m", description="Candle interval"),
):
    """
    Evaluates the simplified 4-Pillar Master Decision Strategy:
    1. Pillar 1: Directional Bias (Supertrend + Multi-EMA + Slope)
    2. Pillar 2: Smart Money Fuel (CVD + Bar Delta + Order Blocks)
    3. Pillar 3: High-Probability Location (Demand/Supply Zones + Camarilla + CPR + Fibonacci)
    4. Pillar 4: Risk-Reward & Quality Gate (Choppiness Index CHOP < 61.8 + Min 5 Pts Move)
    """
    live = market_data_service.get_live_price(symbol)
    cur_p = live["price"] if live and live.get("price") else 0.0
    pcr_data = pcr_service.analyze_pcr(symbol, spot_price=cur_p)
    return unified_strategy_service.evaluate_simple_master_strategy(
        symbol=symbol,
        live_price=cur_p,
        pcr_data=pcr_data,
        period=period,
        interval=interval,
    )


@router.get("/strategies/backtest/{symbol}", summary="Run Backtest & Auto-Correction on Previous Data")
@router.post("/strategies/backtest/{symbol}", summary="Run Backtest & Auto-Correction on Previous Data")
def run_strategy_backtest_and_autocorrect(
    symbol: str,
    period: str = Query("1mo", description="Historical period for backtest"),
    interval: str = Query("15m", description="Candle interval"),
    min_winrate: float = Query(70.0, description="Minimum acceptable target winrate percentage"),
    send_instagram: bool = Query(False, description="Whether to dispatch backtest report with tested dates to Instagram"),
    send_telegram: bool = Query(False, description="Whether to dispatch backtest report with tested dates to Telegram"),
    recipient_id: Optional[str] = Query(None, description="Optional target recipient ID / Chat ID"),
):
    """
    Runs historical candle-by-candle simulation on previous market data, evaluates performance
    metrics (Win Rate, Total Points, Target 1 / 2 Hits, Stop Loss Hits), and automatically
    auto-corrects strategy parameters if win rate is below threshold.
    Optionally dispatches the backtest report with the exact tested date range to Telegram or Instagram.
    """
    res = unified_strategy_service.backtest_and_auto_correct(
        symbol=symbol,
        period=period,
        interval=interval,
        min_acceptable_winrate=min_winrate,
    )

    if send_telegram:
        tg_res = telegram_service.send_backtest_report(
            backtest_data=res,
            chat_id=recipient_id,
        )
        res["telegram_dispatch"] = tg_res

    if send_instagram:
        ig_res = instagram_service.send_backtest_report(
            backtest_data=res,
            recipient_id=recipient_id,
        )
        res["instagram_dispatch"] = ig_res

    return res


@router.post("/strategies/backtest/dispatch-telegram/{symbol}", summary="Run Backtest and Dispatch Report to Telegram")
def dispatch_backtest_to_telegram(
    symbol: str,
    period: str = Query("1mo", description="Historical period for backtest"),
    interval: str = Query("15m", description="Candle interval"),
    min_winrate: float = Query(70.0, description="Minimum acceptable target winrate percentage"),
    chat_id: Optional[str] = Query(None, description="Target Telegram Chat ID (defaults to 9100040008 / configured chat)"),
):
    """
    Simulates historical candle-by-candle execution on previous data, executes auto-correction,
    and directly dispatches the formatted report with tested dates to Telegram (9100040008).
    """
    res = unified_strategy_service.backtest_and_auto_correct(
        symbol=symbol,
        period=period,
        interval=interval,
        min_acceptable_winrate=min_winrate,
    )
    tg_res = telegram_service.send_backtest_report(
        backtest_data=res,
        chat_id=chat_id,
    )
    return {
        "symbol": symbol,
        "date_range": res.get("date_range"),
        "performance": res.get("performance"),
        "calibrated_parameters": res.get("calibrated_parameters"),
        "auto_corrections_applied": res.get("auto_corrections_applied"),
        "telegram_delivery": tg_res,
    }


@router.post("/strategies/backtest/dispatch-instagram/{symbol}", summary="Run Backtest and Dispatch Report to Instagram")
def dispatch_backtest_to_instagram(
    symbol: str,
    period: str = Query("1mo", description="Historical period for backtest"),
    interval: str = Query("15m", description="Candle interval"),
    min_winrate: float = Query(70.0, description="Minimum acceptable target winrate percentage"),
    recipient_id: Optional[str] = Query(None, description="Target Instagram recipient ID (defaults to 9100040008)"),
):
    """
    Simulates historical candle-by-candle execution on previous data, executes auto-correction,
    and directly dispatches the formatted report with tested dates to Instagram Direct (9100040008).
    """
    res = unified_strategy_service.backtest_and_auto_correct(
        symbol=symbol,
        period=period,
        interval=interval,
        min_acceptable_winrate=min_winrate,
    )
    ig_res = instagram_service.send_backtest_report(
        backtest_data=res,
        recipient_id=recipient_id,
    )
    return {
        "symbol": symbol,
        "date_range": res.get("date_range"),
        "performance": res.get("performance"),
        "calibrated_parameters": res.get("calibrated_parameters"),
        "auto_corrections_applied": res.get("auto_corrections_applied"),
        "instagram_delivery": ig_res,
    }



@router.get("/local-data/candles/{symbol}", summary="Load stored historical candles from local disk")
def get_local_candles(
    symbol: str,
    interval: str = Query("15m", description="Candle interval (15m, 1d, 1wk)"),
):
    """Retrieves locally persisted OHLCV candles with zero network latency."""
    canon = market_data_service.normalize_symbol(symbol)
    df = local_data_service.load_candles(canon, interval)
    if df is None or df.empty:
        # Fallback to market_data_service to fetch, merge, and save
        df = market_data_service.get_historical_candles(canon, period="1mo", interval=interval)
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail=f"No local or remote candle data found for {symbol}")

    records = []
    for idx, row in df.tail(100).iterrows():
        records.append({
            "timestamp": str(idx),
            "open": round(float(row["open"]), 2),
            "high": round(float(row["high"]), 2),
            "low": round(float(row["low"]), 2),
            "close": round(float(row["close"]), 2),
            "volume": int(row.get("volume", 0)),
        })
    return {
        "symbol": canon,
        "interval": interval,
        "total_candles": len(df),
        "recent_candles": records,
    }


@router.get("/local-data/snapshots/latest", summary="Get latest locally persisted market snapshot")
def get_latest_local_snapshot():
    """Loads latest saved market snapshot from local disk."""
    snap = local_data_service.load_latest_snapshot()
    if not snap:
        # Generate and save fresh snapshot
        return market_data_service.get_live_dashboard_snapshot()
    return snap


