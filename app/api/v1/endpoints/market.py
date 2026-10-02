from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List, Dict, Any
from app.services.market_data import market_data_service
from app.services.strategy_engine import strategy_engine
from app.services.pcr_service import pcr_service
from app.services.news_service import news_service
from app.services.pattern_service import pattern_service
from app.services.volatility_service import volatility_service
from app.services.angel_service import angel_client

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
def get_order_blocks(
    symbol: str,
    period: str = Query("1mo", description="Historical period"),
    interval: str = Query("15m", description="Candle interval"),
):
    """Identifies institutional Bullish/Bearish Order Blocks (Demand/Supply Zones) and Fair Value Gaps (FVG)."""
    df = market_data_service.get_historical_candles(symbol, period=period, interval=interval)
    if df is None:
        raise HTTPException(status_code=404, detail=f"Candle data not found for symbol: {symbol}")
    return pattern_service.detect_order_blocks(df)


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

