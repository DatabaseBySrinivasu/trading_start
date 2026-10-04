import math
import time
import logging
from typing import Dict, Any, Optional, List, Tuple
import yfinance as yf
import pandas as pd
from app.services.market_data import market_data_service

logger = logging.getLogger(__name__)


class VolatilityService:
    """
    Service for calculating & monitoring:
    1. INDIA VIX (Implied Volatility & Fear Index)
    2. GIFT NIFTY Multi-Source Ingestion & Gap Prediction Engine
       - Direct Feed (NSE IX Ticker / Quotes)
       - Econometric Multi-Factor Synthesis Engine
    3. Global Market Cues & Comprehensive International Indices Cockpit
       - US Markets & Futures (45% Dalal Street Weight)
       - Asian Markets (30% Morning Pre-Market Weight)
       - European Markets (15% Afternoon Session Weight)
       - Macro Commodities & Currencies (10% Inflation/FII Weight)
       - Sector-Specific Impact & FII Flow Predictions
    """

    def __init__(self):
        self._vix_cache: Optional[Dict[str, Any]] = None
        self._vix_cache_time: float = 0.0
        self._gift_cache: Optional[Dict[str, Any]] = None
        self._gift_cache_time: float = 0.0
        self._intl_cache: Optional[Dict[str, Any]] = None
        self._intl_cache_time: float = 0.0

    @staticmethod
    def _clean_float(val: Any, default: float = 0.0) -> float:
        try:
            if val is None:
                return default
            f = float(val)
            return default if (math.isnan(f) or math.isinf(f)) else f
        except Exception:
            return default

    def get_india_vix(self) -> Dict[str, Any]:
        """
        Fetches live India VIX (^INDIAVIX) from NSE/Yahoo Finance and evaluates
        volatility regimes, market fear, and options trading implications.
        """
        now = time.time()
        if self._vix_cache and (now - self._vix_cache_time < 8.0):
            return self._vix_cache

        price = None
        prev_close = None
        day_high = None
        day_low = None

        try:
            ticker = yf.Ticker("^INDIAVIX")
            fast_info = getattr(ticker, "fast_info", None)

            if fast_info:
                price = getattr(fast_info, "last_price", None)
                prev_close = getattr(fast_info, "previous_close", None)
                day_high = getattr(fast_info, "day_high", None)
                day_low = getattr(fast_info, "day_low", None)

            # Fallback to history if fast_info is missing
            if price is None or price <= 0:
                hist = ticker.history(period="5d", interval="1d")
                if not hist.empty:
                    price = float(hist["Close"].iloc[-1])
                    if len(hist) > 1:
                        prev_close = float(hist["Close"].iloc[-2])
                    else:
                        prev_close = price
                    day_high = float(hist["High"].iloc[-1])
                    day_low = float(hist["Low"].iloc[-1])
        except Exception as e:
            logger.debug(f"India VIX fetch fallback: {e}")

        # Check market data service cache if direct fetch failed
        if price is None or price <= 0:
            md_vix = market_data_service.get_live_price("^INDIAVIX")
            if md_vix:
                price = md_vix.get("price", 12.5)
                prev_close = md_vix.get("previous_close", 12.8)

        price = self._clean_float(price, 12.5)
        prev_close = self._clean_float(prev_close, 12.8)
        day_high = self._clean_float(day_high, price * 1.02)
        day_low = self._clean_float(day_low, price * 0.98)

        change = price - prev_close
        change_pct = (change / prev_close * 100.0) if prev_close > 0 else 0.0

        # Volatility Regimes
        if price < 12.0:
            regime = "LOW_VOLATILITY"
            regime_label = "Complacent / Low Volatility (<12)"
            options_strategy = "Option premiums are cheap. Favorable for Option Buyers (CE/PE) on clear breakout trends. Option sellers face low theta reward."
            sentiment = "BULLISH_STABILITY"
        elif price <= 17.5:
            regime = "IDEAL_VOLATILITY"
            regime_label = "Optimal Trading Zone (12-17.5)"
            options_strategy = "Balanced environment. High-probability zone for directional breakout trades (CPR/PDH/PDL) and systematic credit spreads."
            sentiment = "HEALTHY_MARKET"
        elif price <= 24.0:
            regime = "ELEVATED_VOLATILITY"
            regime_label = "High Volatility / Sharp Swings (17.5-24)"
            options_strategy = "Wide price swings. High reward for momentum Option Buyers with strict stop-losses. Option Sellers must hedge wings."
            sentiment = "ELEVATED_FEAR"
        else:
            regime = "EXTREME_PANIC"
            regime_label = "Extreme Panic / Crash Risk (>24)"
            options_strategy = "Extreme fear in market. Severe gamma risk. Favor deep OTM hedging or buying ATM PE on breakdown confirmation."
            sentiment = "EXTREME_BEARISH_FEAR"

        # Trend bias
        if change_pct < -2.5:
            trend_bias = "VIX_FALLING_COOLING"
            trend_desc = f"VIX is cooling off significantly (-{abs(change_pct):.1f}%), indicating strong underlying institutional market stability."
        elif change_pct > 3.0:
            trend_bias = "VIX_SPIKING_FEAR"
            trend_desc = f"VIX is spiking (+{change_pct:.1f}%), indicating rising market anxiety and demand for protective puts."
        else:
            trend_bias = "VIX_STABLE"
            trend_desc = "VIX is holding stable, reflecting controlled intraday volatility."

        res = {
            "symbol": "^INDIAVIX",
            "name": "India VIX",
            "current_vix": round(price, 2),
            "change": round(change, 2),
            "change_percent": round(change_pct, 2),
            "day_high": round(day_high, 2),
            "day_low": round(day_low, 2),
            "regime": regime,
            "regime_label": regime_label,
            "sentiment": sentiment,
            "trend_bias": trend_bias,
            "description": trend_desc,
            "options_strategy": options_strategy,
        }
        self._vix_cache = res
        self._vix_cache_time = now
        return res

    def get_international_indices_cockpit(self) -> Dict[str, Any]:
        """
        Comprehensive International Indices Cockpit analyzing all global markets
        that have direct econometric correlation with Indian stock market indices (NIFTY 50, BANK NIFTY, SENSEX).

        Categorized by:
        1. US Markets & Futures (45% weight)
        2. Asian Markets (30% weight)
        3. European Markets (15% weight)
        4. Macro Commodities & Currencies (10% weight)
        5. Offshore India Proxies
        """
        now = time.time()
        if self._intl_cache and (now - self._intl_cache_time < 8.0):
            return self._intl_cache

        # Define all international benchmarks
        # Grouped by geographic region & asset class
        us_markets_config = [
            {"symbol": "ES=F", "name": "S&P 500 E-mini Futures", "category": "US_FUTURES", "impact": "HIGH", "weight": 0.15, "currency": "USD"},
            {"symbol": "NQ=F", "name": "Nasdaq 100 E-mini Futures", "category": "US_FUTURES", "impact": "HIGH_IT", "weight": 0.15, "currency": "USD"},
            {"symbol": "^GSPC", "name": "S&P 500", "category": "US_EQUITY", "impact": "HIGH", "weight": 0.05, "currency": "USD"},
            {"symbol": "^IXIC", "name": "NASDAQ Composite", "category": "US_EQUITY", "impact": "HIGH_IT", "weight": 0.05, "currency": "USD"},
            {"symbol": "^DJI", "name": "Dow Jones Industrial", "category": "US_EQUITY", "impact": "MEDIUM", "weight": 0.02, "currency": "USD"},
            {"symbol": "YM=F", "name": "Dow Futures", "category": "US_FUTURES", "impact": "MEDIUM", "weight": 0.03, "currency": "USD"},
            {"symbol": "^TNX", "name": "US 10-Yr Treasury Yield", "category": "US_BONDS", "impact": "HIGH_FII_OUTFLOW", "weight": -0.05, "currency": "USD"},
            {"symbol": "DX-Y.NYB", "name": "US Dollar Index (DXY)", "category": "CURRENCY", "impact": "HIGH_CURRENCY", "weight": -0.05, "currency": "USD"},
        ]

        asian_markets_config = [
            {"symbol": "^N225", "name": "Nikkei 225 (Japan)", "category": "ASIA_EQUITY", "impact": "HIGH_MORNING", "weight": 0.10, "currency": "JPY"},
            {"symbol": "^HSI", "name": "Hang Seng (Hong Kong)", "category": "ASIA_EQUITY", "impact": "HIGH_EM", "weight": 0.08, "currency": "HKD"},
            {"symbol": "^TWII", "name": "Taiwan TAIEX", "category": "ASIA_EQUITY", "impact": "MEDIUM_TECH", "weight": 0.05, "currency": "TWD"},
            {"symbol": "^KS11", "name": "KOSPI (South Korea)", "category": "ASIA_EQUITY", "impact": "MEDIUM_EM", "weight": 0.04, "currency": "KRW"},
            {"symbol": "000001.SS", "name": "Shanghai Composite", "category": "ASIA_EQUITY", "impact": "LOW_CHINA", "weight": 0.03, "currency": "CNY"},
        ]

        european_markets_config = [
            {"symbol": "^FTSE", "name": "FTSE 100 (UK)", "category": "EURO_EQUITY", "impact": "MEDIUM_AFTERNOON", "weight": 0.05, "currency": "GBP"},
            {"symbol": "^GDAXI", "name": "DAX 40 (Germany)", "category": "EURO_EQUITY", "impact": "MEDIUM_AFTERNOON", "weight": 0.05, "currency": "EUR"},
            {"symbol": "^FCHI", "name": "CAC 40 (France)", "category": "EURO_EQUITY", "impact": "LOW_AFTERNOON", "weight": 0.05, "currency": "EUR"},
        ]

        macro_config = [
            {"symbol": "BZ=F", "name": "Brent Crude Oil", "category": "COMMODITY", "impact": "HIGH_INFLATION_DEFICIT", "weight": -0.08, "currency": "USD"},
            {"symbol": "USDINR=X", "name": "USD/INR", "category": "CURRENCY", "impact": "HIGH_RUPEE_WEAKNESS", "weight": -0.04, "currency": "INR"},
            {"symbol": "GC=F", "name": "Gold (USD/oz)", "category": "COMMODITY", "impact": "SAFE_HAVEN", "weight": 0.02, "currency": "USD"},
        ]

        offshore_etfs_config = [
            {"symbol": "INDA", "name": "iShares MSCI India ETF", "category": "INDIA_OFFSHORE_ETF", "impact": "DIRECT_FII_PROXY", "weight": 0.08, "currency": "USD"},
            {"symbol": "EPI", "name": "WisdomTree India Earnings", "category": "INDIA_OFFSHORE_ETF", "impact": "DIRECT_FII_PROXY", "weight": 0.05, "currency": "USD"},
            {"symbol": "INDY", "name": "iShares India 50 ETF", "category": "INDIA_OFFSHORE_ETF", "impact": "DIRECT_NIFTY_PROXY", "weight": 0.05, "currency": "USD"},
        ]

        def _fetch_group(items: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], float]:
            group_results = []
            total_change = 0.0
            valid = 0
            for item in items:
                sym = item["symbol"]
                try:
                    q = market_data_service.get_live_price(sym)
                    if q and q.get("price", 0) > 0:
                        chg_pct = self._clean_float(q.get("change_percent"), 0.0)
                        pr = self._clean_float(q.get("price"), 0.0)
                        chg = self._clean_float(q.get("change"), 0.0)
                        group_results.append({
                            "symbol": sym,
                            "name": item["name"],
                            "price": round(pr, 2),
                            "change": round(chg, 2),
                            "change_percent": round(chg_pct, 2),
                            "category": item["category"],
                            "impact": item["impact"],
                            "currency": q.get("currency", item["currency"]),
                            "weight": item["weight"],
                            "status": "BULLISH" if chg_pct > 0.15 else ("BEARISH" if chg_pct < -0.15 else "NEUTRAL"),
                        })
                        total_change += chg_pct
                        valid += 1
                except Exception as ex:
                    logger.debug(f"Error fetching international symbol {sym}: {ex}")
            avg_chg = (total_change / valid) if valid > 0 else 0.0
            return group_results, avg_chg

        us_data, us_avg_pct = _fetch_group(us_markets_config)
        asian_data, asian_avg_pct = _fetch_group(asian_markets_config)
        european_data, euro_avg_pct = _fetch_group(european_markets_config)
        macro_data, macro_avg_pct = _fetch_group(macro_config)
        etf_data, etf_avg_pct = _fetch_group(offshore_etfs_config)

        # Compute Composite Global Sentiment Score (-100 to +100)
        # Weights: US (40%), Asia (25%), Europe (15%), India ETFs (12%), Macro/Currencies (8%)
        nasdaq_pct = next((x["change_percent"] for x in us_data if x["symbol"] in ["NQ=F", "^IXIC"]), us_avg_pct)
        sp500_pct = next((x["change_percent"] for x in us_data if x["symbol"] in ["ES=F", "^GSPC"]), us_avg_pct)
        crude_pct = next((x["change_percent"] for x in macro_data if x["symbol"] == "BZ=F"), 0.0)
        us10y_pct = next((x["change_percent"] for x in us_data if x["symbol"] == "^TNX"), 0.0)
        dxy_pct = next((x["change_percent"] for x in us_data if x["symbol"] in ["DX-Y.NYB", "DXY"]), 0.0)

        # Multi-factor score formula
        raw_sentiment_score = (
            (sp500_pct * 20.0) +
            (nasdaq_pct * 20.0) +
            (asian_avg_pct * 25.0) +
            (euro_avg_pct * 15.0) +
            (etf_avg_pct * 15.0) -
            (crude_pct * 10.0) -  # Spiking crude is negative for Indian imports
            (us10y_pct * 10.0) -  # Spiking bond yield causes FII capital flight
            (dxy_pct * 8.0)       # Spiking USD strengthens DXY against INR
        )

        sentiment_score = round(max(-100.0, min(100.0, raw_sentiment_score)), 1)

        if sentiment_score >= 35.0:
            sentiment_bias = "STRONG_BULLISH"
            sentiment_label = f"Global Bullish Surge (+{sentiment_score:.1f})"
            fii_flow_bias = "NET_INFLOW_FAVORABLE"
            fii_description = "Strong global risk-on rally. High probability of institutional FII buying on Dalal Street."
        elif sentiment_score >= 10.0:
            sentiment_bias = "MILD_BULLISH"
            sentiment_label = f"Positive Global Backdrop (+{sentiment_score:.1f})"
            fii_flow_bias = "MILD_INFLOW"
            fii_description = "Supportive international cues. Favorable for opening gap-ups and dip-buying."
        elif sentiment_score <= -35.0:
            sentiment_bias = "STRONG_BEARISH"
            sentiment_label = f"Global Risk-Off Panic ({sentiment_score:.1f})"
            fii_flow_bias = "NET_OUTFLOW_RISK"
            fii_description = "Severe global headwinds, elevated US bond yields, or spiking crude. High FII selling pressure."
        elif sentiment_score <= -10.0:
            sentiment_bias = "MILD_BEARISH"
            sentiment_label = f"Cautious Global Tone ({sentiment_score:.1f})"
            fii_flow_bias = "MILD_OUTFLOW"
            fii_description = "Subdued international cues. Resistance zones and CPR tops likely to hold."
        else:
            sentiment_bias = "NEUTRAL_CHOP"
            sentiment_label = f"Mixed / Neutral Cues ({sentiment_score:+.1f})"
            fii_flow_bias = "BALANCED_NEUTRAL"
            fii_description = "Global indices divergent. Domestic technical levels (CPR, VWAP, Support/Resistance) will dictate direction."

        # Sector Specific Impacts
        it_sector_impact = {
            "bias": "BULLISH" if nasdaq_pct > 0.25 else ("BEARISH" if nasdaq_pct < -0.25 else "NEUTRAL"),
            "driver": f"Nasdaq / NQ Futures ({nasdaq_pct:+.2f}%)",
            "impacted_stocks": ["TCS", "INFY", "WIPRO", "HCLTECH", "TECHM"],
            "description": "Indian IT bellwethers track Nasdaq momentum closely.",
        }

        banking_sector_impact = {
            "bias": "BULLISH" if (us10y_pct <= 0.0 and dxy_pct <= 0.0) else ("BEARISH" if (us10y_pct > 1.0 or dxy_pct > 0.3) else "NEUTRAL"),
            "driver": f"US 10Y Yield ({us10y_pct:+.2f}%) & DXY ({dxy_pct:+.2f}%)",
            "impacted_stocks": ["HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK"],
            "description": "Bank Nifty is highly sensitive to FII liquidity, interest rate expectations, and sovereign bond yields.",
        }

        auto_energy_impact = {
            "bias": "BEARISH" if crude_pct > 1.5 else ("BULLISH" if crude_pct < -1.5 else "NEUTRAL"),
            "driver": f"Brent Crude Oil ({crude_pct:+.2f}%)",
            "impacted_stocks": ["MARUTI", "TATAMOTORS", "ASIANPAINT", "BPCL", "ONGC"],
            "description": "Lower crude oil prices expand operating margins for Indian Auto, Paint, and Aviation companies.",
        }

        result = {
            "sentiment_score": sentiment_score,
            "sentiment_bias": sentiment_bias,
            "sentiment_label": sentiment_label,
            "fii_flow_bias": fii_flow_bias,
            "fii_description": fii_description,
            "regional_averages": {
                "us_markets_pct": round(us_avg_pct, 2),
                "asian_markets_pct": round(asian_avg_pct, 2),
                "european_markets_pct": round(euro_avg_pct, 2),
                "india_etfs_pct": round(etf_avg_pct, 2),
                "macro_commodities_pct": round(macro_avg_pct, 2),
            },
            "us_markets": us_data,
            "asian_markets": asian_data,
            "european_markets": european_data,
            "macro_commodities": macro_data,
            "offshore_india_etfs": etf_data,
            "sector_impacts": {
                "it_sector": it_sector_impact,
                "banking_sector": banking_sector_impact,
                "auto_and_energy": auto_energy_impact,
            },
        }

        self._intl_cache = result
        self._intl_cache_time = now
        return result

    def get_gift_nifty_multi_source(self) -> Dict[str, Any]:
        """
        Multi-Source GIFT NIFTY Engine:
        1. Direct Ticker / Exchange Feed (NSE IX GIFT NIFTY / ^NSEIX / NIFTY_F1)
        2. Multi-Factor Econometric Synthesis Engine:
           Calculates synthetic GIFT NIFTY from Spot NIFTY, Cost of Carry, US Futures (ES=F, NQ=F),
           Asian cues (Nikkei, Hang Seng, TAIEX), Offshore India ETFs (INDA, EPI, INDY),
           and Currency drag (USD/INR, DXY).
        """
        # Fetch NIFTY 50 spot price
        nifty_spot = market_data_service.get_live_price("^NSEI")
        nifty_price = self._clean_float(nifty_spot.get("price") if nifty_spot else None, 23375.0)

        # 1. Check Direct Feed for GIFT NIFTY
        direct_feed = None
        direct_price = None
        direct_change = 0.0
        direct_change_pct = 0.0
        direct_source_name = "None"

        # Check GIFT_NIFTY, ^NSEIX in market data
        for sym in ["GIFT_NIFTY", "^NSEIX", "NIFTY_F1"]:
            try:
                q = market_data_service.get_live_price(sym)
                if q and q.get("price", 0) > 1000.0:
                    direct_price = float(q["price"])
                    direct_change = self._clean_float(q.get("change"), 0.0)
                    direct_change_pct = self._clean_float(q.get("change_percent"), 0.0)
                    direct_source_name = f"NSE IX Live Ticker ({sym})"
                    direct_feed = {
                        "symbol": sym,
                        "source": direct_source_name,
                        "price": round(direct_price, 2),
                        "change": round(direct_change, 2),
                        "change_percent": round(direct_change_pct, 2),
                        "status": "ACTIVE_LIVE",
                    }
                    break
            except Exception:
                pass

        # 2. Calculate Econometric Multi-Factor Synthetic GIFT NIFTY
        intl_cockpit = self.get_international_indices_cockpit()
        us_avg_pct = intl_cockpit["regional_averages"]["us_markets_pct"]
        asian_avg_pct = intl_cockpit["regional_averages"]["asian_markets_pct"]
        etf_avg_pct = intl_cockpit["regional_averages"]["india_etfs_pct"]
        dxy_pct = next((x["change_percent"] for x in intl_cockpit.get("us_markets", []) if x["symbol"] in ["DX-Y.NYB", "DXY"]), 0.0)
        usdinr_pct = next((x["change_percent"] for x in intl_cockpit.get("macro_commodities", []) if x["symbol"] == "USDINR=X"), 0.0)

        # Futures cost of carry basis (+0.12% to +0.18% annualized)
        futures_carry_pct = 0.15

        # Overnight momentum formulation:
        # 40% US Futures + 30% Asian Morning Cues + 20% Offshore India ETFs - 10% DXY/USDINR Drag
        currency_drag = (dxy_pct * 0.5) + (usdinr_pct * 0.5)
        overnight_momentum_pct = (
            (us_avg_pct * 0.40) +
            (asian_avg_pct * 0.30) +
            (etf_avg_pct * 0.20) -
            (currency_drag * 0.10)
        )

        net_synthetic_pct = futures_carry_pct + overnight_momentum_pct
        synthetic_price = nifty_price * (1.0 + (net_synthetic_pct / 100.0))
        synthetic_spread_pts = synthetic_price - nifty_price

        synthetic_feed = {
            "source": "Econometric Multi-Factor Synthesis Engine",
            "price": round(synthetic_price, 2),
            "basis_carry_pct": futures_carry_pct,
            "overnight_momentum_pct": round(overnight_momentum_pct, 2),
            "projected_spread_pts": round(synthetic_spread_pts, 2),
            "status": "COMPUTED_CONTINUOUS",
            "drivers": {
                "us_futures_contribution": f"{us_avg_pct:+.2f}%",
                "asian_cues_contribution": f"{asian_avg_pct:+.2f}%",
                "india_etf_contribution": f"{etf_avg_pct:+.2f}%",
                "currency_drag": f"{currency_drag:+.2f}%",
            }
        }

        # 3. Determine Selected Active Level
        if direct_feed and direct_price and direct_price > 1000.0:
            selected_price = direct_price
            selected_source = direct_feed["source"]
            is_direct_live = True
        else:
            selected_price = synthetic_price
            selected_source = synthetic_feed["source"]
            is_direct_live = False

        gap_pts = selected_price - nifty_price
        gap_pct = (gap_pts / nifty_price) * 100.0 if nifty_price > 0 else 0.0

        if gap_pts >= 40.0:
            opening_bias = "STRONG_GAP_UP"
            bias_label = f"Strong Gap Up (+{gap_pts:.1f} pts)"
            market_cue = "BULLISH"
            strategy_tip = "Look for dip-buying near Previous Day High (PDH) or CPR Top. Avoid chasing extended green candles."
        elif gap_pts >= 12.0:
            opening_bias = "MILD_GAP_UP"
            bias_label = f"Mild Gap Up (+{gap_pts:.1f} pts)"
            market_cue = "BULLISH_MILD"
            strategy_tip = "Healthy opening. Check 15m CPR breakout and EMA 9/21 cross for long entry."
        elif gap_pts <= -40.0:
            opening_bias = "STRONG_GAP_DOWN"
            bias_label = f"Strong Gap Down ({gap_pts:.1f} pts)"
            market_cue = "BEARISH"
            strategy_tip = "Aggressive downside pressure. Watch Previous Day Low (PDL) breakdown for PUT continuation."
        elif gap_pts <= -12.0:
            opening_bias = "MILD_GAP_DOWN"
            bias_label = f"Mild Gap Down ({gap_pts:.1f} pts)"
            market_cue = "BEARISH_MILD"
            strategy_tip = "Mild weakness. Wait for first 15m candle close relative to Central Pivot Range."
        else:
            opening_bias = "FLAT_NEUTRAL"
            bias_label = f"Flat Opening ({gap_pts:+.1f} pts)"
            market_cue = "NEUTRAL"
            strategy_tip = "Range-bound open. Central Pivot Range (CPR) and Volume VWAP will dictate intraday trend."

        return {
            "selected_price": round(selected_price, 2),
            "selected_source": selected_source,
            "is_direct_live": is_direct_live,
            "nifty_spot_price": round(nifty_price, 2),
            "projected_gap_pts": round(gap_pts, 2),
            "projected_gap_pct": round(gap_pct, 2),
            "opening_bias": opening_bias,
            "bias_label": bias_label,
            "market_cue": market_cue,
            "strategy_tip": strategy_tip,
            "direct_feed": direct_feed or {
                "symbol": "GIFT_NIFTY",
                "source": "NSE IX (Offline/Simulated)",
                "price": round(synthetic_price, 2),
                "change": round(synthetic_spread_pts, 2),
                "change_percent": round(gap_pct, 2),
                "status": "STANDBY_MODE",
            },
            "synthetic_feed": synthetic_feed,
            "international_cockpit": intl_cockpit,
        }

    def get_gift_nifty_and_global_cues(self) -> Dict[str, Any]:
        """
        Main public interface for GIFT NIFTY and Global Cues.
        Seamlessly integrates Multi-Source GIFT NIFTY, International Cockpit,
        and backward-compatible fields.
        """
        now = time.time()
        if self._gift_cache and (now - self._gift_cache_time < 8.0):
            return self._gift_cache

        multi = self.get_gift_nifty_multi_source()
        intl = multi["international_cockpit"]

        # Flatten list of key global markets for dashboard display
        global_data = []
        for m in (intl.get("us_markets", [])[:3] + intl.get("asian_markets", [])[:3] + intl.get("macro_commodities", [])[:2] + intl.get("offshore_india_etfs", [])[:2]):
            global_data.append({
                "symbol": m["symbol"],
                "name": m["name"],
                "price": m["price"],
                "change": m["change"],
                "change_percent": m["change_percent"],
                "currency": m.get("currency", "USD"),
            })

        res = {
            "gift_nifty_price": multi["selected_price"],
            "nifty_spot_price": multi["nifty_spot_price"],
            "projected_gap_pts": multi["projected_gap_pts"],
            "projected_gap_pct": multi["projected_gap_pct"],
            "opening_bias": multi["opening_bias"],
            "bias_label": multi["bias_label"],
            "market_cue": multi["market_cue"],
            "strategy_tip": multi["strategy_tip"],
            "selected_source": multi["selected_source"],
            "is_direct_live": multi["is_direct_live"],
            "global_cues_avg_pct": intl["regional_averages"]["us_markets_pct"],
            "india_etf_avg_pct": intl["regional_averages"]["india_etfs_pct"],
            "global_sentiment_score": intl["sentiment_score"],
            "global_sentiment_bias": intl["sentiment_bias"],
            "fii_flow_bias": intl["fii_flow_bias"],
            "fii_description": intl["fii_description"],
            "sector_impacts": intl["sector_impacts"],
            "global_markets": global_data,
            "international_cockpit": intl,
            "multi_source": {
                "direct_feed": multi["direct_feed"],
                "synthetic_feed": multi["synthetic_feed"],
            },
        }

        self._gift_cache = res
        self._gift_cache_time = now
        return res

    # Alias for convenience
    get_gift_nifty_intel = get_gift_nifty_and_global_cues


volatility_service = VolatilityService()
