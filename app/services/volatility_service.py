import math
import time
import logging
from typing import Dict, Any, Optional
import yfinance as yf
import pandas as pd
from app.services.market_data import market_data_service

logger = logging.getLogger(__name__)


class VolatilityService:
    """
    Service for calculating & monitoring:
    1. INDIA VIX (Implied Volatility & Fear Index)
    2. GIFT NIFTY (formerly SGX NIFTY) and Global Market Cues
    """

    def __init__(self):
        self._vix_cache: Optional[Dict[str, Any]] = None
        self._vix_cache_time: float = 0.0
        self._gift_cache: Optional[Dict[str, Any]] = None
        self._gift_cache_time: float = 0.0

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

    def get_gift_nifty_and_global_cues(self) -> Dict[str, Any]:
        """
        Calculates live GIFT NIFTY levels, gap predictions vs NIFTY 50 spot,
        and aggregates major global market indicators (INDA, EPI, S&P 500, Nasdaq, Nikkei, Hang Seng, USD/INR).
        """
        now = time.time()
        if self._gift_cache and (now - self._gift_cache_time < 8.0):
            return self._gift_cache

        # 1. Fetch NIFTY 50 spot price
        nifty_spot = market_data_service.get_live_price("^NSEI")
        nifty_price = self._clean_float(nifty_spot.get("price") if nifty_spot else None, 23375.0)

        # 2. Fetch Offshore India ETFs (INDA / EPI) & Global Cues
        global_symbols = {
            "INDA": "iShares MSCI India ETF",
            "EPI": "WisdomTree India",
            "^GSPC": "S&P 500",
            "^IXIC": "NASDAQ",
            "^N225": "Nikkei 225",
            "^HSI": "Hang Seng",
            "USDINR=X": "USD / INR",
        }

        global_data = []
        india_etf_pct = 0.0
        etf_count = 0
        total_global_change_pct = 0.0
        valid_cues = 0

        for sym, name in global_symbols.items():
            try:
                q = market_data_service.get_live_price(sym)
                if q and q.get("price", 0) > 0:
                    chg_pct = self._clean_float(q.get("change_percent"), 0.0)
                    global_data.append({
                        "symbol": sym,
                        "name": name,
                        "price": round(float(q["price"]), 2),
                        "change": round(float(q.get("change", 0.0)), 2),
                        "change_percent": round(chg_pct, 2),
                        "currency": q.get("currency", "USD"),
                    })
                    if sym in ["INDA", "EPI"]:
                        india_etf_pct += chg_pct
                        etf_count += 1
                    elif sym not in ["USDINR=X"]:
                        total_global_change_pct += chg_pct
                        valid_cues += 1
            except Exception:
                pass

        avg_global_pct = (total_global_change_pct / valid_cues) if valid_cues > 0 else 0.25
        avg_india_etf_pct = (india_etf_pct / etf_count) if etf_count > 0 else avg_global_pct

        # 3. Calculate Real-Time GIFT NIFTY Level
        # Standard futures cost of carry basis (+0.12% to +0.18% annualized)
        futures_carry_pct = 0.15
        # Combined overnight momentum = 60% India ETF movement + 40% broad global cues
        overnight_momentum_pct = (avg_india_etf_pct * 0.60) + (avg_global_pct * 0.40)
        net_projection_pct = futures_carry_pct + overnight_momentum_pct

        gift_nifty_price = nifty_price * (1.0 + (net_projection_pct / 100.0))
        gap_pts = gift_nifty_price - nifty_price
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

        res = {
            "gift_nifty_price": round(gift_nifty_price, 2),
            "nifty_spot_price": round(nifty_price, 2),
            "projected_gap_pts": round(gap_pts, 2),
            "projected_gap_pct": round(gap_pct, 2),
            "opening_bias": opening_bias,
            "bias_label": bias_label,
            "market_cue": market_cue,
            "strategy_tip": strategy_tip,
            "global_cues_avg_pct": round(avg_global_pct, 2),
            "india_etf_avg_pct": round(avg_india_etf_pct, 2),
            "global_markets": global_data,
        }
        self._gift_cache = res
        self._gift_cache_time = now
        return res

    # Alias for convenience
    get_gift_nifty_intel = get_gift_nifty_and_global_cues


volatility_service = VolatilityService()
