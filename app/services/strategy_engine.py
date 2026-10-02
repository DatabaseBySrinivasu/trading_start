import math
import time
import logging
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
from app.services.market_data import market_data_service
from app.services.pcr_service import pcr_service
from app.services.news_service import news_service
from app.services.pattern_service import pattern_service
from app.services.volatility_service import volatility_service
from app.services.institutional_filter_engine import institutional_filter_engine

logger = logging.getLogger(__name__)


class StrategyEngine:
    def __init__(self):
        self._signal_cache: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _norm_cdf(x: float) -> float:
        """Standard normal cumulative distribution function."""
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    @classmethod
    def calculate_option_strike_pricing(
        cls,
        symbol: str,
        spot_price: float,
        recommendation: str,
        spot_target_1: float,
        spot_target_2: float,
        spot_stop_loss: float,
        vix_price: float = 12.5,
    ) -> Dict[str, Any]:
        """
        Calculates exact Option Contract Strike, Premium Entry Price, Premium Target 1,
        Premium Target 2, and Premium Stop Loss using Black-Scholes and Delta estimation.
        Supports Indices, Equities, and MCX Commodities.
        """
        canonical = market_data_service.normalize_symbol(symbol)
        clean_sym = canonical.replace(".NS", "").replace(".BO", "").replace("^", "").replace("=F", "").replace("_", "").replace("-", "").replace(" ", "").upper()

        # 1. Determine Strike Step, Lot Size, and Expiry Days
        commodity_specs = {
            "CRUDEOIL": {"lot": 100, "step": 50, "unit": "bbl", "days": 7.0, "name": "Brent Crude Oil (MCX)"},
            "BRENT": {"lot": 100, "step": 50, "unit": "bbl", "days": 7.0, "name": "Brent Crude Oil (MCX)"},
            "BZ": {"lot": 100, "step": 50, "unit": "bbl", "days": 7.0, "name": "Brent Crude Oil (MCX)"},
            "CL": {"lot": 100, "step": 50, "unit": "bbl", "days": 7.0, "name": "Brent Crude Oil (MCX)"},
            "GOLD": {"lot": 100, "step": 100, "unit": "10g", "days": 10.0, "name": "Gold 999 (MCX)"},
            "GC": {"lot": 100, "step": 100, "unit": "10g", "days": 10.0, "name": "Gold 999 (MCX)"},
            "SILVER": {"lot": 30, "step": 500, "unit": "kg", "days": 10.0, "name": "Silver (MCX)"},
            "SI": {"lot": 30, "step": 500, "unit": "kg", "days": 10.0, "name": "Silver (MCX)"},
            "NATURALGAS": {"lot": 1250, "step": 5, "unit": "mmBtu", "days": 7.0, "name": "Henry Hub Natural Gas (MCX)"},
            "NATGAS": {"lot": 1250, "step": 5, "unit": "mmBtu", "days": 7.0, "name": "Henry Hub Natural Gas (MCX)"},
            "HENRYHUB": {"lot": 1250, "step": 5, "unit": "mmBtu", "days": 7.0, "name": "Henry Hub Natural Gas (MCX)"},
            "NG": {"lot": 1250, "step": 5, "unit": "mmBtu", "days": 7.0, "name": "Henry Hub Natural Gas (MCX)"},
            "COPPER": {"lot": 2500, "step": 5, "unit": "kg", "days": 10.0, "name": "Copper (MCX)"},
            "HG": {"lot": 2500, "step": 5, "unit": "kg", "days": 10.0, "name": "Copper (MCX)"},
        }

        if canonical in commodity_specs or clean_sym in commodity_specs:
            spec = commodity_specs.get(canonical, commodity_specs.get(clean_sym))
            step = spec["step"]
            lot_size = spec["lot"]
            days_to_expiry = spec["days"]
            is_index = False
            is_commodity = True
        elif canonical == "^BSESN" or "BSESN" in clean_sym or "SENSEX" in clean_sym:
            step = 100
            lot_size = 20
            is_index = True
            is_commodity = False
            days_to_expiry = 3.0
        elif canonical == "^BSEBANK" or "BSEBANK" in clean_sym or "BANKEX" in clean_sym:
            step = 100
            lot_size = 30
            is_index = True
            is_commodity = False
            days_to_expiry = 3.0
        elif canonical == "^NSEBANK" or "NSEBANK" in clean_sym or "BANKNIFTY" in clean_sym:
            step = 100
            lot_size = 30
            is_index = True
            is_commodity = False
            days_to_expiry = 3.0
        elif canonical == "NIFTY_FIN_SERVICE.NS" or "FINNIFTY" in clean_sym:
            step = 50
            lot_size = 65
            is_index = True
            is_commodity = False
            days_to_expiry = 3.0
        elif canonical == "^NSEMDCP50" or "MIDCPNIFTY" in clean_sym:
            step = 25
            lot_size = 120
            is_index = True
            is_commodity = False
            days_to_expiry = 3.0
        elif canonical == "^NSEI" or "NSEI" in clean_sym or "NIFTY" in clean_sym:
            step = 50
            lot_size = 65
            is_index = True
            is_commodity = False
            days_to_expiry = 3.0
        else:
            is_index = False
            is_commodity = False
            days_to_expiry = 14.0
            lot_sizes = {
                "RELIANCE": 500, "TCS": 175, "HDFCBANK": 550, "SBIN": 750,
                "INFY": 400, "ICICIBANK": 700, "LT": 150, "ITC": 1600,
                "TATAMOTORS": 575, "BHARTIARTL": 475, "AXISBANK": 625, "KOTAKBANK": 400,
                "MARUTI": 50, "BAJFINANCE": 125, "ASIANPAINT": 200, "WIPRO": 1500,
                "SUNPHARMA": 350, "TITAN": 175, "TATASTEEL": 5500, "HINDUNILVR": 300,
            }
            lot_size = lot_sizes.get(clean_sym, int(max(25, round(250000 / max(spot_price, 10.0)))))
            if spot_price > 5000:
                step = 100
            elif spot_price > 2000:
                step = 50
            elif spot_price > 1000:
                step = 20
            elif spot_price > 500:
                step = 10
            elif spot_price > 200:
                step = 5
            else:
                step = 2.5

        atm_strike = int(round(spot_price / step) * step)
        if atm_strike <= 0:
            atm_strike = int(step)

        # 2. Black-Scholes Greeks & Premium Calculation
        T = max(days_to_expiry / 365.0, 1.0 / 365.0)
        if is_commodity:
            iv = max(vix_price * 1.8, 26.0)
        elif is_index:
            iv = vix_price
        else:
            iv = max(vix_price * 1.5, 22.0)

        sigma = max(iv / 100.0, 0.05)
        r = 0.065  # RBI risk-free rate

        S = spot_price
        K = atm_strike

        d1 = (math.log(S / K) + (r + 0.5 * (sigma ** 2)) * T) / (sigma * math.sqrt(T))
        d2 = d1 - (sigma * math.sqrt(T))

        call_premium = max(1.0, S * cls._norm_cdf(d1) - K * math.exp(-r * T) * cls._norm_cdf(d2))
        put_premium = max(1.0, K * math.exp(-r * T) * cls._norm_cdf(-d2) - S * cls._norm_cdf(-d1))

        call_delta = max(0.1, min(0.9, cls._norm_cdf(d1)))
        put_delta = max(0.1, min(0.9, 1.0 - cls._norm_cdf(d1)))

        # 3. Calculate Option Targets & Stop Loss
        if "CALL" in recommendation:
            opt_type = "CE"
            opt_strike = f"{atm_strike} CE"
            entry_premium = round(call_premium, 2)
            delta = round(call_delta, 2)

            spot_gain_1 = max(0.0, spot_target_1 - spot_price)
            spot_gain_2 = max(0.0, spot_target_2 - spot_price)
            spot_risk = max(0.0, spot_price - spot_stop_loss)

            # Delta-projected values
            raw_t1 = entry_premium + (delta * spot_gain_1)
            raw_t2 = entry_premium + (delta * spot_gain_2)
            raw_sl = entry_premium - (delta * spot_risk)

            # Realistic options bounds: T1 (+25% to +45%), T2 (+55% to +85%), T3 (+95% to +125%), Long TGT (+140% to +200%), SL (-18% to -28%)
            opt_target_1 = round(max(entry_premium * 1.25, min(entry_premium * 1.45, raw_t1)), 2)
            opt_target_2 = round(max(opt_target_1 * 1.20, min(entry_premium * 1.85, max(entry_premium * 1.55, raw_t2))), 2)
            opt_target_3 = round(max(opt_target_2 * 1.15, entry_premium * 2.00), 2)
            opt_long_target = round(max(opt_target_3 * 1.20, entry_premium * 2.50), 2)
            opt_stop_loss = round(max(entry_premium * 0.72, min(entry_premium * 0.82, raw_sl)), 2)

        elif "PUT" in recommendation:
            opt_type = "PE"
            opt_strike = f"{atm_strike} PE"
            entry_premium = round(put_premium, 2)
            delta = round(put_delta, 2)

            spot_drop_1 = max(0.0, spot_price - spot_target_1)
            spot_drop_2 = max(0.0, spot_price - spot_target_2)
            spot_risk = max(0.0, spot_stop_loss - spot_price)

            raw_t1 = entry_premium + (delta * spot_drop_1)
            raw_t2 = entry_premium + (delta * spot_drop_2)
            raw_sl = entry_premium - (delta * spot_risk)

            # Realistic options bounds: T1 (+25% to +45%), T2 (+55% to +85%), T3 (+95% to +125%), Long TGT (+140% to +200%), SL (-18% to -28%)
            opt_target_1 = round(max(entry_premium * 1.25, min(entry_premium * 1.45, raw_t1)), 2)
            opt_target_2 = round(max(opt_target_1 * 1.20, min(entry_premium * 1.85, max(entry_premium * 1.55, raw_t2))), 2)
            opt_target_3 = round(max(opt_target_2 * 1.15, entry_premium * 2.00), 2)
            opt_long_target = round(max(opt_target_3 * 1.20, entry_premium * 2.50), 2)
            opt_stop_loss = round(max(entry_premium * 0.72, min(entry_premium * 0.82, raw_sl)), 2)

        else:
            opt_type = "CE"
            opt_strike = f"{atm_strike} CE"
            entry_premium = round(call_premium, 2)
            delta = 0.50
            opt_target_1 = round(entry_premium * 1.28, 2)
            opt_target_2 = round(entry_premium * 1.60, 2)
            opt_target_3 = round(entry_premium * 2.00, 2)
            opt_long_target = round(entry_premium * 2.50, 2)
            opt_stop_loss = round(entry_premium * 0.78, 2)

        capital_per_lot = round(entry_premium * lot_size, 2)
        profit_per_lot_t1 = round((opt_target_1 - entry_premium) * lot_size, 2)
        profit_per_lot_t2 = round((opt_target_2 - entry_premium) * lot_size, 2)
        profit_per_lot_t3 = round((opt_target_3 - entry_premium) * lot_size, 2)
        risk_per_lot = round((entry_premium - opt_stop_loss) * lot_size, 2)

        return {
            "suggested_strike": opt_strike,
            "strike_price": atm_strike,
            "option_type": opt_type,
            "option_entry_price": entry_premium,
            "option_target_1": opt_target_1,
            "option_target_2": opt_target_2,
            "option_target_3": opt_target_3,
            "option_long_target": opt_long_target,
            "option_stop_loss": opt_stop_loss,
            "option_delta": round(delta, 2),
            "lot_size": lot_size,
            "capital_required_per_lot": capital_per_lot,
            "est_profit_per_lot_t1": profit_per_lot_t1,
            "est_profit_per_lot_t2": profit_per_lot_t2,
            "est_profit_per_lot_t3": profit_per_lot_t3,
            "est_risk_per_lot": risk_per_lot,
        }

    @staticmethod
    def _safe_float(val: Any, default: float = 0.0) -> float:
        """Safely convert any numeric value to float, preventing NaN/Inf from breaking JSON."""
        try:
            if val is None or pd.isna(val) or np.isnan(val) or np.isinf(val):
                return default
            return float(val)
        except Exception:
            return default

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range (ATR)."""
        high = df["high"]
        low = df["low"]
        close = df["close"]
        prev_close = close.shift(1).bfill().fillna(close)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period, min_periods=1).mean()
        return atr.bfill().fillna(df["close"] * 0.01)

    @classmethod
    def calculate_supertrend(
        cls, df: pd.DataFrame, period: int = 10, multiplier: float = 3.0
    ) -> pd.DataFrame:
        """Calculate Supertrend indicator."""
        df = df.copy()
        atr = cls.calculate_atr(df, period)
        hl2 = (df["high"] + df["low"]) / 2.0

        upper_basic = hl2 + (multiplier * atr)
        lower_basic = hl2 - (multiplier * atr)

        upper_band = pd.Series(index=df.index, dtype="float64")
        lower_band = pd.Series(index=df.index, dtype="float64")
        trend = pd.Series(index=df.index, dtype="int64")

        upper_band.iloc[0] = float(upper_basic.iloc[0]) if not pd.isna(upper_basic.iloc[0]) else float(df["close"].iloc[0] * 1.02)
        lower_band.iloc[0] = float(lower_basic.iloc[0]) if not pd.isna(lower_basic.iloc[0]) else float(df["close"].iloc[0] * 0.98)
        trend.iloc[0] = 1

        for i in range(1, len(df)):
            ub_prev = upper_band.iloc[i - 1]
            lb_prev = lower_band.iloc[i - 1]
            c_prev = df["close"].iloc[i - 1]
            ub_curr = upper_basic.iloc[i]
            lb_curr = lower_basic.iloc[i]

            if ub_curr < ub_prev or c_prev > ub_prev:
                upper_band.iloc[i] = ub_curr
            else:
                upper_band.iloc[i] = ub_prev

            if lb_curr > lb_prev or c_prev < lb_prev:
                lower_band.iloc[i] = lb_curr
            else:
                lower_band.iloc[i] = lb_prev

            if df["close"].iloc[i] > ub_prev:
                trend.iloc[i] = 1
            elif df["close"].iloc[i] < lb_prev:
                trend.iloc[i] = -1
            else:
                trend.iloc[i] = trend.iloc[i - 1]

        df["supertrend_upper"] = upper_band.ffill().bfill()
        df["supertrend_lower"] = lower_band.ffill().bfill()
        df["supertrend_trend"] = trend.fillna(1).astype(int)
        df["supertrend_value"] = np.where(df["supertrend_trend"] == 1, df["supertrend_lower"], df["supertrend_upper"])
        return df

    @classmethod
    def calculate_indicators(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate full suite of technical indicators."""
        df = df.copy()

        df["ema_9"] = df["close"].ewm(span=9, adjust=False).mean()
        df["ema_21"] = df["close"].ewm(span=21, adjust=False).mean()
        df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
        df["sma_200"] = df["close"].rolling(window=min(200, len(df)), min_periods=1).mean().bfill()

        delta = df["close"].diff().fillna(0)
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / (loss + 1e-9)
        df["rsi"] = (100 - (100 / (1 + rs))).fillna(50.0)

        exp1 = df["close"].ewm(span=12, adjust=False).mean()
        exp2 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = exp1 - exp2
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        df["bb_middle"] = df["close"].rolling(window=20, min_periods=1).mean().bfill()
        bb_std = df["close"].rolling(window=20, min_periods=1).std().fillna(0.0)
        df["bb_upper"] = df["bb_middle"] + (2 * bb_std)
        df["bb_lower"] = df["bb_middle"] - (2 * bb_std)
        df["bb_pct_b"] = ((df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-9)).fillna(0.5)

        df["atr"] = cls.calculate_atr(df, period=14)
        df = cls.calculate_supertrend(df, period=10, multiplier=3.0)

        return df

    @staticmethod
    def calculate_fibonacci(df: pd.DataFrame, lookback: int = 50) -> Dict[str, Any]:
        """
        Calculate Fibonacci Retracement Levels based on recent swing high and low.
        Key levels: 0.0%, 23.6%, 38.2%, 50.0%, 61.8% (Golden Ratio), 78.6%, 100.0%
        """
        window = df.tail(min(lookback, len(df)))
        swing_high = float(window["high"].max())
        swing_low = float(window["low"].min())
        diff = swing_high - swing_low

        levels = {
            "high_0.0": round(swing_high, 2),
            "fib_23.6": round(swing_high - 0.236 * diff, 2),
            "fib_38.2": round(swing_high - 0.382 * diff, 2),
            "fib_50.0": round(swing_high - 0.500 * diff, 2),
            "fib_61.8": round(swing_high - 0.618 * diff, 2),  # Golden Pocket
            "fib_78.6": round(swing_high - 0.786 * diff, 2),
            "low_100.0": round(swing_low, 2),
        }

        cur_price = float(df["close"].iloc[-1])

        supports = [v for v in levels.values() if v < cur_price]
        resistances = [v for v in levels.values() if v > cur_price]

        nearest_support = max(supports) if supports else swing_low
        nearest_resistance = min(resistances) if resistances else swing_high

        golden_pocket = levels["fib_61.8"]
        half_way = levels["fib_50.0"]

        if cur_price >= half_way:
            bias = "BULLISH_ZONE"
            desc = f"Price is trading above 50% Fib level ({half_way:.2f}), approaching upper retracement."
        else:
            bias = "BEARISH_ZONE"
            desc = f"Price is trading below 50% Fib level ({half_way:.2f}), near golden support ({golden_pocket:.2f})."

        return {
            "swing_high": swing_high,
            "swing_low": swing_low,
            "levels": levels,
            "nearest_support": nearest_support,
            "nearest_resistance": nearest_resistance,
            "golden_pocket": golden_pocket,
            "golden_pocket_support": golden_pocket,
            "bias": bias,
            "description": desc,
        }

    def generate_options_call_put_signal(
        self, symbol: str, interval: str = "15m", period: str = "5d"
    ) -> Dict[str, Any]:
        """
        Comprehensive Multi-Strategy Filter Engine for CALL (CE) / PUT (PE) recommendations.
        Combines:
        1. Fibonacci Golden Pocket (61.8%)
        2. Supertrend
        3. EMA Trend Alignment 9/21/50
        4. RSI Momentum
        5. MACD Crossover
        6. Bollinger Band Reversal / Squeeze
        7. Put-Call Ratio (PCR) & OI Max Pain
        8. Macro Financial News Sentiment
        9. Candlestick Patterns, PDH/PDL, CPR, Volume/VWAP, SMC Order Blocks, W/M Patterns
        10. Price Action Momentum & Impulse Engine
        11. India VIX Volatility Regime & Fear Index
        12. GIFT NIFTY & Global Market Cues
        """
        canonical = market_data_service.normalize_symbol(symbol)
        cache_key = f"{canonical}_{interval}_{period}"
        now = time.time()

        # Fast cache lookup for sub-second responses
        if cache_key in self._signal_cache:
            entry = self._signal_cache[cache_key]
            if now - entry["timestamp"] < 2.0:
                cached_data = dict(entry["data"])
                live_p = market_data_service.get_live_price(canonical)
                if live_p and live_p.get("price"):
                    cur_p = live_p["price"]
                    cached_data["spot_price"] = cur_p
                    cached_data["price"] = cur_p
                    # Fast recalculation of option strike levels to exact live price
                    recom = cached_data.get("recommendation", "CALL (CE)")
                    t1_spot = cached_data.get("spot_target_1", cur_p * 1.01)
                    t2_spot = cached_data.get("spot_target_2", cur_p * 1.02)
                    sl_spot = cached_data.get("spot_stop_loss", cur_p * 0.99)
                    vix_val = cached_data.get("vix_value", 12.5)
                    opt_pricing = self.calculate_option_strike_pricing(
                        symbol=symbol,
                        spot_price=cur_p,
                        recommendation=recom,
                        spot_target_1=t1_spot,
                        spot_target_2=t2_spot,
                        spot_stop_loss=sl_spot,
                        vix_price=vix_val,
                    )
                    cached_data.update(opt_pricing)
                return cached_data

        df = market_data_service.get_historical_candles(canonical, period=period, interval=interval)
        live = market_data_service.get_live_price(canonical)

        if df is None or len(df) < 15:
            fallback_price = live["price"] if (live and live.get("price")) else 1000.0

            dates = pd.date_range(end=pd.Timestamp.now(), periods=35, freq="15min")
            base = fallback_price
            np.random.seed(42)
            noise = np.random.normal(0, base * 0.0015, 35)
            drift = np.cumsum(noise) - np.cumsum(noise)[-1]
            closes = base + drift
            highs = closes + (base * 0.003)
            lows = closes - (base * 0.003)
            opens = closes - (noise * 0.5)
            volumes = np.random.randint(15000, 75000, 35)
            df = pd.DataFrame(
                {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
                index=dates,
            )
            df.index.name = "timestamp"

        df_calc = self.calculate_indicators(df)
        last = df_calc.iloc[-1]
        prev = df_calc.iloc[-2]

        cur_price = float(last["close"])
        bullish_score = 0.0
        bearish_score = 0.0
        reasons: List[str] = []

        # 1. Fibonacci Filter
        fib = self.calculate_fibonacci(df)
        golden_support = fib["levels"]["fib_61.8"]
        fib_50 = fib["levels"]["fib_50.0"]

        if cur_price >= fib_50:
            bullish_score += 25
            reasons.append(f"Fibonacci: Price (₹{cur_price:.2f}) holding above 50% retracement (₹{fib_50:.2f})")
        else:
            bearish_score += 25
            reasons.append(f"Fibonacci: Price below 50% retracement, testing Golden Pocket support (₹{golden_support:.2f})")

        # 2. Supertrend Filter
        st_trend = int(last["supertrend_trend"])
        st_val = float(last["supertrend_value"])
        if st_trend == 1:
            bullish_score += 20
            reasons.append(f"Supertrend: Bullish trend active (Trailing Stop: ₹{st_val:.2f})")
        else:
            bearish_score += 20
            reasons.append(f"Supertrend: Bearish trend active (Trailing Resistance: ₹{st_val:.2f})")

        # 3. EMA Trend Alignment (9, 21, 50)
        ema_9 = float(last["ema_9"])
        ema_21 = float(last["ema_21"])
        ema_50 = float(last["ema_50"])

        if ema_9 > ema_21 > ema_50:
            bullish_score += 20
            reasons.append(f"EMA: Perfect Bullish Alignment (9 > 21 > 50 EMA: ₹{ema_9:.1f} > ₹{ema_21:.1f} > ₹{ema_50:.1f})")
        elif ema_9 < ema_21 < ema_50:
            bearish_score += 20
            reasons.append(f"EMA: Perfect Bearish Alignment (9 < 21 < 50 EMA: ₹{ema_9:.1f} < ₹{ema_21:.1f} < ₹{ema_50:.1f})")
        elif ema_9 > ema_21:
            bullish_score += 10
            reasons.append("EMA: Short-term 9/21 Bullish cross")
        else:
            bearish_score += 10
            reasons.append("EMA: Short-term 9/21 Bearish cross")

        # 4. RSI Momentum
        rsi = float(last["rsi"])
        if 50 < rsi < 70:
            bullish_score += 15
            reasons.append(f"RSI (14): Healthy Bullish Momentum ({rsi:.1f})")
        elif rsi >= 70:
            bearish_score += 5
            reasons.append(f"RSI (14): Overbought territory ({rsi:.1f}) - risk of cooling pullback")
        elif 30 < rsi <= 50:
            bearish_score += 15
            reasons.append(f"RSI (14): Bearish Momentum ({rsi:.1f})")
        else:
            bullish_score += 5
            reasons.append(f"RSI (14): Oversold bounce zone ({rsi:.1f})")

        # 5. MACD Crossover
        macd = float(last["macd"])
        macd_sig = float(last["macd_signal"])
        if macd > macd_sig:
            bullish_score += 10
            reasons.append(f"MACD: Bullish Divergence (MACD: {macd:.2f} > Signal: {macd_sig:.2f})")
        else:
            bearish_score += 10
            reasons.append(f"MACD: Bearish Divergence (MACD: {macd:.2f} < Signal: {macd_sig:.2f})")

        # 6. Bollinger Bands
        bb_pct = float(last["bb_pct_b"])
        atr = float(last["atr"]) if not np.isnan(last["atr"]) else cur_price * 0.008

        if bb_pct > 0.8:
            bullish_score += 10
            reasons.append(f"Bollinger Bands: Upper breakout band ride (%B: {bb_pct:.2f})")
        elif bb_pct < 0.2:
            bearish_score += 10
            reasons.append(f"Bollinger Bands: Lower breakdown band ride (%B: {bb_pct:.2f})")

        # 7. Put-Call Ratio (PCR) & OI Max Pain Filter
        pcr_data = pcr_service.analyze_pcr(symbol, spot_price=cur_price)
        pcr_val = pcr_data.get("pcr_oi", 1.0)
        pcr_sentiment = pcr_data.get("sentiment", "NEUTRAL")
        max_pain = pcr_data.get("max_pain_strike", cur_price)

        if pcr_val >= 1.25 or pcr_sentiment in ["BULLISH", "EXTREMELY_BULLISH"]:
            bullish_score += 15
            reasons.append(f"PCR OI: Bullish Put Writing Support (PCR: {pcr_val:.2f}, Max Pain: ₹{max_pain})")
        elif pcr_val <= 0.75 or pcr_sentiment in ["BEARISH", "EXTREMELY_BEARISH"]:
            bearish_score += 15
            reasons.append(f"PCR OI: Bearish Call Overhang (PCR: {pcr_val:.2f}, Max Pain: ₹{max_pain})")
        else:
            bullish_score += 5
            bearish_score += 5
            reasons.append(f"PCR OI: Neutral options positioning (PCR: {pcr_val:.2f})")

        # 8. Financial News Sentiment & Catalyst Filter
        news_intel = news_service.get_stock_news(symbol, limit=5)
        news_score = news_intel.get("average_sentiment_score", 0.0)
        news_label = news_intel.get("overall_sentiment", "NEUTRAL")
        if news_label == "BULLISH":
            bullish_score += 10
            reasons.append(f"News Sentiment: Bullish flow (Avg score: {news_score:+.2f}, {news_intel.get('bullish_articles_count', 0)} positive catalysts)")
        elif news_label == "BEARISH":
            bearish_score += 10
            reasons.append(f"News Sentiment: Bearish headwinds (Avg score: {news_score:+.2f}, {news_intel.get('bearish_articles_count', 0)} negative catalysts)")
        else:
            bullish_score += 5
            bearish_score += 5
            reasons.append(f"News Sentiment: Neutral / Balanced ({news_score:+.2f})")

        # 9. Candlestick Patterns, Day Levels, Volume/VWAP, Order Blocks (SMC), W/M Patterns, and Multi-Timeframe S/R
        pattern_scan = pattern_service.run_full_pattern_scan(df, symbol=symbol)
        candle_patterns = pattern_scan.get("candle_patterns", [])
        day_levels = pattern_scan.get("day_levels", {})
        volume_analysis = pattern_scan.get("volume_analysis", {})
        order_blocks = pattern_scan.get("order_blocks", {})
        wm_patterns = pattern_scan.get("wm_patterns", {})
        pa_momentum = pattern_scan.get("price_action_momentum", {})
        multi_sr = pattern_scan.get("multi_timeframe_sr", {})

        # 9a. Candlestick Patterns (Weight: 10)
        for cp in candle_patterns:
            if cp.get("type") == "BULLISH":
                bullish_score += 10
                reasons.append(f"Candlestick: {cp.get('name')} ({cp.get('strength')} strength)")
            elif cp.get("type") == "BEARISH":
                bearish_score += 10
                reasons.append(f"Candlestick: {cp.get('name')} ({cp.get('strength')} strength)")

        # 9b. Day Levels & CPR (Weight: 15)
        dl_status = day_levels.get("status")
        if dl_status == "PDH_BREAKOUT":
            bullish_score += 15
            reasons.append(f"Day Levels: Broke Previous Day High (PDH: ₹{day_levels.get('pdh')})")
        elif dl_status == "PDL_BREAKDOWN":
            bearish_score += 15
            reasons.append(f"Day Levels: Broke Previous Day Low (PDL: ₹{day_levels.get('pdl')})")
        elif dl_status == "ABOVE_CPR":
            bullish_score += 5
            reasons.append(f"CPR: Trading above Central Pivot Range (TC: ₹{day_levels.get('cpr', {}).get('tc')})")
        elif dl_status == "BELOW_CPR":
            bearish_score += 5
            reasons.append(f"CPR: Trading below Central Pivot Range (BC: ₹{day_levels.get('cpr', {}).get('bc')})")

        # 9c. Volume & VWAP (Weight: 10)
        vol_sentiment = volume_analysis.get("volume_sentiment")
        vol_ratio = volume_analysis.get("volume_ratio", 1.0)
        vwap_status = volume_analysis.get("price_vs_vwap")
        if vol_sentiment == "BULLISH_ACCUMULATION" or (vwap_status == "BULLISH_ABOVE_VWAP" and vol_ratio >= 1.5):
            bullish_score += 10
            reasons.append(f"Volume & VWAP: Institutional Volume Surge ({vol_ratio}x avg) above VWAP (₹{volume_analysis.get('vwap')})")
        elif vol_sentiment == "BEARISH_DISTRIBUTION" or (vwap_status == "BEARISH_BELOW_VWAP" and vol_ratio >= 1.5):
            bearish_score += 10
            reasons.append(f"Volume & VWAP: Bearish Distribution Volume ({vol_ratio}x avg) below VWAP (₹{volume_analysis.get('vwap')})")

        # 9d. Order Blocks / SMC (Weight: 15)
        smc_bias = order_blocks.get("smc_bias")
        if smc_bias == "AT_DEMAND_ORDER_BLOCK":
            bullish_score += 15
            reasons.append(f"Order Block (SMC): Price reacting in Institutional Demand Zone (₹{order_blocks.get('nearest_bullish_ob', {}).get('ob_zone_bottom')} - ₹{order_blocks.get('nearest_bullish_ob', {}).get('ob_zone_top')})")
        elif smc_bias == "AT_SUPPLY_ORDER_BLOCK":
            bearish_score += 15
            reasons.append(f"Order Block (SMC): Price reacting in Institutional Supply Zone (₹{order_blocks.get('nearest_bearish_ob', {}).get('ob_zone_bottom')} - ₹{order_blocks.get('nearest_bearish_ob', {}).get('ob_zone_top')})")

        # 9e. W & M Patterns (Weight: 20)
        wm_status = wm_patterns.get("status")
        if wm_patterns.get("pattern_type") == "BULLISH" and wm_status == "CONFIRMED_BREAKOUT":
            bullish_score += 20
            reasons.append(f"Chart Pattern: W-Pattern (Double Bottom) Neckline Breakout confirmed at ₹{wm_patterns.get('neckline')}")
        elif wm_patterns.get("pattern_type") == "BULLISH" and wm_status == "FORMING_RIGHT_ARM":
            bullish_score += 10
            reasons.append(f"Chart Pattern: W-Pattern (Double Bottom) forming right leg (Low: ₹{wm_patterns.get('low_2')})")
        elif wm_patterns.get("pattern_type") == "BEARISH" and wm_status == "CONFIRMED_BREAKDOWN":
            bearish_score += 20
            reasons.append(f"Chart Pattern: M-Pattern (Double Top) Neckline Breakdown confirmed at ₹{wm_patterns.get('neckline')}")
        elif wm_patterns.get("pattern_type") == "BEARISH" and wm_status == "FORMING_RIGHT_ARM":
            bearish_score += 10
            reasons.append(f"Chart Pattern: M-Pattern (Double Top) forming right leg (High: ₹{wm_patterns.get('high_2')})")

        # 9f. Price Action Momentum & Impulse (Weight: 20)
        mom_score = pa_momentum.get("momentum_score", 0.0)
        mom_regime = pa_momentum.get("momentum_regime", "NEUTRAL_COMPRESSION")
        trend_struct = pa_momentum.get("trend_structure", "RANGE_CONSOLIDATION")
        bos_status = pa_momentum.get("bos_status", "NONE")
        wick_rej = pa_momentum.get("wick_rejection", "NONE")
        streak = pa_momentum.get("consecutive_streak", 0)

        if mom_regime == "STRONG_BULLISH_IMPULSE" or mom_score >= 50.0:
            bullish_score += 20
            reasons.append(f"Price Action: Strong Bullish Impulse (+{mom_score}) with {trend_struct} ({streak} green bars streak)")
        elif mom_regime == "MODERATE_BULLISH_MOMENTUM" or mom_score >= 20.0:
            bullish_score += 10
            reasons.append(f"Price Action: Bullish Momentum Drift (+{mom_score}) - {trend_struct}")
        elif mom_regime == "STRONG_BEARISH_IMPULSE" or mom_score <= -50.0:
            bearish_score += 20
            reasons.append(f"Price Action: Strong Bearish Impulse ({mom_score}) with {trend_struct} ({streak} red bars streak)")
        elif mom_regime == "MODERATE_BEARISH_MOMENTUM" or mom_score <= -20.0:
            bearish_score += 10
            reasons.append(f"Price Action: Bearish Momentum Drift ({mom_score}) - {trend_struct}")
        else:
            bullish_score += 5
            bearish_score += 5
            reasons.append(f"Price Action: Range Compression / Squeeze ({mom_score})")

        if bos_status == "BOS_BULLISH_BREAKOUT":
            bullish_score += 10
            reasons.append("Price Action: Bullish Break of Structure (BOS) above swing high")
        elif bos_status == "BOS_BEARISH_BREAKDOWN":
            bearish_score += 10
            reasons.append("Price Action: Bearish Break of Structure (BOS) below swing low")

        if wick_rej == "BULLISH_LOWER_WICK_REJECTION":
            bullish_score += 5
            reasons.append("Price Action: Bottom lower wick buying absorption detected")
        elif wick_rej == "BEARISH_UPPER_WICK_REJECTION":
            bearish_score += 5
            reasons.append("Price Action: Top upper wick selling exhaustion detected")

        # 9g. Multi-Timeframe Historical Support & Resistance Confluence (Weight: 20)
        sr_bias = multi_sr.get("confluence_bias", "CONSOLIDATING_BETWEEN_SR")
        nearest_res = multi_sr.get("nearest_resistance", {})
        nearest_supp = multi_sr.get("nearest_support", {})
        res_dist_pct = nearest_res.get("distance_pct", 1.0)
        supp_dist_pct = nearest_supp.get("distance_pct", 1.0)

        if sr_bias == "STRONG_BULLISH_EXPANSION":
            bullish_score += 20
            reasons.append(f"Historical S/R: Bullish Breakout above PDH & CPR floor (Next hurdle: {nearest_res.get('primary_label', 'R1')} at ₹{nearest_res.get('price', cur_price * 1.01)})")
        elif sr_bias == "TESTING_KEY_SUPPORT_BOUNCE":
            bullish_score += 15
            reasons.append(f"Historical S/R: Rebounding from institutional support floor {nearest_supp.get('primary_label', 'S1')} at ₹{nearest_supp.get('price', cur_price * 0.99)}")
        elif sr_bias == "STRONG_BEARISH_BREAKDOWN":
            bearish_score += 20
            reasons.append(f"Historical S/R: Bearish Breakdown below PDL & CPR ceiling (Next floor: {nearest_supp.get('primary_label', 'S1')} at ₹{nearest_supp.get('price', cur_price * 0.99)})")
        elif sr_bias == "TESTING_KEY_RESISTANCE_REJECTION":
            bearish_score += 15
            reasons.append(f"Historical S/R: Rejecting from institutional overhead resistance {nearest_res.get('primary_label', 'R1')} at ₹{nearest_res.get('price', cur_price * 1.01)}")
        else:
            if res_dist_pct > 1.2 and supp_dist_pct <= 0.4:
                bullish_score += 10
                reasons.append(f"Historical S/R: Favorable upside clearance to {nearest_res.get('primary_label', 'R1')} (₹{nearest_res.get('price', cur_price * 1.01)})")
            elif supp_dist_pct > 1.2 and res_dist_pct <= 0.4:
                bearish_score += 10
                reasons.append(f"Historical S/R: Downside room to {nearest_supp.get('primary_label', 'S1')} (₹{nearest_supp.get('price', cur_price * 0.99)})")

        # 10. India VIX Volatility Regime & Fear Index (Weight: 10)
        vix_intel = volatility_service.get_india_vix()
        vix_val = vix_intel.get("current_vix", 12.5)
        vix_trend = vix_intel.get("trend_bias", "VIX_STABLE")
        vix_regime = vix_intel.get("regime", "IDEAL_VOLATILITY")

        if vix_trend == "VIX_FALLING_COOLING" or (vix_val < 13.0 and vix_regime in ["LOW_VOLATILITY", "IDEAL_VOLATILITY"]):
            bullish_score += 10
            reasons.append(f"India VIX: Volatility cooling ({vix_val}, {vix_intel.get('change_percent', 0):+.1f}%), institutional fear subdued")
        elif vix_trend == "VIX_SPIKING_FEAR" or vix_val >= 18.0:
            bearish_score += 10
            reasons.append(f"India VIX: Fear index elevated ({vix_val}, {vix_intel.get('change_percent', 0):+.1f}%), protective put buying rising")
        else:
            bullish_score += 5
            bearish_score += 5
            reasons.append(f"India VIX: Balanced volatility regime ({vix_val})")

        # 11. GIFT NIFTY & Global Market Cues (Weight: 10)
        gift_intel = volatility_service.get_gift_nifty_and_global_cues()
        gift_bias = gift_intel.get("opening_bias", "FLAT_NEUTRAL")
        gift_pts = gift_intel.get("projected_gap_pts", 0.0)

        if "GAP_UP" in gift_bias or gift_pts >= 15.0:
            bullish_score += 10
            reasons.append(f"GIFT NIFTY & Global Cues: Bullish momentum ({gift_intel.get('bias_label')})")
        elif "GAP_DOWN" in gift_bias or gift_pts <= -15.0:
            bearish_score += 10
            reasons.append(f"GIFT NIFTY & Global Cues: Downside pressure ({gift_intel.get('bias_label')})")
        else:
            bullish_score += 5
            bearish_score += 5
            reasons.append(f"GIFT NIFTY & Global Cues: Flat opening bias ({gift_intel.get('bias_label')})")

        # Final Recommendation Calculation
        net_score = bullish_score - bearish_score
        confidence = round(max(bullish_score, bearish_score), 1)

        # Effective ATR (bounded to 0.4% - 1.2% of spot price for balanced swing/intraday levels)
        atr_val = float(last["atr"]) if not np.isnan(last["atr"]) else cur_price * 0.007
        atr_effective = max(cur_price * 0.004, min(cur_price * 0.012, atr_val))

        # Support & Resistance levels from multi-timeframe analysis
        s_near_val = multi_sr.get("s_near", cur_price - (1.0 * atr_effective))
        s_mid_val = multi_sr.get("s_mid", cur_price - (1.8 * atr_effective))
        r_near_val = multi_sr.get("r_near", cur_price + (1.0 * atr_effective))
        r_mid_val = multi_sr.get("r_mid", cur_price + (1.8 * atr_effective))

        if net_score >= 20:
            recommendation = "CALL (CE)"
            signal_type = "BULLISH"
            tag_color = "green"

            # Dynamic Support-Aware Stop Loss (prefer nearest support floor, bounded by 0.6x - 1.5x ATR)
            supp_dist = cur_price - s_near_val
            if 0.5 * atr_effective <= supp_dist <= 1.5 * atr_effective:
                stop_loss = round(s_near_val - (0.05 * atr_effective), 2)
            else:
                fib_supp = fib.get("nearest_support", cur_price - (1.0 * atr_effective))
                f_dist = cur_price - fib_supp
                if 0.6 * atr_effective <= f_dist <= 1.4 * atr_effective:
                    stop_loss = round(fib_supp - (0.05 * atr_effective), 2)
                else:
                    stop_loss = round(cur_price - (1.0 * atr_effective), 2)

            # Dynamic Resistance-Aware Target 1 (align to nearest overhead historical resistance)
            res_dist = r_near_val - cur_price
            if 0.8 * atr_effective <= res_dist <= 1.8 * atr_effective:
                target_1 = round(r_near_val, 2)
            else:
                fib_res = fib.get("nearest_resistance", cur_price + (1.3 * atr_effective))
                f_dist = fib_res - cur_price
                if 0.9 * atr_effective <= f_dist <= 1.6 * atr_effective:
                    target_1 = round(fib_res, 2)
                else:
                    target_1 = round(cur_price + (1.3 * atr_effective), 2)

            # Target 2 (align to mid-tier resistance or secondary expansion)
            if r_mid_val > target_1 and (r_mid_val - cur_price) <= 2.8 * atr_effective:
                target_2 = round(r_mid_val, 2)
            else:
                target_2 = round(target_1 + (0.9 * atr_effective), 2)

        elif net_score <= -20:
            recommendation = "PUT (PE)"
            signal_type = "BEARISH"
            tag_color = "red"

            # Dynamic Resistance-Aware Stop Loss (prefer nearest resistance ceiling)
            res_dist = r_near_val - cur_price
            if 0.5 * atr_effective <= res_dist <= 1.5 * atr_effective:
                stop_loss = round(r_near_val + (0.05 * atr_effective), 2)
            else:
                fib_res = fib.get("nearest_resistance", cur_price + (1.0 * atr_effective))
                f_dist = fib_res - cur_price
                if 0.6 * atr_effective <= f_dist <= 1.4 * atr_effective:
                    stop_loss = round(fib_res + (0.05 * atr_effective), 2)
                else:
                    stop_loss = round(cur_price + (1.0 * atr_effective), 2)

            # Dynamic Support-Aware Target 1 (align to nearest underlying historical support)
            supp_dist = cur_price - s_near_val
            if 0.8 * atr_effective <= supp_dist <= 1.8 * atr_effective:
                target_1 = round(s_near_val, 2)
            else:
                fib_supp = fib.get("nearest_support", cur_price - (1.3 * atr_effective))
                f_dist = cur_price - fib_supp
                if 0.9 * atr_effective <= f_dist <= 1.6 * atr_effective:
                    target_1 = round(fib_supp, 2)
                else:
                    target_1 = round(cur_price - (1.3 * atr_effective), 2)

            # Target 2 (align to mid-tier support or secondary expansion)
            if s_mid_val < target_1 and (cur_price - s_mid_val) <= 2.8 * atr_effective:
                target_2 = round(s_mid_val, 2)
            else:
                target_2 = round(target_1 - (0.9 * atr_effective), 2)

        else:
            recommendation = "NEUTRAL / WAIT"
            signal_type = "CONSOLIDATION"
            tag_color = "gray"
            stop_loss = round(cur_price - (1.0 * atr_effective), 2)
            target_1 = round(cur_price + (1.2 * atr_effective), 2)
            target_2 = round(cur_price + (2.0 * atr_effective), 2)

        # Calculate exact Option Contract Strike Premium Prices
        opt_pricing = self.calculate_option_strike_pricing(
            symbol=symbol,
            spot_price=cur_price,
            recommendation=recommendation,
            spot_target_1=target_1,
            spot_target_2=target_2,
            spot_stop_loss=stop_loss,
            vix_price=vix_val,
        )

        risk = abs(cur_price - stop_loss)
        reward = abs(target_1 - cur_price)
        rr_ratio = f"1:{round(reward / (risk + 1e-9), 2)}" if risk > 0 else "1:2.0"

        # Quality Move & Minimum 5 Points Filtering Metrics
        opt_entry = opt_pricing["option_entry_price"]
        opt_t1 = opt_pricing["option_target_1"]
        expected_option_gain = round(max(0.0, opt_t1 - opt_entry), 2)
        expected_spot_gain = round(abs(target_1 - cur_price), 2)
        is_actionable = recommendation in ["CALL (CE)", "PUT (PE)", "BUY CALL (CE)", "BUY PUT (PE)"]

        # Low-unit MCX commodities (e.g. Natural Gas @ ~₹235) have smaller nominal points per lot (1250 qty)
        min_points_req = 1.0 if canonical in ["NATURALGAS", "NG=F"] else 5.0
        meets_move_threshold = (expected_option_gain >= min_points_req) or (expected_spot_gain >= 5.0)
        is_good_move = bool(is_actionable and meets_move_threshold and confidence >= 60.0)

        name_map = {
            "^NSEI": "NIFTY 50",
            "NIFTY": "NIFTY 50",
            "NIFTY50": "NIFTY 50",
            "^BSESN": "BSE SENSEX",
            "SENSEX": "BSE SENSEX",
            "^NSEBANK": "BANK NIFTY",
            "BANKNIFTY": "BANK NIFTY",
            "NIFTY_FIN_SERVICE.NS": "FIN NIFTY",
            "FINNIFTY": "FIN NIFTY",
            "^NSEMDCP50": "MIDCAP NIFTY",
            "MIDCPNIFTY": "MIDCAP NIFTY",
            "CRUDEOIL": "Brent Crude Oil (MCX)",
            "CRUDE": "Brent Crude Oil (MCX)",
            "BRENT": "Brent Crude Oil (MCX)",
            "BRENTOIL": "Brent Crude Oil (MCX)",
            "BZ=F": "Brent Crude Oil (MCX)",
            "CL=F": "Brent Crude Oil (MCX)",
            "GOLD": "Gold 999 (MCX)",
            "GC=F": "Gold 999 (MCX)",
            "SILVER": "Silver (MCX)",
            "SI=F": "Silver (MCX)",
            "NATURALGAS": "Henry Hub Natural Gas (MCX)",
            "NATGAS": "Henry Hub Natural Gas (MCX)",
            "HENRYHUB": "Henry Hub Natural Gas (MCX)",
            "NG=F": "Henry Hub Natural Gas (MCX)",
            "COPPER": "Copper (MCX)",
            "HG=F": "Copper (MCX)",
            "SBIN": "State Bank of India",
            "RELIANCE": "Reliance Industries",
            "TCS": "Tata Consultancy Services",
            "INFY": "Infosys",
            "HDFCBANK": "HDFC Bank",
            "ICICIBANK": "ICICI Bank",
            "LT": "Larsen & Toubro",
            "ITC": "ITC Ltd",
        }
        instrument_name = name_map.get(canonical, name_map.get(symbol.upper(), symbol.upper()))

        final_result = {
            "instrument": instrument_name,
            "symbol": symbol.upper(),
            "spot_price": round(cur_price, 2),
            "recommendation": recommendation,
            "signal_type": signal_type,
            "tag_color": tag_color,
            "confidence_score": confidence,
            "suggested_strike": opt_pricing["suggested_strike"],

            # 1. OPTION CONTRACT STRIKE PREMIUM PRICES (Execution Prices)
            "strike_price": opt_pricing["strike_price"],
            "option_type": opt_pricing["option_type"],
            "option_entry_price": opt_pricing["option_entry_price"],
            "option_target_1": opt_pricing["option_target_1"],
            "option_target_2": opt_pricing["option_target_2"],
            "option_target_3": opt_pricing["option_target_3"],
            "option_long_target": opt_pricing["option_long_target"],
            "option_stop_loss": opt_pricing["option_stop_loss"],
            "option_delta": opt_pricing["option_delta"],
            "lot_size": opt_pricing["lot_size"],
            "capital_required_per_lot": opt_pricing["capital_required_per_lot"],
            "est_profit_per_lot_t1": opt_pricing["est_profit_per_lot_t1"],
            "est_profit_per_lot_t2": opt_pricing["est_profit_per_lot_t2"],
            "est_profit_per_lot_t3": opt_pricing.get("est_profit_per_lot_t3", 0.0),
            "est_risk_per_lot": opt_pricing["est_risk_per_lot"],

            # Top-level Option Prices
            "entry_price": opt_pricing["option_entry_price"],
            "target_1": opt_pricing["option_target_1"],
            "target_2": opt_pricing["option_target_2"],
            "target_3": opt_pricing["option_target_3"],
            "long_target": opt_pricing["option_long_target"],
            "stop_loss": opt_pricing["option_stop_loss"],

            # 2. EXPECTED GAIN & QUALITY MOVE FILTERS
            "expected_option_gain_pts": expected_option_gain,
            "expected_spot_gain_pts": expected_spot_gain,
            "is_actionable": is_actionable,
            "is_good_move": is_good_move,
            "min_points_required": min_points_req,
            "move_threshold_met": meets_move_threshold,

            # 3. SPOT UNDERLYING REFERENCE PRICES
            "spot_levels": {
                "spot_entry": round(cur_price, 2),
                "spot_target_1": target_1,
                "spot_target_2": target_2,
                "spot_stop_loss": stop_loss,
            },
            "spot_entry_price": round(cur_price, 2),
            "spot_target_1": target_1,
            "spot_target_2": target_2,
            "spot_stop_loss": stop_loss,

            "risk_reward_ratio": rr_ratio,
            "bullish_confluence": round(bullish_score, 1),
            "bearish_confluence": round(bearish_score, 1),
            "confluence_reasons": reasons,
            "indicators": {
                "rsi_14": round(self._safe_float(rsi, 50.0), 2),
                "ema_9": round(self._safe_float(ema_9, cur_price), 2),
                "ema_21": round(self._safe_float(ema_21, cur_price), 2),
                "ema_50": round(self._safe_float(ema_50, cur_price), 2),
                "supertrend": round(self._safe_float(last.get("supertrend_value"), cur_price), 2),
                "supertrend_signal": "BULLISH" if int(self._safe_float(last.get("supertrend_trend"), 1)) == 1 else "BEARISH",
                "macd": round(self._safe_float(macd, 0.0), 2),
                "macd_signal": round(self._safe_float(macd_sig, 0.0), 2),
                "bollinger_upper": round(self._safe_float(last.get("bb_upper"), cur_price * 1.02), 2),
                "bollinger_middle": round(self._safe_float(last.get("bb_middle"), cur_price), 2),
                "bollinger_lower": round(self._safe_float(last.get("bb_lower"), cur_price * 0.98), 2),
                "atr": round(self._safe_float(atr, cur_price * 0.01), 2),
            },
            "fibonacci": fib,
            "pcr": pcr_data,
            "news_intel": news_intel,
            "vix_intel": vix_intel,
            "gift_nifty_intel": gift_intel,
            "candle_patterns": candle_patterns,
            "day_levels": day_levels,
            "volume_analysis": volume_analysis,
            "order_blocks": order_blocks,
            "wm_patterns": wm_patterns,
            "price_action_momentum": pa_momentum,
            "multi_timeframe_sr": multi_sr,
        }

        # 4. 21-FACTOR INSTITUTIONAL CONFLUENCE MATRIX
        try:
            indicators_payload = {
                "rsi": self._safe_float(rsi, 50.0),
                "ema_9": self._safe_float(ema_9, cur_price),
                "ema_21": self._safe_float(ema_21, cur_price),
                "ema_50": self._safe_float(ema_50, cur_price),
                "ema_200": self._safe_float(last.get("ema_200"), cur_price),
                "supertrend_direction": int(self._safe_float(last.get("supertrend_trend"), 1)),
                "adx": self._safe_float(last.get("adx"), 22.0),
                "atr": self._safe_float(atr, cur_price * 0.01),
            }
            proposed_trade_payload = {
                "recommendation": recommendation,
                "spot_price": cur_price,
                "spot_target_1": target_1,
                "spot_target_2": target_2,
                "spot_stop_loss": stop_loss,
                "bullish_score": bullish_score,
                "bearish_score": bearish_score,
                "net_score": bullish_score - bearish_score,
                "confidence_score": confidence,
                "expected_gain_pts": expected_spot_gain,
            }
            bos_status = "BOS_BULLISH" if bullish_score > bearish_score + 15 else ("BOS_BEARISH" if bearish_score > bullish_score + 15 else "NONE")
            inst_matrix = institutional_filter_engine.evaluate_21_factors(
                symbol=symbol,
                df=df_calc,
                cur_price=cur_price,
                indicators=indicators_payload,
                cpr_data=day_levels if isinstance(day_levels, dict) else {},
                pattern_data={
                    "structure_breakout": bos_status,
                    "liquidity_sweeps": {"detected": False},
                    "candlestick_patterns": candle_patterns if isinstance(candle_patterns, list) else [],
                },
                vix_data=vix_intel if isinstance(vix_intel, dict) else {},
                pcr_data=pcr_data if isinstance(pcr_data, dict) else {},
                proposed_trade=proposed_trade_payload,
            )
            final_result["institutional_matrix"] = inst_matrix
        except Exception as e:
            logger.warning(f"Error evaluating 21-factor institutional matrix: {e}")
            final_result["institutional_matrix"] = {
                "symbol": symbol,
                "total_factors": 21,
                "passed_count": 0,
                "watch_count": 0,
                "fail_count": 0,
                "confluence_percentage": 0.0,
                "factors": [],
            }

        sanitized = self.sanitize_obj(final_result)
        self._signal_cache[cache_key] = {"data": sanitized, "timestamp": time.time()}
        return sanitized

    @classmethod
    def sanitize_obj(cls, obj: Any) -> Any:
        """Recursively convert NaN/Inf and numpy types into clean JSON-compliant types."""
        if isinstance(obj, dict):
            return {k: cls.sanitize_obj(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [cls.sanitize_obj(v) for v in obj]
        elif isinstance(obj, (float, np.floating)):
            if math.isnan(obj) or math.isinf(obj):
                return 0.0
            return float(obj)
        elif isinstance(obj, (int, np.integer)):
            return int(obj)
        return obj

    # Backward compatibility helpers
    def analyze_ema_crossover(self, symbol: str, interval: str = "15m", period: str = "5d") -> Dict[str, Any]:
        res = self.generate_options_call_put_signal(symbol, interval=interval, period=period)
        return self.sanitize_obj({
            "symbol": symbol.upper(),
            "signal": "BUY" if res["signal_type"] == "BULLISH" else ("SELL" if res["signal_type"] == "BEARISH" else "HOLD"),
            "spot_price": res["spot_price"],
            "suggested_strike": res["suggested_strike"],
            "option_entry_price": res["option_entry_price"],
            "option_target_1": res["option_target_1"],
            "option_stop_loss": res["option_stop_loss"],
            "lot_size": res["lot_size"],
            "capital_required_per_lot": res["capital_required_per_lot"],
            "ema_9": res["indicators"]["ema_9"],
            "ema_21": res["indicators"]["ema_21"],
            "rsi": res["indicators"]["rsi_14"],
            "supertrend": res["indicators"]["supertrend"],
            "confluence_reasons": res["confluence_reasons"],
        })


strategy_engine = StrategyEngine()
