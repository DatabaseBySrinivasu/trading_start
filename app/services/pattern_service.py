import math
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PatternService:
    """
    Advanced Technical & Smart Money Concepts (SMC) Pattern Engine.
    Covers:
    1. Candlestick Pattern Recognition (Engulfing, Hammer, Shooting Star, Morning/Evening Star, Marubozu, Doji)
    2. Day High / Day Low, Central Pivot Range (CPR), and Camarilla Pivots
    3. Volume Spikes, Volume Moving Averages, and Intraday VWAP
    4. Order Blocks (Institutional Demand/Supply Zones) and Fair Value Gaps (FVG)
    5. W-Patterns (Double Bottom) and M-Patterns (Double Top) with Breakout Triggers & Targets
    """

    @staticmethod
    def _clean_float(val: Any, default: float = 0.0) -> float:
        try:
            if val is None:
                return default
            f = float(val)
            return default if (math.isnan(f) or math.isinf(f)) else f
        except Exception:
            return default

    @classmethod
    def _clean_df(cls, df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        if df is None or df.empty:
            return None
        cleaned = df.copy().dropna(subset=["open", "high", "low", "close"])
        return cleaned if not cleaned.empty else None

    @classmethod
    def calculate_vwap(cls, df: pd.DataFrame) -> pd.Series:
        """Calculate Volume Weighted Average Price (VWAP)."""
        df = cls._clean_df(df)
        if df is None or len(df) == 0:
            return pd.Series([0.0])
        typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
        volume = df["volume"].fillna(1).replace(0, 1.0)
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        return vwap

    @classmethod
    def analyze_volume(cls, df: pd.DataFrame, lookback: int = 20) -> Dict[str, Any]:
        """
        Analyze volume trends, volume spikes (>1.5x / >2.0x 20-period SMA), and VWAP status.
        """
        df = cls._clean_df(df)
        if df is None or len(df) < 5:
            return {
                "current_volume": 0,
                "volume_sma": 0,
                "volume_ratio": 1.0,
                "volume_spike": False,
                "volume_bias": "NEUTRAL",
                "vwap": 0.0,
                "price_vs_vwap": "NEUTRAL",
                "spike_status": "NORMAL",
                "spike_description": "Normal volume activity",
                "volume_sentiment": "NEUTRAL",
                "vwap_description": "Price is tracking near VWAP",
            }

        vol_sma = df["volume"].rolling(window=min(lookback, len(df)), min_periods=1).mean()
        vwap_series = cls.calculate_vwap(df)

        cur_vol = cls._clean_float(df["volume"].iloc[-1], 1000.0)
        avg_vol = cls._clean_float(vol_sma.iloc[-1], 1000.0)
        if avg_vol <= 0:
            avg_vol = 1.0
        ratio = round(cur_vol / avg_vol, 2)
        cur_price = cls._clean_float(df["close"].iloc[-1], 100.0)
        cur_vwap = cls._clean_float(vwap_series.iloc[-1], cur_price)

        # Spike classification
        if ratio >= 2.0:
            spike_status = "EXTREME_SPIKE"
            spike_desc = f"Institutional Volume Surge ({ratio}x of 20-period avg)"
        elif ratio >= 1.5:
            spike_status = "MODERATE_SPIKE"
            spike_desc = f"Elevated Volume Expansion ({ratio}x of 20-period avg)"
        elif ratio < 0.6:
            spike_status = "LOW_VOLUME"
            spike_desc = f"Subdued / Low Volume ({ratio}x of 20-period avg)"
        else:
            spike_status = "NORMAL"
            spike_desc = f"Normal Volume Flow ({ratio}x avg)"

        # Price vs VWAP
        if cur_price > cur_vwap:
            vwap_bias = "BULLISH_ABOVE_VWAP"
            diff = cur_price - cur_vwap
            vwap_diff_pct = round((diff / cur_vwap) * 100, 2) if cur_vwap > 0 else 0.0
            vwap_desc = f"Price is ₹{diff:.2f} (+{vwap_diff_pct}%) above VWAP ({cur_vwap:.2f})"
        else:
            vwap_bias = "BEARISH_BELOW_VWAP"
            diff = cur_vwap - cur_price
            vwap_diff_pct = round((diff / cur_vwap) * 100, 2) if cur_vwap > 0 else 0.0
            vwap_desc = f"Price is ₹{diff:.2f} (-{vwap_diff_pct}%) below VWAP ({cur_vwap:.2f})"

        # Volume Price Confluence
        last_candle_bull = cls._clean_float(df["close"].iloc[-1]) >= cls._clean_float(df["open"].iloc[-1])
        if ratio >= 1.5 and last_candle_bull:
            vol_sentiment = "BULLISH_ACCUMULATION"
        elif ratio >= 1.5 and not last_candle_bull:
            vol_sentiment = "BEARISH_DISTRIBUTION"
        else:
            vol_sentiment = "NEUTRAL"

        return {
            "current_volume": int(cur_vol),
            "volume_sma_20": int(avg_vol),
            "volume_ratio": ratio,
            "spike_status": spike_status,
            "spike_description": spike_desc,
            "volume_sentiment": vol_sentiment,
            "vwap": round(cur_vwap, 2),
            "price_vs_vwap": vwap_bias,
            "vwap_description": vwap_desc,
        }

    @classmethod
    def calculate_day_levels(cls, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculates:
        - Previous Day High (PDH), Previous Day Low (PDL), Previous Day Close (PDC)
        - Current Day High (CDH), Current Day Low (CDL), Current Day Open (CDO)
        - Central Pivot Range (CPR): Pivot (P), Top Central (TC), Bottom Central (BC)
        - Classical Pivots (R1, R2, S1, S2)
        - Breakout / Breakdown evaluation
        """
        df = cls._clean_df(df)
        if df is None or len(df) < 2:
            return {
                "current_price": 0.0,
                "pdh": 0.0,
                "pdl": 0.0,
                "pdc": 0.0,
                "cdh": 0.0,
                "cdl": 0.0,
                "cdo": 0.0,
                "cpr": {
                    "pivot": 0.0,
                    "tc": 0.0,
                    "bc": 0.0,
                    "width_pct": 0.0,
                    "cpr_type": "AVERAGE_CPR",
                    "expectation": "Normal Volatility",
                },
                "pivots": {"r2": 0.0, "r1": 0.0, "pivot": 0.0, "s1": 0.0, "s2": 0.0},
                "status": "NORMAL",
                "bias": "NEUTRAL",
                "description": "Insufficient historical candle data",
            }

        cur = df.iloc[-1]
        cur_price = cls._clean_float(cur["close"], 0.0)

        prev = df.iloc[-2]
        pdh = cls._clean_float(prev["high"], cur_price)
        pdl = cls._clean_float(prev["low"], cur_price)
        pdc = cls._clean_float(prev["close"], cur_price)

        cdh = cls._clean_float(cur["high"], cur_price)
        cdl = cls._clean_float(cur["low"], cur_price)
        cdo = cls._clean_float(cur["open"], cur_price)

        # Central Pivot Range (CPR)
        pivot = (pdh + pdl + pdc) / 3.0
        bc = (pdh + pdl) / 2.0
        tc = (pivot - bc) + pivot

        # Ensure TC is top and BC is bottom
        cpr_top = max(tc, bc)
        cpr_bottom = min(tc, bc)
        cpr_width = abs(tc - bc)
        cpr_width_pct = round((cpr_width / (pivot + 1e-9)) * 100, 3) if pivot > 0 else 0.0

        cpr_type = "NARROW_CPR" if cpr_width_pct <= 0.25 else ("WIDE_CPR" if cpr_width_pct >= 0.6 else "AVERAGE_CPR")
        cpr_expectation = (
            "Trending / High Momentum Day Expected (Narrow CPR)"
            if cpr_type == "NARROW_CPR"
            else ("Sideways / Range-bound Consolidation Expected (Wide CPR)" if cpr_type == "WIDE_CPR" else "Normal Volatility")
        )

        # Classic Support & Resistance
        r1 = (2.0 * pivot) - pdl
        s1 = (2.0 * pivot) - pdh
        r2 = pivot + (pdh - pdl)
        s2 = pivot - (pdh - pdl)
        r3 = pdh + 2.0 * (pivot - pdl)
        s3 = pdl - 2.0 * (pdh - pivot)
        r4 = r3 + (pdh - pdl)
        s4 = s3 - (pdh - pdl)

        # Camarilla Pivots (H1-H5, L1-L5)
        cam = cls.calculate_camarilla_pivots(pdh, pdl, pdc, cur_price)

        # Breakout status
        if cur_price > pdh and pdh > 0:
            status = "PDH_BREAKOUT"
            bias = "STRONG_BULLISH"
            reason = f"Price (₹{cur_price:.2f}) broke above Previous Day High (PDH: ₹{pdh:.2f})"
        elif cur_price < pdl and pdl > 0:
            status = "PDL_BREAKDOWN"
            bias = "STRONG_BEARISH"
            reason = f"Price (₹{cur_price:.2f}) broke below Previous Day Low (PDL: ₹{pdl:.2f})"
        elif cur_price > cpr_top and cpr_top > 0:
            status = "ABOVE_CPR"
            bias = "BULLISH_BIAS"
            reason = f"Price is trading above Central Pivot Range (TC: ₹{cpr_top:.2f})"
        elif cur_price < cpr_bottom and cpr_bottom > 0:
            status = "BELOW_CPR"
            bias = "BEARISH_BIAS"
            reason = f"Price is trading below Central Pivot Range (BC: ₹{cpr_bottom:.2f})"
        else:
            status = "INSIDE_CPR"
            bias = "NEUTRAL_CONSOLIDATION"
            reason = f"Price is inside Central Pivot Range (₹{cpr_bottom:.2f} - ₹{cpr_top:.2f})"

        return {
            "current_price": round(cur_price, 2),
            "pdh": round(pdh, 2),
            "pdl": round(pdl, 2),
            "pdc": round(pdc, 2),
            "cdh": round(cdh, 2),
            "cdl": round(cdl, 2),
            "cdo": round(cdo, 2),
            "cpr": {
                "pivot": round(pivot, 2),
                "tc": round(cpr_top, 2),
                "bc": round(cpr_bottom, 2),
                "width_pct": cpr_width_pct,
                "cpr_type": cpr_type,
                "expectation": cpr_expectation,
            },
            "pivots": {
                "r4": round(r4, 2),
                "r3": round(r3, 2),
                "r2": round(r2, 2),
                "r1": round(r1, 2),
                "pivot": round(pivot, 2),
                "s1": round(s1, 2),
                "s2": round(s2, 2),
                "s3": round(s3, 2),
                "s4": round(s4, 2),
            },
            "classical_pivots": {
                "r4": round(r4, 2),
                "r3": round(r3, 2),
                "r2": round(r2, 2),
                "r1": round(r1, 2),
                "pivot": round(pivot, 2),
                "s1": round(s1, 2),
                "s2": round(s2, 2),
                "s3": round(s3, 2),
                "s4": round(s4, 2),
            },
            "camarilla": cam,
            "camarilla_pivots": cam,
            "status": status,
            "bias": bias,
            "description": reason,
        }

    @classmethod
    def detect_candlestick_patterns(cls, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Scans recent candles for high-probability Japanese candlestick patterns:
        - Bullish & Bearish Engulfing
        - Hammer & Inverted Hammer
        - Shooting Star & Hanging Man
        - Morning Star & Evening Star
        - Bullish & Bearish Marubozu
        - Doji (Standard, Dragonfly, Gravestone)
        """
        df = cls._clean_df(df)
        if df is None or len(df) < 4:
            return []

        patterns = []
        c0 = df.iloc[-1]  # Current candle
        c1 = df.iloc[-2]  # Previous candle
        c2 = df.iloc[-3]  # Two candles ago

        def candle_props(row):
            o = cls._clean_float(row["open"])
            h = cls._clean_float(row["high"])
            l = cls._clean_float(row["low"])
            c = cls._clean_float(row["close"])
            body = abs(c - o)
            candle_range = max(h - l, 1e-5)
            upper_wick = h - max(o, c)
            lower_wick = min(o, c) - l
            is_bull = c >= o
            is_bear = c < o
            body_pct = body / candle_range
            return {
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "body": body,
                "range": candle_range,
                "upper_wick": upper_wick,
                "lower_wick": lower_wick,
                "is_bull": is_bull,
                "is_bear": is_bear,
                "body_pct": body_pct,
            }

        p0 = candle_props(c0)
        p1 = candle_props(c1)
        p2 = candle_props(c2)

        # 1. Bullish Engulfing
        if p1["is_bear"] and p0["is_bull"] and p0["open"] <= p1["close"] and p0["close"] >= p1["open"] and p0["body"] > p1["body"]:
            patterns.append({
                "name": "Bullish Engulfing",
                "type": "BULLISH",
                "strength": "HIGH",
                "description": "Green candle completely engulfs previous red candle body. Strong institutional buying demand.",
            })

        # 2. Bearish Engulfing
        if p1["is_bull"] and p0["is_bear"] and p0["open"] >= p1["close"] and p0["close"] <= p1["open"] and p0["body"] > p1["body"]:
            patterns.append({
                "name": "Bearish Engulfing",
                "type": "BEARISH",
                "strength": "HIGH",
                "description": "Red candle completely engulfs previous green candle body. Strong institutional selling pressure.",
            })

        # 3. Hammer (Bullish Reversal)
        if p0["lower_wick"] >= (2.0 * p0["body"]) and p0["upper_wick"] <= (0.25 * p0["body"]) and p0["range"] > 0:
            patterns.append({
                "name": "Hammer",
                "type": "BULLISH",
                "strength": "MEDIUM-HIGH",
                "description": "Long lower rejection shadow with small upper body, indicating buyers aggressively defended lower prices.",
            })

        # 4. Shooting Star (Bearish Reversal)
        if p0["upper_wick"] >= (2.0 * p0["body"]) and p0["lower_wick"] <= (0.25 * p0["body"]) and p0["range"] > 0:
            patterns.append({
                "name": "Shooting Star",
                "type": "BEARISH",
                "strength": "MEDIUM-HIGH",
                "description": "Long upper rejection wick with small lower body, signaling buyers exhausted and sellers taking control.",
            })

        # 5. Morning Star (3-Candle Bullish Reversal)
        if p2["is_bear"] and p2["body_pct"] >= 0.5 and p1["body_pct"] <= 0.3 and p0["is_bull"] and p0["close"] >= (p2["open"] + p2["close"]) / 2:
            patterns.append({
                "name": "Morning Star",
                "type": "BULLISH",
                "strength": "VERY_HIGH",
                "description": "3-candle bullish reversal pattern: Bearish drop -> Indecision star -> Strong bullish candle taking back >50% of drop.",
            })

        # 6. Evening Star (3-Candle Bearish Reversal)
        if p2["is_bull"] and p2["body_pct"] >= 0.5 and p1["body_pct"] <= 0.3 and p0["is_bear"] and p0["close"] <= (p2["open"] + p2["close"]) / 2:
            patterns.append({
                "name": "Evening Star",
                "type": "BEARISH",
                "strength": "VERY_HIGH",
                "description": "3-candle bearish reversal pattern: Strong rally -> Exhaustion star at top -> Powerful bearish engulfing descent.",
            })

        # 7. Marubozu (Institutional Momentum Candle)
        if p0["body_pct"] >= 0.85:
            if p0["is_bull"]:
                patterns.append({
                    "name": "Bullish Marubozu",
                    "type": "BULLISH",
                    "strength": "HIGH",
                    "description": "Solid green candle with virtually no wicks, showing dominant unilateral buying momentum.",
                })
            elif p0["is_bear"]:
                patterns.append({
                    "name": "Bearish Marubozu",
                    "type": "BEARISH",
                    "strength": "HIGH",
                    "description": "Solid red candle with virtually no wicks, showing dominant unilateral selling momentum.",
                })

        # 8. Doji (Indecision)
        if p0["body_pct"] <= 0.1:
            if p0["lower_wick"] >= 2.0 * p0["upper_wick"] and p0["lower_wick"] > 0:
                doji_name = "Dragonfly Doji (Bullish Bias)"
                doji_type = "BULLISH"
            elif p0["upper_wick"] >= 2.0 * p0["lower_wick"] and p0["upper_wick"] > 0:
                doji_name = "Gravestone Doji (Bearish Bias)"
                doji_type = "BEARISH"
            else:
                doji_name = "Standard Doji"
                doji_type = "NEUTRAL"
            patterns.append({
                "name": doji_name,
                "type": doji_type,
                "strength": "MEDIUM",
                "description": "Open and close virtually identical, indicating equilibrium or impending volatility expansion.",
            })

        return patterns

    @classmethod
    def detect_order_blocks(cls, df: pd.DataFrame, lookback: int = 40) -> Dict[str, Any]:
        """
        Smart Money Concepts (SMC) Order Block & Fair Value Gap (FVG) Detector.
        """
        df = cls._clean_df(df)
        if df is None or len(df) < 10:
            return {
                "nearest_bullish_ob": None,
                "nearest_bearish_ob": None,
                "active_bullish_obs_count": 0,
                "active_bearish_obs_count": 0,
                "latest_fvg": None,
                "smc_bias": "NEUTRAL_RANGE",
                "description": "Insufficient candle depth for SMC analysis",
            }

        window = df.tail(lookback).copy().reset_index(drop=True)
        cur_price = cls._clean_float(window["close"].iloc[-1], 0.0)

        bullish_obs = []
        bearish_obs = []
        fvgs = []

        # Detect FVGs & Order Blocks
        for i in range(2, len(window) - 1):
            c_prev2 = window.iloc[i - 2]
            c_prev1 = window.iloc[i - 1]
            c_curr = window.iloc[i]
            c_next = window.iloc[i + 1]

            # 1. Bullish FVG
            c_curr_low = cls._clean_float(c_curr["low"])
            c_prev2_high = cls._clean_float(c_prev2["high"])
            if c_curr_low > c_prev2_high:
                fvgs.append({
                    "type": "BULLISH_FVG",
                    "top": round(c_curr_low, 2),
                    "bottom": round(c_prev2_high, 2),
                    "size": round(c_curr_low - c_prev2_high, 2),
                    "candle_idx": i,
                })

            # 2. Bearish FVG
            c_curr_high = cls._clean_float(c_curr["high"])
            c_prev2_low = cls._clean_float(c_prev2["low"])
            if c_curr_high < c_prev2_low:
                fvgs.append({
                    "type": "BEARISH_FVG",
                    "top": round(c_prev2_low, 2),
                    "bottom": round(c_curr_high, 2),
                    "size": round(c_prev2_low - c_curr_high, 2),
                    "candle_idx": i,
                })

            # 3. Bullish Order Block
            if cls._clean_float(c_prev1["close"]) < cls._clean_float(c_prev1["open"]):
                if cls._clean_float(c_curr["close"]) > cls._clean_float(c_prev1["high"]) and cls._clean_float(c_next["close"]) > cls._clean_float(c_curr["close"]):
                    ob_high = cls._clean_float(c_prev1["high"])
                    ob_low = cls._clean_float(c_prev1["low"])
                    subsequent = window.iloc[i + 1:]
                    min_subsequent = cls._clean_float(subsequent["low"].min(), cur_price) if not subsequent.empty else cur_price
                    mitigated = min_subsequent < ob_low

                    if not mitigated and ob_low <= cur_price:
                        bullish_obs.append({
                            "ob_zone_top": round(ob_high, 2),
                            "ob_zone_bottom": round(ob_low, 2),
                            "mid_level": round((ob_high + ob_low) / 2, 2),
                            "status": "ACTIVE_DEMAND_ZONE",
                            "distance_pts": round(cur_price - ob_high, 2),
                        })

            # 4. Bearish Order Block
            if cls._clean_float(c_prev1["close"]) > cls._clean_float(c_prev1["open"]):
                if cls._clean_float(c_curr["close"]) < cls._clean_float(c_prev1["low"]) and cls._clean_float(c_next["close"]) < cls._clean_float(c_curr["close"]):
                    ob_high = cls._clean_float(c_prev1["high"])
                    ob_low = cls._clean_float(c_prev1["low"])
                    subsequent = window.iloc[i + 1:]
                    max_subsequent = cls._clean_float(subsequent["high"].max(), cur_price) if not subsequent.empty else cur_price
                    mitigated = max_subsequent > ob_high

                    if not mitigated and ob_high >= cur_price:
                        bearish_obs.append({
                            "ob_zone_top": round(ob_high, 2),
                            "ob_zone_bottom": round(ob_low, 2),
                            "mid_level": round((ob_high + ob_low) / 2, 2),
                            "status": "ACTIVE_SUPPLY_ZONE",
                            "distance_pts": round(ob_low - cur_price, 2),
                        })

        nearest_bull_ob = bullish_obs[-1] if bullish_obs else None
        nearest_bear_ob = bearish_obs[-1] if bearish_obs else None
        latest_fvg = fvgs[-1] if fvgs else None

        # SMC Bias
        if nearest_bull_ob and cur_price > 0 and abs(cur_price - nearest_bull_ob["ob_zone_top"]) <= (cur_price * 0.007):
            smc_bias = "AT_DEMAND_ORDER_BLOCK"
            smc_desc = f"Price is reacting inside Bullish Order Block demand zone (₹{nearest_bull_ob['ob_zone_bottom']} - ₹{nearest_bull_ob['ob_zone_top']})"
        elif nearest_bear_ob and cur_price > 0 and abs(nearest_bear_ob["ob_zone_bottom"] - cur_price) <= (cur_price * 0.007):
            smc_bias = "AT_SUPPLY_ORDER_BLOCK"
            smc_desc = f"Price is reacting inside Bearish Order Block supply zone (₹{nearest_bear_ob['ob_zone_bottom']} - ₹{nearest_bear_ob['ob_zone_top']})"
        else:
            smc_bias = "NEUTRAL_RANGE"
            smc_desc = "Price is traversing between institutional Order Block liquidity pools"

        return {
            "nearest_bullish_ob": nearest_bull_ob,
            "nearest_bearish_ob": nearest_bear_ob,
            "active_bullish_obs_count": len(bullish_obs),
            "active_bearish_obs_count": len(bearish_obs),
            "latest_fvg": latest_fvg,
            "smc_bias": smc_bias,
            "description": smc_desc,
        }

    @classmethod
    def detect_wm_patterns(cls, df: pd.DataFrame, lookback: int = 35) -> Dict[str, Any]:
        """
        Detects W-Pattern (Double Bottom / Bullish Reversal) and M-Pattern (Double Top / Bearish Reversal).
        """
        df = cls._clean_df(df)
        if df is None or len(df) < 15:
            return {
                "pattern": "NONE",
                "pattern_type": "NEUTRAL",
                "status": "NO_DATA",
                "bias": "NEUTRAL",
                "description": "Insufficient candles to detect W/M formations",
                "neckline": None,
                "target": None,
                "stop_loss": None,
            }

        window = df.tail(lookback).copy().reset_index(drop=True)
        closes = [cls._clean_float(x) for x in window["close"].values]
        highs = [cls._clean_float(x) for x in window["high"].values]
        lows = [cls._clean_float(x) for x in window["low"].values]
        cur_price = closes[-1] if closes else 0.0

        # Find local peaks and troughs
        peaks: List[Tuple[int, float]] = []
        troughs: List[Tuple[int, float]] = []

        for i in range(2, len(window) - 2):
            if highs[i] > highs[i - 1] and highs[i] > highs[i - 2] and highs[i] > highs[i + 1] and highs[i] > highs[i + 2]:
                peaks.append((i, highs[i]))
            if lows[i] < lows[i - 1] and lows[i] < lows[i - 2] and lows[i] < lows[i + 1] and lows[i] < lows[i + 2]:
                troughs.append((i, lows[i]))

        # 1. Check for W-Pattern (Double Bottom)
        if len(troughs) >= 2 and len(peaks) >= 1:
            t1_idx, t1_val = troughs[-2]
            t2_idx, t2_val = troughs[-1]

            middle_peaks = [p for p in peaks if t1_idx < p[0] < t2_idx]
            if middle_peaks and t1_val > 0:
                neck_idx, neck_val = max(middle_peaks, key=lambda x: x[1])
                diff_pct = abs(t1_val - t2_val) / t1_val

                if diff_pct <= 0.025 and neck_val > max(t1_val, t2_val) * 1.005:
                    pattern_height = neck_val - min(t1_val, t2_val)
                    target = round(neck_val + pattern_height, 2)
                    stop_loss = round(min(t1_val, t2_val), 2)

                    if cur_price >= neck_val:
                        w_status = "CONFIRMED_BREAKOUT"
                        bias = "BULLISH_CALL"
                        desc = f"W-Pattern Double Bottom confirmed! Price broke above Neckline ₹{neck_val:.2f}. Target: ₹{target:.2f}"
                    elif cur_price > t2_val:
                        w_status = "FORMING_RIGHT_ARM"
                        bias = "POTENTIAL_BULLISH"
                        desc = f"W-Pattern Double Bottom forming near ₹{t2_val:.2f}. Approaching Neckline resistance at ₹{neck_val:.2f}."
                    else:
                        w_status = "INVALIDATED"
                        bias = "NEUTRAL"
                        desc = "W-Pattern invalidated below swing low."

                    if w_status != "INVALIDATED":
                        return {
                            "pattern": "W_PATTERN (DOUBLE BOTTOM)",
                            "pattern_type": "BULLISH",
                            "status": w_status,
                            "bias": bias,
                            "low_1": round(t1_val, 2),
                            "low_2": round(t2_val, 2),
                            "neckline": round(neck_val, 2),
                            "target": target,
                            "stop_loss": stop_loss,
                            "description": desc,
                        }

        # 2. Check for M-Pattern (Double Top)
        if len(peaks) >= 2 and len(troughs) >= 1:
            p1_idx, p1_val = peaks[-2]
            p2_idx, p2_val = peaks[-1]

            middle_troughs = [t for t in troughs if p1_idx < t[0] < p2_idx]
            if middle_troughs and p1_val > 0:
                neck_idx, neck_val = min(middle_troughs, key=lambda x: x[1])
                diff_pct = abs(p1_val - p2_val) / p1_val

                if diff_pct <= 0.025 and neck_val < min(p1_val, p2_val) * 0.995:
                    pattern_height = max(p1_val, p2_val) - neck_val
                    target = round(neck_val - pattern_height, 2)
                    stop_loss = round(max(p1_val, p2_val), 2)

                    if cur_price <= neck_val:
                        m_status = "CONFIRMED_BREAKDOWN"
                        bias = "BEARISH_PUT"
                        desc = f"M-Pattern Double Top confirmed! Price broke below Neckline ₹{neck_val:.2f}. Target: ₹{target:.2f}"
                    elif cur_price < p2_val:
                        m_status = "FORMING_RIGHT_PEAK"
                        bias = "POTENTIAL_BEARISH"
                        desc = f"M-Pattern Double Top forming near ₹{p2_val:.2f}. Approaching Neckline support at ₹{neck_val:.2f}."
                    else:
                        m_status = "INVALIDATED"
                        bias = "NEUTRAL"
                        desc = "M-Pattern invalidated above swing high."

                    if m_status != "INVALIDATED":
                        return {
                            "pattern": "M_PATTERN (DOUBLE TOP)",
                            "pattern_type": "BEARISH",
                            "status": m_status,
                            "bias": bias,
                            "peak_1": round(p1_val, 2),
                            "peak_2": round(p2_val, 2),
                            "neckline": round(neck_val, 2),
                            "target": target,
                            "stop_loss": stop_loss,
                            "description": desc,
                        }

        return {
            "pattern": "NO_ACTIVE_W_OR_M",
            "pattern_type": "NEUTRAL",
            "status": "CONSOLIDATING",
            "bias": "NEUTRAL",
            "description": "Price action is consolidating without complete W or M formation.",
            "neckline": None,
            "target": None,
            "stop_loss": None,
        }

    @classmethod
    def analyze_price_action_momentum(cls, df: pd.DataFrame, lookback: int = 25) -> Dict[str, Any]:
        """
        Evaluates Price Action Momentum, Impulse Velocity, Candle Body Expansion,
        Higher High/Higher Low vs Lower High/Lower Low structure, Break of Structure (BOS),
        and Rejection Wicks.
        """
        df = cls._clean_df(df)
        if df is None or len(df) < 6:
            return {
                "momentum_score": 0.0,
                "momentum_regime": "NEUTRAL_COMPRESSION",
                "trend_structure": "CONSOLIDATION",
                "structure_description": "Insufficient candles",
                "impulse_direction": "FLAT",
                "body_expansion_ratio": 1.0,
                "consecutive_streak": 0,
                "streak_direction": "FLAT",
                "bos_status": "NONE",
                "bos_description": "No data",
                "wick_rejection": "NONE",
                "rejection_description": "No data",
                "velocity_pts_per_candle": 0.0,
                "net_thrust_pts": 0.0,
                "description": "Insufficient candles to assess price action momentum",
            }

        window = df.tail(lookback).copy().reset_index(drop=True)
        recent_n = min(5, len(window))

        # 1. Candle Bodies and Ranges
        bodies = (window["close"] - window["open"]).abs()
        ranges = (window["high"] - window["low"]).replace(0, 1e-6)

        avg_hist_body = float(bodies.iloc[:-recent_n].mean()) if len(bodies) > recent_n else float(bodies.mean())
        if avg_hist_body <= 0:
            avg_hist_body = 1e-4
        avg_recent_body = float(bodies.tail(recent_n).mean())
        body_expansion_ratio = round(avg_recent_body / avg_hist_body, 2)

        # 2. Consecutive Candle Streak & Net Directional Velocity
        last_c = window["close"].iloc[-1]
        last_o = window["open"].iloc[-1]
        streak = 0
        streak_dir = "BULLISH" if last_c >= last_o else "BEARISH"

        for i in range(len(window) - 1, -1, -1):
            c_close = window["close"].iloc[i]
            c_open = window["open"].iloc[i]
            is_green = c_close >= c_open
            if (streak_dir == "BULLISH" and is_green) or (streak_dir == "BEARISH" and not is_green):
                streak += 1
            else:
                break

        net_thrust = float(window["close"].iloc[-1] - window["close"].iloc[-recent_n])
        velocity = round(net_thrust / recent_n, 2)

        # 3. Trend Structure (HH/HL vs LH/LL)
        highs = window["high"].tail(recent_n).values
        lows = window["low"].tail(recent_n).values
        hh_count = sum(1 for i in range(1, len(highs)) if highs[i] > highs[i - 1])
        hl_count = sum(1 for i in range(1, len(lows)) if lows[i] > lows[i - 1])
        lh_count = sum(1 for i in range(1, len(highs)) if highs[i] < highs[i - 1])
        ll_count = sum(1 for i in range(1, len(lows)) if lows[i] < lows[i - 1])

        if hh_count >= 2 and hl_count >= 2:
            trend_structure = "BULLISH_HH_HL"
            structure_desc = "Higher Highs & Higher Lows (Impulse Up)"
        elif lh_count >= 2 and ll_count >= 2:
            trend_structure = "BEARISH_LH_LL"
            structure_desc = "Lower Highs & Lower Lows (Impulse Down)"
        else:
            trend_structure = "RANGE_CONSOLIDATION"
            structure_desc = "Range Compression / Consolidation"

        # 4. Break of Structure (BOS)
        cur_price = float(window["close"].iloc[-1])
        swing_high = float(window["high"].iloc[:-recent_n].max()) if len(window) > recent_n else float(window["high"].max())
        swing_low = float(window["low"].iloc[:-recent_n].min()) if len(window) > recent_n else float(window["low"].min())

        if cur_price > swing_high:
            bos_status = "BOS_BULLISH_BREAKOUT"
            bos_desc = f"Broke structure swing high (₹{swing_high:.2f})"
        elif cur_price < swing_low:
            bos_status = "BOS_BEARISH_BREAKDOWN"
            bos_desc = f"Broke structure swing low (₹{swing_low:.2f})"
        else:
            bos_status = "WITHIN_STRUCTURE"
            bos_desc = "Trading within structure"

        # 5. Wick Rejection & Exhaustion
        last_bar = window.iloc[-1]
        c_h = float(last_bar["high"])
        c_l = float(last_bar["low"])
        c_c = float(last_bar["close"])
        c_o = float(last_bar["open"])
        c_rng = max(c_h - c_l, 1e-4)
        upper_wick = c_h - max(c_c, c_o)
        lower_wick = min(c_c, c_o) - c_l
        cur_body = abs(c_c - c_o)

        if upper_wick >= 2.0 * cur_body and (upper_wick / c_rng) >= 0.45:
            wick_rejection = "BEARISH_UPPER_WICK_REJECTION"
            rejection_desc = "Top selling rejection wick / upside exhaustion"
        elif lower_wick >= 2.0 * cur_body and (lower_wick / c_rng) >= 0.45:
            wick_rejection = "BULLISH_LOWER_WICK_REJECTION"
            rejection_desc = "Bottom buying rejection wick / downside absorption"
        else:
            wick_rejection = "NONE"
            rejection_desc = "Clean directional body"

        # 6. Composite Momentum Score (-100 to +100)
        score = 0.0

        # Structure weight: 30
        if trend_structure == "BULLISH_HH_HL":
            score += 30.0
        elif trend_structure == "BEARISH_LH_LL":
            score -= 30.0

        # BOS weight: 25
        if bos_status == "BOS_BULLISH_BREAKOUT":
            score += 25.0
        elif bos_status == "BOS_BEARISH_BREAKDOWN":
            score -= 25.0

        # Streak & Velocity weight: 20
        if streak_dir == "BULLISH":
            score += min(20.0, streak * 6.0)
        else:
            score -= min(20.0, streak * 6.0)

        # Body expansion weight: 15
        if body_expansion_ratio >= 1.3:
            if streak_dir == "BULLISH":
                score += 15.0
            else:
                score -= 15.0

        # Wick Rejection adjustment: 10
        if wick_rejection == "BULLISH_LOWER_WICK_REJECTION":
            score += 10.0
        elif wick_rejection == "BEARISH_UPPER_WICK_REJECTION":
            score -= 10.0

        score = max(-100.0, min(100.0, round(score, 1)))

        # Momentum Regime
        if score >= 55.0:
            regime = "STRONG_BULLISH_IMPULSE"
            impulse_dir = "BULLISH_SURGE"
            regime_desc = f"Strong Bullish Momentum (+{score}) with {structure_desc} and {streak} consecutive green bars."
        elif score >= 20.0:
            regime = "MODERATE_BULLISH_MOMENTUM"
            impulse_dir = "BULLISH_DRIFT"
            regime_desc = f"Moderate Bullish Momentum (+{score}). Price grinding higher with positive velocity (₹{velocity}/candle)."
        elif score <= -55.0:
            regime = "STRONG_BEARISH_IMPULSE"
            impulse_dir = "BEARISH_SURGE"
            regime_desc = f"Strong Bearish Momentum ({score}) with {structure_desc} and {streak} consecutive red bars."
        elif score <= -20.0:
            regime = "MODERATE_BEARISH_MOMENTUM"
            impulse_dir = "BEARISH_DRIFT"
            regime_desc = f"Moderate Bearish Momentum ({score}). Price sliding lower with negative velocity (₹{velocity}/candle)."
        else:
            regime = "NEUTRAL_COMPRESSION"
            impulse_dir = "RANGE_COIL"
            regime_desc = f"Range Compression / Momentum Squeeze ({score}). Low volatility coiling before next breakout."

        return {
            "momentum_score": score,
            "momentum_regime": regime,
            "trend_structure": trend_structure,
            "structure_description": structure_desc,
            "impulse_direction": impulse_dir,
            "body_expansion_ratio": body_expansion_ratio,
            "consecutive_streak": streak,
            "streak_direction": streak_dir,
            "bos_status": bos_status,
            "bos_description": bos_desc,
            "wick_rejection": wick_rejection,
            "rejection_description": rejection_desc,
            "velocity_pts_per_candle": velocity,
            "net_thrust_pts": round(net_thrust, 2),
            "description": regime_desc,
        }

    @classmethod
    def calculate_camarilla_pivots(
        cls, high: float, low: float, close: float, cur_price: float = 0.0
    ) -> Dict[str, Any]:
        """
        Calculates Camarilla Pivot Points (H1-H5, L1-L5).
        Key Trading Zones:
        - H4: Institutional Long Breakout Target / Expansion
        - H3: Range High Resistance / Short Reversal Zone
        - L3: Range Low Support / Long Bounce Zone
        - L4: Institutional Short Breakdown Target / Expansion
        - H5 / L5: Ultimate Extension / Exhaustion Targets
        """
        high = cls._clean_float(high)
        low = cls._clean_float(low)
        close = cls._clean_float(close)
        rng = max(0.001, high - low)
        cur_p = cls._clean_float(cur_price, close)

        h5 = (high / (low + 1e-9)) * close if low > 0 else close + (rng * 1.1)
        h4 = close + (rng * 0.55)
        h3 = close + (rng * 0.275)
        h2 = close + (rng * 0.1833)
        h1 = close + (rng * 0.0916)

        l1 = close - (rng * 0.0916)
        l2 = close - (rng * 0.1833)
        l3 = close - (rng * 0.275)
        l4 = close - (rng * 0.55)
        l5 = close - (h5 - close)

        # Status & Actionable Bias
        if cur_p >= h4:
            cam_status = "BULLISH_H4_BREAKOUT"
            cam_bias = "STRONG_BULLISH"
            cam_desc = f"Price (₹{cur_p:.2f}) broke above Camarilla H4 (₹{h4:.2f}) - Expansion towards H5 (₹{h5:.2f})"
        elif cur_p >= h3:
            cam_status = "TESTING_H3_RESISTANCE"
            cam_bias = "RESISTANCE_CAUTION"
            cam_desc = f"Price (₹{cur_p:.2f}) testing Camarilla H3 Resistance (₹{h3:.2f}) - Watch for reversal or breakout to H4 (₹{h4:.2f})"
        elif cur_p <= l4:
            cam_status = "BEARISH_L4_BREAKDOWN"
            cam_bias = "STRONG_BEARISH"
            cam_desc = f"Price (₹{cur_p:.2f}) broke below Camarilla L4 (₹{l4:.2f}) - Expansion towards L5 (₹{l5:.2f})"
        elif cur_p <= l3:
            cam_status = "TESTING_L3_SUPPORT"
            cam_bias = "SUPPORT_BOUNCE"
            cam_desc = f"Price (₹{cur_p:.2f}) testing Camarilla L3 Support (₹{l3:.2f}) - Watch for bounce or breakdown to L4 (₹{l4:.2f})"
        else:
            cam_status = "INSIDE_CAMARILLA_RANGE"
            cam_bias = "NEUTRAL_RANGE"
            cam_desc = f"Price (₹{cur_p:.2f}) oscillating between Camarilla L3 (₹{l3:.2f}) Support and H3 (₹{h3:.2f}) Resistance"

        return {
            "h5": round(h5, 2),
            "h4": round(h4, 2),
            "h3": round(h3, 2),
            "h2": round(h2, 2),
            "h1": round(h1, 2),
            "l1": round(l1, 2),
            "l2": round(l2, 2),
            "l3": round(l3, 2),
            "l4": round(l4, 2),
            "l5": round(l5, 2),
            "status": cam_status,
            "bias": cam_bias,
            "description": cam_desc,
        }

    @classmethod
    def calculate_swing_clusters(
        cls, df: pd.DataFrame, lookback: int = 60, cluster_tolerance_pct: float = 0.25
    ) -> Dict[str, Any]:
        """
        Identifies structural swing high (supply) and swing low (demand) fractal clusters.
        Groups nearby peaks and troughs within cluster_tolerance_pct to establish strong horizontal barriers.
        """
        df = cls._clean_df(df)
        if df is None or len(df) < 5:
            return {"supply_clusters": [], "demand_clusters": [], "total_clusters": 0}

        window = df.tail(lookback).copy().reset_index(drop=True)
        highs = window["high"].values
        lows = window["low"].values
        n = len(window)

        raw_peaks = []
        raw_troughs = []

        for i in range(2, n - 2):
            if highs[i] >= highs[i - 1] and highs[i] >= highs[i - 2] and highs[i] >= highs[i + 1] and highs[i] >= highs[i + 2]:
                raw_peaks.append(float(highs[i]))
            if lows[i] <= lows[i - 1] and lows[i] <= lows[i - 2] and lows[i] <= lows[i + 1] and lows[i] <= lows[i + 2]:
                raw_troughs.append(float(lows[i]))

        supply_clusters: List[Dict[str, Any]] = []
        for p in sorted(raw_peaks, reverse=True):
            matched = False
            for c in supply_clusters:
                avg_p = c["price"]
                if abs(p - avg_p) / (avg_p + 1e-9) <= (cluster_tolerance_pct / 100.0):
                    c["touches"] += 1
                    c["price"] = round((c["price"] * (c["touches"] - 1) + p) / c["touches"], 2)
                    c["min_price"] = min(c["min_price"], p)
                    c["max_price"] = max(c["max_price"], p)
                    matched = True
                    break
            if not matched:
                supply_clusters.append({
                    "price": round(p, 2),
                    "min_price": round(p, 2),
                    "max_price": round(p, 2),
                    "touches": 1,
                    "type": "SUPPLY_RESISTANCE_ZONE",
                    "strength": "MODERATE",
                })

        demand_clusters: List[Dict[str, Any]] = []
        for t in sorted(raw_troughs):
            matched = False
            for c in demand_clusters:
                avg_t = c["price"]
                if abs(t - avg_t) / (avg_t + 1e-9) <= (cluster_tolerance_pct / 100.0):
                    c["touches"] += 1
                    c["price"] = round((c["price"] * (c["touches"] - 1) + t) / c["touches"], 2)
                    c["min_price"] = min(c["min_price"], t)
                    c["max_price"] = max(c["max_price"], t)
                    matched = True
                    break
            if not matched:
                demand_clusters.append({
                    "price": round(t, 2),
                    "min_price": round(t, 2),
                    "max_price": round(t, 2),
                    "touches": 1,
                    "type": "DEMAND_SUPPORT_ZONE",
                    "strength": "MODERATE",
                })

        for sc in supply_clusters:
            sc["strength"] = "VERY_STRONG" if sc["touches"] >= 3 else ("STRONG" if sc["touches"] == 2 else "MODERATE")
        for dc in demand_clusters:
            dc["strength"] = "VERY_STRONG" if dc["touches"] >= 3 else ("STRONG" if dc["touches"] == 2 else "MODERATE")

        return {
            "supply_clusters": sorted(supply_clusters, key=lambda x: x["price"], reverse=True),
            "demand_clusters": sorted(demand_clusters, key=lambda x: x["price"]),
            "total_clusters": len(supply_clusters) + len(demand_clusters),
        }

    @classmethod
    def calculate_macro_fibonacci(cls, df: pd.DataFrame, lookback: int = 60) -> Dict[str, Any]:
        """
        Calculates Macro Multi-Day Fibonacci Retracements & Extensions.
        Levels: 0.0%, 23.6%, 38.2%, 50.0%, 61.8% (Golden Pocket), 78.6%, 100.0%, 161.8% Golden Extension.
        """
        df = cls._clean_df(df)
        if df is None or len(df) < 5:
            return {"levels": {}, "nearest_support": 0.0, "nearest_resistance": 0.0, "status": "NO_DATA"}

        window = df.tail(lookback)
        swing_high = cls._clean_float(window["high"].max())
        swing_low = cls._clean_float(window["low"].min())
        cur_price = cls._clean_float(window["close"].iloc[-1])
        diff = max(0.001, swing_high - swing_low)

        fib_0 = swing_high
        fib_236 = swing_high - (0.236 * diff)
        fib_382 = swing_high - (0.382 * diff)
        fib_500 = swing_high - (0.500 * diff)
        fib_618 = swing_high - (0.618 * diff)
        fib_786 = swing_high - (0.786 * diff)
        fib_1000 = swing_low
        fib_1618 = swing_high + (0.618 * diff)

        levels = {
            "fib_161.8": round(fib_1618, 2),
            "fib_0.0": round(fib_0, 2),
            "fib_23.6": round(fib_236, 2),
            "fib_38.2": round(fib_382, 2),
            "fib_50.0": round(fib_500, 2),
            "fib_61.8": round(fib_618, 2),
            "fib_78.6": round(fib_786, 2),
            "fib_100.0": round(fib_1000, 2),
        }

        supports = [v for v in levels.values() if v < cur_price]
        resistances = [v for v in levels.values() if v > cur_price]

        nearest_supp = max(supports) if supports else round(cur_price * 0.98, 2)
        nearest_res = min(resistances) if resistances else round(cur_price * 1.02, 2)

        return {
            "swing_high": round(swing_high, 2),
            "swing_low": round(swing_low, 2),
            "current_price": round(cur_price, 2),
            "levels": levels,
            "nearest_support": round(nearest_supp, 2),
            "nearest_resistance": round(nearest_res, 2),
            "golden_pocket_support": round(fib_618, 2),
            "macro_range_pts": round(diff, 2),
        }

    @classmethod
    def calculate_multi_timeframe_sr(
        cls,
        df_intraday: Optional[pd.DataFrame] = None,
        df_daily: Optional[pd.DataFrame] = None,
        df_weekly: Optional[pd.DataFrame] = None,
        symbol: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Unified Multi-Timeframe Support & Resistance Engine.
        Ingests Intraday, Daily, and Weekly historical data to compute:
        1. Daily & Weekly Central Pivot Range (CPR)
        2. Daily & Weekly Classical Pivots (R1-R4, S1-S4)
        3. Daily & Weekly Camarilla Pivots (H1-H5, L1-L5)
        4. Multi-Day Extremes (PDH/PDL, 2DH/2DL, 3DH/3DL, 5DH/5DL or PWH/PWL, 10DH/10DL, 20DH/20DL or PMH/PML)
        5. Swing High/Low Supply & Demand Clusters
        6. Macro Fibonacci Retracements & Extensions
        7. Consolidated Resistance (R_near, R_mid, R_major) & Support (S_near, S_mid, S_major) Hierarchy
        8. Confluence Tagging & Actionable Structural Matrix
        """
        from app.services.market_data import market_data_service

        # 1. Fetch data if not supplied
        if df_daily is None or df_daily.empty:
            if symbol:
                df_daily = market_data_service.get_historical_candles(symbol, period="3mo", interval="1d")
            elif df_intraday is not None and not df_intraday.empty:
                df_daily = df_intraday.copy()

        if df_weekly is None or df_weekly.empty:
            if symbol:
                df_weekly = market_data_service.get_historical_candles(symbol, period="6mo", interval="1wk")
            elif df_daily is not None and not df_daily.empty:
                df_weekly = df_daily.copy()

        df_intra_clean = cls._clean_df(df_intraday)
        df_daily_clean = cls._clean_df(df_daily)
        df_weekly_clean = cls._clean_df(df_weekly)

        # Determine current spot price
        if df_intra_clean is not None and len(df_intra_clean) > 0:
            cur_price = cls._clean_float(df_intra_clean["close"].iloc[-1])
        elif df_daily_clean is not None and len(df_daily_clean) > 0:
            cur_price = cls._clean_float(df_daily_clean["close"].iloc[-1])
        elif symbol:
            live_quote = market_data_service.get_live_price(symbol)
            cur_price = cls._clean_float(live_quote["price"]) if live_quote else 1000.0
        else:
            cur_price = 1000.0

        if df_daily_clean is None or len(df_daily_clean) < 2:
            # Fallback baseline when insufficient history
            return {
                "symbol": symbol or "UNKNOWN",
                "current_price": round(cur_price, 2),
                "nearest_resistance": {"price": round(cur_price * 1.008, 2), "distance_pts": round(cur_price * 0.008, 2), "distance_pct": 0.8, "tags": ["Estimated R1"], "strength": "MODERATE"},
                "nearest_support": {"price": round(cur_price * 0.992, 2), "distance_pts": round(cur_price * 0.008, 2), "distance_pct": 0.8, "tags": ["Estimated S1"], "strength": "MODERATE"},
                "r_near": round(cur_price * 1.008, 2),
                "r_mid": round(cur_price * 1.016, 2),
                "r_major": round(cur_price * 1.025, 2),
                "s_near": round(cur_price * 0.992, 2),
                "s_mid": round(cur_price * 0.984, 2),
                "s_major": round(cur_price * 0.975, 2),
                "daily_cpr": {"pivot": round(cur_price, 2), "tc": round(cur_price * 1.002, 2), "bc": round(cur_price * 0.998, 2), "width_pct": 0.4, "type": "AVERAGE_CPR"},
                "weekly_cpr": {"pivot": round(cur_price, 2), "tc": round(cur_price * 1.005, 2), "bc": round(cur_price * 0.995, 2), "width_pct": 1.0, "type": "AVERAGE_CPR"},
                "multi_day_extremes": {
                    "pdh": round(cur_price * 1.008, 2), "pdl": round(cur_price * 0.992, 2), "pdc": round(cur_price, 2),
                    "high_2d": round(cur_price * 1.012, 2), "low_2d": round(cur_price * 0.988, 2),
                    "high_3d": round(cur_price * 1.015, 2), "low_3d": round(cur_price * 0.985, 2),
                    "high_5d": round(cur_price * 1.020, 2), "low_5d": round(cur_price * 0.980, 2),
                    "high_10d": round(cur_price * 1.030, 2), "low_10d": round(cur_price * 0.970, 2),
                    "high_20d": round(cur_price * 1.045, 2), "low_20d": round(cur_price * 0.955, 2),
                },
                "resistance_hierarchy": [],
                "support_hierarchy": [],
                "confluence_bias": "CONSOLIDATING_BETWEEN_SR",
                "description": f"Price (₹{cur_price:.2f}) trading within baseline support & resistance bands.",
            }

        # 2. Extract Daily Bar Levels
        prev_day = df_daily_clean.iloc[-2] if len(df_daily_clean) >= 2 else df_daily_clean.iloc[-1]
        pdh = cls._clean_float(prev_day["high"], cur_price)
        pdl = cls._clean_float(prev_day["low"], cur_price)
        pdc = cls._clean_float(prev_day["close"], cur_price)
        pdo = cls._clean_float(prev_day["open"], cur_price)

        # Multi-day high & low ranges
        h2d = cls._clean_float(df_daily_clean["high"].tail(3).iloc[:-1].max() if len(df_daily_clean) >= 3 else pdh)
        l2d = cls._clean_float(df_daily_clean["low"].tail(3).iloc[:-1].min() if len(df_daily_clean) >= 3 else pdl)
        h3d = cls._clean_float(df_daily_clean["high"].tail(4).iloc[:-1].max() if len(df_daily_clean) >= 4 else h2d)
        l3d = cls._clean_float(df_daily_clean["low"].tail(4).iloc[:-1].min() if len(df_daily_clean) >= 4 else l2d)
        h5d = cls._clean_float(df_daily_clean["high"].tail(6).iloc[:-1].max() if len(df_daily_clean) >= 6 else h3d)
        l5d = cls._clean_float(df_daily_clean["low"].tail(6).iloc[:-1].min() if len(df_daily_clean) >= 6 else l3d)
        h10d = cls._clean_float(df_daily_clean["high"].tail(11).iloc[:-1].max() if len(df_daily_clean) >= 11 else h5d)
        l10d = cls._clean_float(df_daily_clean["low"].tail(11).iloc[:-1].min() if len(df_daily_clean) >= 11 else l5d)
        h20d = cls._clean_float(df_daily_clean["high"].tail(21).iloc[:-1].max() if len(df_daily_clean) >= 21 else h10d)
        l20d = cls._clean_float(df_daily_clean["low"].tail(21).iloc[:-1].min() if len(df_daily_clean) >= 21 else l10d)

        # 3. Daily Pivots & CPR
        p_d = (pdh + pdl + pdc) / 3.0
        bc_d = (pdh + pdl) / 2.0
        tc_d = (p_d - bc_d) + p_d
        top_cpr_d = max(tc_d, bc_d)
        bot_cpr_d = min(tc_d, bc_d)
        cpr_w_pct_d = round((abs(tc_d - bc_d) / (p_d + 1e-9)) * 100, 3)
        cpr_type_d = "NARROW_CPR" if cpr_w_pct_d <= 0.25 else ("WIDE_CPR" if cpr_w_pct_d >= 0.6 else "AVERAGE_CPR")

        r1_d = (2.0 * p_d) - pdl
        s1_d = (2.0 * p_d) - pdh
        r2_d = p_d + (pdh - pdl)
        s2_d = p_d - (pdh - pdl)
        r3_d = pdh + 2.0 * (p_d - pdl)
        s3_d = pdl - 2.0 * (pdh - p_d)
        r4_d = r3_d + (pdh - pdl)
        s4_d = s3_d - (pdh - pdl)

        cam_d = cls.calculate_camarilla_pivots(pdh, pdl, pdc, cur_price)

        # 4. Weekly Pivots & CPR
        if df_weekly_clean is not None and len(df_weekly_clean) >= 2:
            prev_wk = df_weekly_clean.iloc[-2]
            pwh = cls._clean_float(prev_wk["high"], pdh)
            pwl = cls._clean_float(prev_wk["low"], pdl)
            pwc = cls._clean_float(prev_wk["close"], pdc)
        else:
            pwh, pwl, pwc = h5d, l5d, pdc

        p_w = (pwh + pwl + pwc) / 3.0
        bc_w = (pwh + pwl) / 2.0
        tc_w = (p_w - bc_w) + p_w
        top_cpr_w = max(tc_w, bc_w)
        bot_cpr_w = min(tc_w, bc_w)
        cpr_w_pct_w = round((abs(tc_w - bc_w) / (p_w + 1e-9)) * 100, 3)
        cpr_type_w = "NARROW_CPR" if cpr_w_pct_w <= 0.50 else ("WIDE_CPR" if cpr_w_pct_w >= 1.2 else "AVERAGE_CPR")

        r1_w = (2.0 * p_w) - pwl
        s1_w = (2.0 * p_w) - pwh
        r2_w = p_w + (pwh - pwl)
        s2_w = p_w - (pwh - pwl)
        cam_w = cls.calculate_camarilla_pivots(pwh, pwl, pwc, cur_price)

        # 5. Macro Fibonacci & Swing Fractal Clusters
        macro_fib = cls.calculate_macro_fibonacci(df_daily_clean, lookback=45)
        fib_levels = macro_fib.get("levels", {})

        swing_clusters = cls.calculate_swing_clusters(df_daily_clean, lookback=45, cluster_tolerance_pct=0.25)
        supply_zones = swing_clusters.get("supply_clusters", [])
        demand_zones = swing_clusters.get("demand_clusters", [])

        # 6. Raw Level Pool Aggregation
        raw_candidates: List[Tuple[str, float]] = [
            ("PDH (Prev Day High)", pdh),
            ("PDL (Prev Day Low)", pdl),
            ("PDC (Prev Day Close)", pdc),
            ("2-Day High", h2d),
            ("2-Day Low", l2d),
            ("3-Day High", h3d),
            ("3-Day Low", l3d),
            ("5-Day / Prev Week High", pwh),
            ("5-Day / Prev Week Low", pwl),
            ("10-Day High", h10d),
            ("10-Day Low", l10d),
            ("20-Day / Prev Month High", h20d),
            ("20-Day / Prev Month Low", l20d),
            ("Daily Pivot (P)", p_d),
            ("Daily R1", r1_d),
            ("Daily R2", r2_d),
            ("Daily R3", r3_d),
            ("Daily R4", r4_d),
            ("Daily S1", s1_d),
            ("Daily S2", s2_d),
            ("Daily S3", s3_d),
            ("Daily S4", s4_d),
            ("Daily CPR TC", tc_d),
            ("Daily CPR BC", bc_d),
            ("Daily Camarilla H3 (Short Resistance)", cam_d["h3"]),
            ("Daily Camarilla H4 (Breakout Target)", cam_d["h4"]),
            ("Daily Camarilla H5 (Exhaustion)", cam_d["h5"]),
            ("Daily Camarilla L3 (Long Bounce Support)", cam_d["l3"]),
            ("Daily Camarilla L4 (Breakdown Target)", cam_d["l4"]),
            ("Daily Camarilla L5 (Exhaustion)", cam_d["l5"]),
            ("Weekly Pivot (P_w)", p_w),
            ("Weekly R1", r1_w),
            ("Weekly R2", r2_w),
            ("Weekly S1", s1_w),
            ("Weekly S2", s2_w),
            ("Weekly CPR TC", tc_w),
            ("Weekly CPR BC", bc_w),
            ("Weekly Camarilla H4", cam_w["h4"]),
            ("Weekly Camarilla L4", cam_w["l4"]),
        ]

        for fib_k, fib_v in fib_levels.items():
            raw_candidates.append((f"Macro {fib_k}", float(fib_v)))

        for sz in supply_zones:
            raw_candidates.append((f"Supply Zone ({sz['touches']}x tested)", float(sz["price"])))
        for dz in demand_zones:
            raw_candidates.append((f"Demand Zone ({dz['touches']}x tested)", float(dz["price"])))

        # 7. Deduplication & Confluence Grouping into Resistances & Supports
        raw_resistances = [c for c in raw_candidates if c[1] > cur_price * 1.0005]
        raw_supports = [c for c in raw_candidates if c[1] < cur_price * 0.9995]

        # Group resistances
        grouped_res: List[Dict[str, Any]] = []
        for tag, p in sorted(raw_resistances, key=lambda x: x[1]):
            matched = False
            for gr in grouped_res:
                if abs(p - gr["price"]) / (gr["price"] + 1e-9) <= 0.0015:  # Within 0.15%
                    gr["tags"].append(tag)
                    gr["price"] = round((gr["price"] * (len(gr["tags"]) - 1) + p) / len(gr["tags"]), 2)
                    matched = True
                    break
            if not matched:
                grouped_res.append({"price": round(p, 2), "tags": [tag]})

        # Group supports
        grouped_supp: List[Dict[str, Any]] = []
        for tag, p in sorted(raw_supports, key=lambda x: x[1], reverse=True):
            matched = False
            for gs in grouped_supp:
                if abs(p - gs["price"]) / (gs["price"] + 1e-9) <= 0.0015:  # Within 0.15%
                    gs["tags"].append(tag)
                    gs["price"] = round((gs["price"] * (len(gs["tags"]) - 1) + p) / len(gs["tags"]), 2)
                    matched = True
                    break
            if not matched:
                grouped_supp.append({"price": round(p, 2), "tags": [tag]})

        # Format final resistance hierarchy (sorted ascending by distance from spot)
        res_hierarchy: List[Dict[str, Any]] = []
        for r in sorted(grouped_res, key=lambda x: x["price"]):
            dist_pts = round(r["price"] - cur_price, 2)
            dist_pct = round((dist_pts / (cur_price + 1e-9)) * 100, 2)
            c_count = len(r["tags"])
            strength = "VERY_STRONG" if c_count >= 3 else ("STRONG" if c_count == 2 else "MODERATE")
            res_hierarchy.append({
                "price": r["price"],
                "distance_pts": dist_pts,
                "distance_pct": dist_pct,
                "confluence_count": c_count,
                "strength": strength,
                "tags": r["tags"][:4],
                "primary_label": r["tags"][0],
            })

        # Format final support hierarchy (sorted descending by price, closest to spot first)
        supp_hierarchy: List[Dict[str, Any]] = []
        for s in sorted(grouped_supp, key=lambda x: x["price"], reverse=True):
            dist_pts = round(cur_price - s["price"], 2)
            dist_pct = round((dist_pts / (cur_price + 1e-9)) * 100, 2)
            c_count = len(s["tags"])
            strength = "VERY_STRONG" if c_count >= 3 else ("STRONG" if c_count == 2 else "MODERATE")
            supp_hierarchy.append({
                "price": s["price"],
                "distance_pts": dist_pts,
                "distance_pct": dist_pct,
                "confluence_count": c_count,
                "strength": strength,
                "tags": s["tags"][:4],
                "primary_label": s["tags"][0],
            })

        # Extract primary R1/R2/R3 and S1/S2/S3
        r_near_obj = res_hierarchy[0] if res_hierarchy else {"price": round(cur_price * 1.008, 2), "distance_pts": round(cur_price * 0.008, 2), "distance_pct": 0.8, "tags": ["R1 Resistance"], "strength": "MODERATE", "primary_label": "R1 Resistance"}
        r_mid_obj = res_hierarchy[1] if len(res_hierarchy) > 1 else {"price": round(r_near_obj["price"] * 1.008, 2), "distance_pts": round(cur_price * 0.016, 2), "distance_pct": 1.6, "tags": ["R2 Resistance"], "strength": "MODERATE", "primary_label": "R2 Resistance"}
        r_major_obj = res_hierarchy[2] if len(res_hierarchy) > 2 else {"price": round(r_mid_obj["price"] * 1.008, 2), "distance_pts": round(cur_price * 0.024, 2), "distance_pct": 2.4, "tags": ["R3 Major Ceiling"], "strength": "STRONG", "primary_label": "R3 Major Ceiling"}

        s_near_obj = supp_hierarchy[0] if supp_hierarchy else {"price": round(cur_price * 0.992, 2), "distance_pts": round(cur_price * 0.008, 2), "distance_pct": 0.8, "tags": ["S1 Support"], "strength": "MODERATE", "primary_label": "S1 Support"}
        s_mid_obj = supp_hierarchy[1] if len(supp_hierarchy) > 1 else {"price": round(s_near_obj["price"] * 0.992, 2), "distance_pts": round(cur_price * 0.016, 2), "distance_pct": 1.6, "tags": ["S2 Support"], "strength": "MODERATE", "primary_label": "S2 Support"}
        s_major_obj = supp_hierarchy[2] if len(supp_hierarchy) > 2 else {"price": round(s_mid_obj["price"] * 0.992, 2), "distance_pts": round(cur_price * 0.024, 2), "distance_pct": 2.4, "tags": ["S3 Major Floor"], "strength": "STRONG", "primary_label": "S3 Major Floor"}

        # 8. Structural Confluence Bias Analysis
        r_lbl = r_near_obj.get("primary_label", "R1 Resistance")
        s_lbl = s_near_obj.get("primary_label", "S1 Support")
        r_tags = ", ".join(r_near_obj.get("tags", [r_lbl]))
        s_tags = ", ".join(s_near_obj.get("tags", [s_lbl]))

        if cur_price >= pdh and cur_price >= top_cpr_d:
            conf_bias = "STRONG_BULLISH_EXPANSION"
            conf_desc = f"Price (₹{cur_price:.2f}) broke above PDH (₹{pdh:.2f}) & Daily CPR (₹{top_cpr_d:.2f}). Immediate overhead hurdle is {r_lbl} at ₹{r_near_obj['price']:.2f} (+{r_near_obj['distance_pts']} pts)."
        elif cur_price <= pdl and cur_price <= bot_cpr_d:
            conf_bias = "STRONG_BEARISH_BREAKDOWN"
            conf_desc = f"Price (₹{cur_price:.2f}) broke below PDL (₹{pdl:.2f}) & Daily CPR (₹{bot_cpr_d:.2f}). Immediate underlying floor is {s_lbl} at ₹{s_near_obj['price']:.2f} (-{s_near_obj['distance_pts']} pts)."
        elif s_near_obj.get("distance_pct", 1.0) <= 0.20:
            conf_bias = "TESTING_KEY_SUPPORT_BOUNCE"
            conf_desc = f"Price (₹{cur_price:.2f}) holding tightly above key Support {s_lbl} at ₹{s_near_obj['price']:.2f} (Confluence: {s_tags})."
        elif r_near_obj.get("distance_pct", 1.0) <= 0.20:
            conf_bias = "TESTING_KEY_RESISTANCE_REJECTION"
            conf_desc = f"Price (₹{cur_price:.2f}) testing key Overhead Resistance {r_lbl} at ₹{r_near_obj['price']:.2f} (Confluence: {r_tags})."
        else:
            conf_bias = "CONSOLIDATING_BETWEEN_SR"
            conf_desc = f"Price (₹{cur_price:.2f}) trading inside range between Support ₹{s_near_obj['price']:.2f} ({s_lbl}) and Resistance ₹{r_near_obj['price']:.2f} ({r_lbl})."

        return {
            "symbol": symbol or "UNKNOWN",
            "current_price": round(cur_price, 2),
            "nearest_resistance": r_near_obj,
            "nearest_support": s_near_obj,
            "r_near": r_near_obj["price"],
            "r_mid": r_mid_obj["price"],
            "r_major": r_major_obj["price"],
            "s_near": s_near_obj["price"],
            "s_mid": s_mid_obj["price"],
            "s_major": s_major_obj["price"],
            "daily_cpr": {
                "pivot": round(p_d, 2),
                "tc": round(top_cpr_d, 2),
                "bc": round(bot_cpr_d, 2),
                "width_pct": cpr_w_pct_d,
                "cpr_type": cpr_type_d,
            },
            "weekly_cpr": {
                "pivot": round(p_w, 2),
                "tc": round(top_cpr_w, 2),
                "bc": round(bot_cpr_w, 2),
                "width_pct": cpr_w_pct_w,
                "cpr_type": cpr_type_w,
            },
            "camarilla_daily": cam_d,
            "camarilla_weekly": cam_w,
            "multi_day_extremes": {
                "pdh": round(pdh, 2),
                "pdl": round(pdl, 2),
                "pdc": round(pdc, 2),
                "pdo": round(pdo, 2),
                "high_2d": round(h2d, 2),
                "low_2d": round(l2d, 2),
                "high_3d": round(h3d, 2),
                "low_3d": round(l3d, 2),
                "high_5d": round(h5d, 2),
                "low_5d": round(l5d, 2),
                "high_10d": round(h10d, 2),
                "low_10d": round(l10d, 2),
                "high_20d": round(h20d, 2),
                "low_20d": round(l20d, 2),
            },
            "macro_fibonacci": macro_fib,
            "swing_clusters": swing_clusters,
            "resistance_hierarchy": res_hierarchy[:8],
            "support_hierarchy": supp_hierarchy[:8],
            "confluence_bias": conf_bias,
            "description": conf_desc,
        }

    def run_full_pattern_scan(
        self,
        df: pd.DataFrame,
        df_daily: Optional[pd.DataFrame] = None,
        df_weekly: Optional[pd.DataFrame] = None,
        symbol: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executes unified scan for Candlesticks, Day Levels & CPR, Volumes/VWAP, SMC Order Blocks,
        W/M Patterns, Price Action Momentum, and Multi-Timeframe Support & Resistance.
        """
        candles = self.detect_candlestick_patterns(df)
        day_levels = self.calculate_day_levels(df)
        volume_analysis = self.analyze_volume(df)
        order_blocks = self.detect_order_blocks(df)
        wm_patterns = self.detect_wm_patterns(df)
        pa_momentum = self.analyze_price_action_momentum(df)
        multi_sr = self.calculate_multi_timeframe_sr(
            df_intraday=df,
            df_daily=df_daily,
            df_weekly=df_weekly,
            symbol=symbol,
        )

        return {
            "candle_patterns": candles,
            "day_levels": day_levels,
            "volume_analysis": volume_analysis,
            "order_blocks": order_blocks,
            "wm_patterns": wm_patterns,
            "price_action_momentum": pa_momentum,
            "multi_timeframe_sr": multi_sr,
        }


pattern_service = PatternService()
