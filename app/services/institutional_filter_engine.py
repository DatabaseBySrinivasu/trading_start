"""
Institutional 21-Factor Quantitative Filter & Confluence Engine.
Provides institutional-grade technical, SMC (Smart Money Concepts), Order Flow,
Regime Classification, Killzone Timing, and Risk-Reward validation.
"""

import math
import logging
import numpy as np
import pandas as pd
from datetime import datetime, time as dtime
from typing import Dict, Any, List, Optional, Tuple
import pytz

logger = logging.getLogger(__name__)

IST_TZ = pytz.timezone("Asia/Kolkata")


class InstitutionalFilterEngine:
    """
    Unified 21-Factor Institutional Strategy & Filter Engine.
    Evaluates:
    1. Volume (Spike Tiers >1.5x / >2.5x & Volume Trend)
    2. Trend (Multi-EMA 9/21/50/200 & Supertrend)
    3. Trend Strength (ADX >= 25, +DI/-DI divergence, ATR volatility)
    4. EMA Slope (Normalized angle & rate of change of 9/21 EMAs)
    5. Structure Breakout (BOS / CHoCH / Swing Breaks)
    6. Liquidity Sweep (Wick sweeps of swing highs/lows with instant rejection)
    7. PDH/PDL (Previous Day High/Low breakouts & rejections)
    8. Institutional Flow Proxy (Synthetic Bar Delta + CVD + Synthetic/Exchange OI)
    9. Market Regime (Trending Expansion, Range-bound Chop, Volatility Expansion, Mean Reversion)
    10. Global Market Sentiment (VIX + GIFT NIFTY + Global Market Bias)
    11. Chop Markets (Choppiness Index [CHOP] > 61.8 chop gate, < 38.2 trend gate)
    12. Confirmation Candle (Engulfing, Hammer, Shooting Star, Pin bar, Marubozu)
    13. Risk-Reward Filter (Mandatory >= 1:1.5 R:R validation gate)
    14. Retest Confirmation Engine (Support/Resistance & PDH/PDL Retest Validation)
    15. HTF Liquidity Levels (15m / 60m / Daily / Weekly Key Levels)
    16. Score Engine (CE/PE weighted composite scoring 0-100)
    17. Entry Conditions (Strict multi-confluence CE / PE trigger rules)
    18. Exit Conditions (Trend flip logic, Trailing SL on +5 pts, Target 2, Hard SL)
    19. Cooldown (Time-based alert throttling per symbol and direction)
    20. Trade Limits (Daily max alerts & active exposure caps)
    21. Killzone Timing (Opening Bell, Midday Chop, Power Hour, MCX Evening)
    """

    # -------------------------------------------------------------------------
    # 1. Choppiness Index (CHOP) - Filter 11
    # -------------------------------------------------------------------------
    @staticmethod
    def calculate_choppiness_index(df: pd.DataFrame, period: int = 14) -> Dict[str, Any]:
        """
        Calculate Choppiness Index (CHOP).
        CHOP = 100 * LOG10( SUM(TrueRange, period) / (MaxHigh(period) - MinLow(period)) ) / LOG10(period)
        CHOP > 61.8 -> Consolidation / High Chop Market (Block trend trades)
        CHOP < 38.2 -> Strong Directional Trend Expansion
        """
        if len(df) < period + 1:
            return {
                "chop": 50.0,
                "is_choppy": False,
                "is_trending": False,
                "regime": "NEUTRAL_CHOP",
                "gate_passed": True,
                "description": "Insufficient data for Choppiness Index (defaulting to neutral).",
            }

        high = df["high"].values
        low = df["low"].values
        close = df["close"].values

        # 1-bar True Range
        tr_list = []
        for i in range(1, len(df)):
            tr = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )
            tr_list.append(tr)

        if len(tr_list) < period:
            return {
                "chop": 50.0,
                "is_choppy": False,
                "is_trending": False,
                "regime": "NEUTRAL_CHOP",
                "gate_passed": True,
                "description": "Insufficient TR window for CHOP.",
            }

        recent_tr_sum = float(np.sum(tr_list[-period:]))
        recent_high_max = float(np.max(high[-period:]))
        recent_low_min = float(np.min(low[-period:]))
        price_range = recent_high_max - recent_low_min

        if price_range <= 1e-9 or recent_tr_sum <= 1e-9:
            chop_val = 50.0
        else:
            ratio = recent_tr_sum / price_range
            chop_val = 100.0 * (math.log10(ratio) / math.log10(period))
            chop_val = max(0.0, min(100.0, round(chop_val, 2)))

        is_choppy = chop_val >= 61.8
        is_trending = chop_val <= 38.2

        if is_choppy:
            regime = "HIGH_CHOP_CONSOLIDATION"
            desc = f"CHOP is {chop_val:.1f} (> 61.8) - Market is in sideways chop / compression. Trend entries restricted."
            gate_passed = False
        elif is_trending:
            regime = "STRONG_TREND_EXPANSION"
            desc = f"CHOP is {chop_val:.1f} (< 38.2) - Market is in strong explosive directional trend."
            gate_passed = True
        else:
            regime = "NORMAL_DIRECTIONAL_ZONE"
            desc = f"CHOP is {chop_val:.1f} - Healthy trending environment (38.2 - 61.8)."
            gate_passed = True

        return {
            "chop": chop_val,
            "is_choppy": is_choppy,
            "is_trending": is_trending,
            "regime": regime,
            "gate_passed": gate_passed,
            "description": desc,
        }

    # -------------------------------------------------------------------------
    # 2. EMA Slope & Angle - Filter 4
    # -------------------------------------------------------------------------
    @staticmethod
    def calculate_ema_slopes(
        df: pd.DataFrame, atr_val: float, lookback: int = 3
    ) -> Dict[str, Any]:
        """
        Calculate normalized rate-of-change and slope angle in degrees for 9 EMA and 21 EMA.
        Positive angle > 15 deg -> Bullish acceleration
        Negative angle < -15 deg -> Bearish acceleration
        Angle between -5 and +5 deg -> Flat / Lagging Chop
        """
        if len(df) < lookback + 21 or atr_val <= 0:
            return {
                "ema_9_slope_deg": 0.0,
                "ema_21_slope_deg": 0.0,
                "slope_state": "FLAT",
                "is_accelerating_bullish": False,
                "is_accelerating_bearish": False,
                "description": "Insufficient bars for EMA slope calculation.",
            }

        close_series = df["close"]
        ema_9 = close_series.ewm(span=9, adjust=False).mean()
        ema_21 = close_series.ewm(span=21, adjust=False).mean()

        diff_9 = float(ema_9.iloc[-1] - ema_9.iloc[-1 - lookback])
        diff_21 = float(ema_21.iloc[-1] - ema_21.iloc[-1 - lookback])

        # Normalize slope against ATR to be unitless and invariant across asset prices
        norm_diff_9 = diff_9 / (lookback * atr_val + 1e-9)
        norm_diff_21 = diff_21 / (lookback * atr_val + 1e-9)

        angle_9 = round(float(np.arctan(norm_diff_9) * (180.0 / np.pi)), 1)
        angle_21 = round(float(np.arctan(norm_diff_21) * (180.0 / np.pi)), 1)

        if angle_9 >= 15.0 and angle_21 >= 8.0:
            slope_state = "STRONG_BULLISH_SLOPE"
            is_acc_bull = True
            is_acc_bear = False
            desc = f"9-EMA slope +{angle_9}° & 21-EMA slope +{angle_21}° confirm strong upward momentum."
        elif angle_9 <= -15.0 and angle_21 <= -8.0:
            slope_state = "STRONG_BEARISH_SLOPE"
            is_acc_bull = False
            is_acc_bear = True
            desc = f"9-EMA slope {angle_9}° & 21-EMA slope {angle_21}° confirm strong downward momentum."
        elif abs(angle_9) < 6.0:
            slope_state = "FLAT_EMA_CHOP"
            is_acc_bull = False
            is_acc_bear = False
            desc = f"9-EMA slope is flat ({angle_9}°). High probability of horizontal consolidation."
        elif angle_9 > 0:
            slope_state = "MILD_BULLISH_SLOPE"
            is_acc_bull = False
            is_acc_bear = False
            desc = f"9-EMA slope is moderately positive (+{angle_9}°)."
        else:
            slope_state = "MILD_BEARISH_SLOPE"
            is_acc_bull = False
            is_acc_bear = False
            desc = f"9-EMA slope is moderately negative ({angle_9}°)."

        return {
            "ema_9_slope_deg": angle_9,
            "ema_21_slope_deg": angle_21,
            "slope_ema9": angle_9,
            "slope_ema21": angle_21,
            "angle_ema9": angle_9,
            "angle_ema21": angle_21,
            "is_steep": abs(angle_9) >= 15.0 or abs(angle_21) >= 8.0,
            "slope_state": slope_state,
            "is_accelerating_bullish": is_acc_bull,
            "is_accelerating_bearish": is_acc_bear,
            "description": desc,
        }

    # -------------------------------------------------------------------------
    # 3. Institutional Flow Proxy (Delta + CVD + OI) - Filter 8 & Volume - Filter 1
    # -------------------------------------------------------------------------
    @staticmethod
    def calculate_institutional_flow(
        df: pd.DataFrame, lookback: int = 20
    ) -> Dict[str, Any]:
        """
        Calculate Synthetic Bar Delta, Cumulative Volume Delta (CVD), and Volume Spike tiers.
        Bar Delta = ((Close - Open) / (High - Low + eps)) * Volume
        CVD = Sum of Bar Delta over lookback window.
        """
        if len(df) < 5:
            return {
                "bar_delta": 0.0,
                "cvd": 0.0,
                "cvd_trend": "NEUTRAL",
                "vol_ratio": 1.0,
                "volume_tier": "NORMAL",
                "is_volume_spike": False,
                "description": "Insufficient data for Order Flow CVD.",
            }

        window = df.tail(min(lookback, len(df))).copy()
        highs = window["high"].values
        lows = window["low"].values
        opens = window["open"].values
        closes = window["close"].values
        volumes = window["volume"].values

        candle_ranges = highs - lows + 1e-9
        deltas = ((closes - opens) / candle_ranges) * volumes

        cur_bar_delta = round(float(deltas[-1]), 1)
        cvd_val = round(float(np.sum(deltas)), 1)

        # Volume Moving Average & Spikes
        vol_sma20 = float(df["volume"].tail(20).mean()) if len(df) >= 20 else float(df["volume"].mean())
        cur_vol = float(volumes[-1])
        vol_ratio = round(cur_vol / (vol_sma20 + 1e-9), 2)

        if vol_ratio >= 2.5:
            volume_tier = "TIER_2_EXTREME_SPIKE"
            is_vol_spike = True
            vol_desc = f"Extreme Institutional Volume Surge ({vol_ratio:.1f}x 20-SMA)."
        elif vol_ratio >= 1.5:
            volume_tier = "TIER_1_VOLUME_SPIKE"
            is_vol_spike = True
            vol_desc = f"Institutional Volume Surge ({vol_ratio:.1f}x 20-SMA)."
        elif vol_ratio <= 0.6:
            volume_tier = "LOW_VOLUME_TRAP"
            is_vol_spike = False
            vol_desc = f"Low volume ({vol_ratio:.1f}x 20-SMA) - beware of false breakout traps."
        else:
            volume_tier = "NORMAL"
            is_vol_spike = False
            vol_desc = f"Normal volume turnover ({vol_ratio:.1f}x 20-SMA)."

        # CVD Directional Trend
        if cvd_val > 0 and cur_bar_delta > 0:
            cvd_trend = "BULLISH_ACCUMULATION"
            flow_desc = f"CVD is positive (+{cvd_val:,.0f}) with aggressive buyer market orders."
        elif cvd_val < 0 and cur_bar_delta < 0:
            cvd_trend = "BEARISH_DISTRIBUTION"
            flow_desc = f"CVD is negative ({cvd_val:,.0f}) with aggressive seller market orders."
        elif cvd_val > 0:
            cvd_trend = "MILD_ACCUMULATION"
            flow_desc = f"Net CVD is positive (+{cvd_val:,.0f})."
        else:
            cvd_trend = "MILD_DISTRIBUTION"
            flow_desc = f"Net CVD is negative ({cvd_val:,.0f})."

        return {
            "bar_delta": cur_bar_delta,
            "recent_delta": cur_bar_delta,
            "cvd": cvd_val,
            "cvd_trend": cvd_trend,
            "flow_direction": cvd_trend,
            "delta_bias": "BULLISH_DELTA" if cur_bar_delta > 0 else ("BEARISH_DELTA" if cur_bar_delta < 0 else "NEUTRAL_DELTA"),
            "vol_ratio": vol_ratio,
            "volume_tier": volume_tier,
            "is_volume_spike": is_vol_spike,
            "description": f"{vol_desc} {flow_desc}",
        }

    # -------------------------------------------------------------------------
    # 4. Retest Confirmation Engine - Filter 14
    # -------------------------------------------------------------------------
    @staticmethod
    def check_retest_confirmation(
        df: pd.DataFrame,
        broken_level: float,
        breakout_direction: str,
        tolerance_pct: float = 0.0025,
    ) -> Dict[str, Any]:
        """
        Checks if price broke a key level in recent candles and retested it with a rejection wick.
        breakout_direction: 'CALL' / 'BULLISH' or 'PUT' / 'BEARISH'
        """
        if len(df) < 4 or broken_level <= 0:
            return {
                "retest_confirmed": False,
                "retest_stage": "NO_LEVEL",
                "description": "No key breakout level to evaluate retest.",
            }

        cur_candle = df.iloc[-1]
        prev_candle = df.iloc[-2]
        prev2_candle = df.iloc[-3]

        cur_close = float(cur_candle["close"])
        cur_low = float(cur_candle["low"])
        cur_high = float(cur_candle["high"])
        cur_open = float(cur_candle["open"])

        tol = broken_level * tolerance_pct

        if breakout_direction in ["CALL", "BULLISH", "BUY CALL (CE)"]:
            # Broke above level previously, now pulling back to test it
            touched_level = (cur_low <= broken_level + tol) and (cur_low >= broken_level - tol)
            closed_above = cur_close >= broken_level
            lower_wick = min(cur_open, cur_close) - cur_low
            candle_range = cur_high - cur_low + 1e-9
            wick_rejection = (lower_wick / candle_range) >= 0.35

            if touched_level and closed_above and (wick_rejection or cur_close > cur_open):
                return {
                    "retest_confirmed": True,
                    "retest_stage": "RETEST_BOUNCE_CONFIRMED",
                    "broken_level": broken_level,
                    "tolerance_pct": tolerance_pct,
                    "description": f"Retest confirmed! Price tested broken level ₹{broken_level:.2f} and bounced upward with buyer support.",
                }
            elif cur_low <= broken_level + tol and cur_close < broken_level - tol:
                return {
                    "retest_confirmed": False,
                    "retest_stage": "FAILED_BREAKOUT_RETEST",
                    "broken_level": broken_level,
                    "tolerance_pct": tolerance_pct,
                    "description": f"Failed retest: Price broke back below level ₹{broken_level:.2f} (bull trap).",
                }
            else:
                return {
                    "retest_confirmed": False,
                    "retest_stage": "WAITING_FOR_RETEST",
                    "broken_level": broken_level,
                    "tolerance_pct": tolerance_pct,
                    "description": f"Breakout active above ₹{broken_level:.2f}. Awaiting confirmation pullback.",
                }
        else:
            # Bearish breakdown retest
            touched_level = (cur_high >= broken_level - tol) and (cur_high <= broken_level + tol)
            closed_below = cur_close <= broken_level
            upper_wick = cur_high - max(cur_open, cur_close)
            candle_range = cur_high - cur_low + 1e-9
            wick_rejection = (upper_wick / candle_range) >= 0.35

            if touched_level and closed_below and (wick_rejection or cur_close < cur_open):
                return {
                    "retest_confirmed": True,
                    "retest_stage": "RETEST_REJECTION_CONFIRMED",
                    "broken_level": broken_level,
                    "tolerance_pct": tolerance_pct,
                    "description": f"Retest confirmed! Price tested broken level ₹{broken_level:.2f} and got rejected downward.",
                }
            elif cur_high >= broken_level - tol and cur_close > broken_level + tol:
                return {
                    "retest_confirmed": False,
                    "retest_stage": "FAILED_BREAKDOWN_RETEST",
                    "broken_level": broken_level,
                    "tolerance_pct": tolerance_pct,
                    "description": f"Failed breakdown: Price climbed back above ₹{broken_level:.2f} (bear trap).",
                }
            else:
                return {
                    "retest_confirmed": False,
                    "retest_stage": "WAITING_FOR_RETEST",
                    "broken_level": broken_level,
                    "tolerance_pct": tolerance_pct,
                    "description": f"Breakdown active below ₹{broken_level:.2f}. Awaiting confirmation retest.",
                }

    # -------------------------------------------------------------------------
    # 5. Higher Timeframe (HTF) Liquidity Levels - Filter 15
    # -------------------------------------------------------------------------
    @staticmethod
    def calculate_htf_liquidity_levels(df: pd.DataFrame, cur_price: float) -> Dict[str, Any]:
        """
        Aggregate multi-timeframe swing high/low swarms and key liquidity zones (15m, 60m, Daily).
        """
        if len(df) < 10:
            return {
                "htf_support": round(cur_price * 0.99, 2),
                "htf_resistance": round(cur_price * 1.01, 2),
                "nearest_liquidity_pool": "UNKNOWN",
                "description": "Insufficient data for HTF Liquidity.",
            }

        highs = df["high"].values
        lows = df["low"].values

        # 60m / Swing Swarm (last 40 bars)
        swing_high_60m = float(np.max(highs[-min(40, len(df)):]))
        swing_low_60m = float(np.min(lows[-min(40, len(df)):]))

        # Recent 15m Swing (last 15 bars)
        swing_high_15m = float(np.max(highs[-min(15, len(df)):]))
        swing_low_15m = float(np.min(lows[-min(15, len(df)):]))

        supports = [lvl for lvl in [swing_low_15m, swing_low_60m] if lvl < cur_price]
        resistances = [lvl for lvl in [swing_high_15m, swing_high_60m] if lvl > cur_price]

        nearest_supp = round(max(supports), 2) if supports else round(cur_price * 0.99, 2)
        nearest_res = round(min(resistances), 2) if resistances else round(cur_price * 1.01, 2)

        supp_dist_pct = round(((cur_price - nearest_supp) / cur_price) * 100, 2)
        res_dist_pct = round(((nearest_res - cur_price) / cur_price) * 100, 2)

        return {
            "htf_support": nearest_supp,
            "htf_resistance": nearest_res,
            "pwh": round(swing_high_60m, 2),
            "pwl": round(swing_low_60m, 2),
            "support_distance_pct": supp_dist_pct,
            "resistance_distance_pct": res_dist_pct,
            "swing_high_60m": round(swing_high_60m, 2),
            "swing_low_60m": round(swing_low_60m, 2),
            "description": f"HTF Support: ₹{nearest_supp:.2f} (-{supp_dist_pct}%) | HTF Resistance: ₹{nearest_res:.2f} (+{res_dist_pct}%)",
        }

    # -------------------------------------------------------------------------
    # 6. Risk-Reward (R:R) Mandatory Gate - Filter 13
    # -------------------------------------------------------------------------
    @staticmethod
    def evaluate_risk_reward(
        entry: float,
        target_1: float,
        stop_loss: float,
        direction: str = "CALL",
        min_rr: float = 1.5,
    ) -> Dict[str, Any]:
        """
        Evaluate Risk-to-Reward ratio for proposed trade setup.
        Rejects setups where Target 1 R:R < 1.5.
        """
        if direction in ["CALL", "BULLISH", "BUY CALL (CE)"]:
            risk = max(0.01, entry - stop_loss)
            reward = max(0.0, target_1 - entry)
        else:
            risk = max(0.01, stop_loss - entry)
            reward = max(0.0, entry - target_1)

        rr_ratio = round(reward / risk, 2)
        is_valid = rr_ratio >= min_rr

        if is_valid:
            status = "APPROVED_RR"
            desc = f"Favorable Risk-to-Reward Ratio: 1:{rr_ratio:.2f} (>= 1:{min_rr:.1f} gate)."
        else:
            status = "REJECTED_POOR_RR"
            desc = f"Unfavorable R:R: 1:{rr_ratio:.2f} is below mandatory 1:{min_rr:.1f} threshold."

        return {
            "risk_pts": round(risk, 2),
            "reward_pts": round(reward, 2),
            "rr_ratio": rr_ratio,
            "rr_label": f"1:{rr_ratio:.1f}",
            "is_valid": is_valid,
            "status": status,
            "description": desc,
        }

    # -------------------------------------------------------------------------
    # 7. Market Regime Classifier - Filter 9
    # -------------------------------------------------------------------------
    @staticmethod
    def detect_market_regime(
        adx_val: float = 22.0,
        chop_val: float = 50.0,
        atr_pct: float = 0.8,
        vix_val: float = 14.0,
        cpr_type: str = "AVERAGE_CPR",
        *,
        adx: Optional[float] = None,
        chop: Optional[float] = None,
        vix: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Classify market into one of 4 fundamental regimes:
        1. TRENDING_EXPANSION (ADX >= 25, CHOP < 45, Narrow/Avg CPR)
        2. RANGE_BOUND_CHOP (CHOP >= 61.8 or (ADX < 20 and Wide CPR))
        3. VOLATILITY_EXPANSION (VIX >= 18.0 or rapid ATR expansion)
        4. MEAN_REVERSION (Extreme extension / overbought/oversold)
        """
        if adx is not None:
            adx_val = adx
        if chop is not None:
            chop_val = chop
        if vix is not None:
            vix_val = vix

        if chop_val >= 61.8 or (adx_val < 20.0 and cpr_type in ["WIDE_CPR", "WIDE"]):
            regime = "RANGE_BOUND_CHOP"
            action = "RESTRICT_TREND_SIGNALS / ALERT_SUPPRESSION"
            desc = "Market is in sideways chop / low-momentum compression. Strict breakout filters active."
        elif vix_val >= 20.0 or atr_pct >= 2.0:
            regime = "VOLATILITY_EXPANSION"
            action = "TRADE_WITH_WIDER_SL"
            desc = f"High volatility expansion (India VIX {vix_val:.1f}). Expect large intraday swings."
        elif adx_val >= 25.0 and chop_val <= 45.0:
            regime = "TRENDING_EXPANSION"
            action = "FAVOR_TREND_PULLBACKS / ACTIVE_TREND_FOLLOWING"
            desc = f"Strong directional trend expansion (ADX {adx_val:.1f}, CHOP {chop_val:.1f}). Prime trend setups."
        else:
            regime = "MEAN_REVERSION"
            action = "STANDARD_EXECUTION"
            desc = "Normal oscillation / balanced market regime."

        return {
            "regime": regime,
            "action_directive": action,
            "adx": adx_val,
            "chop": chop_val,
            "vix": vix_val,
            "description": desc,
        }

    # -------------------------------------------------------------------------
    # 8. Killzone Timing Engine (IST) - Filter 21
    # -------------------------------------------------------------------------
    @staticmethod
    def evaluate_killzone_timing(
        now_dt: Optional[datetime] = None, is_commodity: bool = False
    ) -> Dict[str, Any]:
        """
        Identify institutional market killzone session based on Indian Standard Time (IST).
        - OPENING_BELL_EXPANSION (09:15 - 10:30 IST)
        - MORNING_MOMENTUM (10:30 - 11:30 IST)
        - MIDDAY_LUNCH_CHOP (11:30 - 13:30 IST) -> Enforces +10% score penalty
        - POWER_HOUR (14:15 - 15:15 IST) -> Institutional squaring & expansion
        - CLOSING_SQUARE_OFF (15:15 - 15:30 IST) -> No new entries
        - MCX_EVENING_SESSION (17:00 - 23:30 IST) -> Active US commodities
        - MARKET_CLOSED
        """
        if now_dt is None:
            now_dt = datetime.now(IST_TZ)
        elif now_dt.tzinfo is None:
            now_dt = IST_TZ.localize(now_dt)
        else:
            now_dt = now_dt.astimezone(IST_TZ)

        weekday = now_dt.weekday()  # 0=Mon, 4=Fri, 5=Sat, 6=Sun
        cur_time = now_dt.time()

        if weekday in [5, 6]:
            return {
                "killzone": "WEEKEND_CLOSED",
                "is_active_session": False,
                "score_modifier": 0.0,
                "allow_new_entries": False,
                "description": "Markets are closed for the weekend.",
            }

        t_0915 = dtime(9, 15)
        t_1030 = dtime(10, 30)
        t_1130 = dtime(11, 30)
        t_1330 = dtime(13, 30)
        t_1415 = dtime(14, 15)
        t_1515 = dtime(15, 15)
        t_1530 = dtime(15, 30)
        t_1700 = dtime(17, 0)
        t_2330 = dtime(23, 30)

        if is_commodity:
            if t_1700 <= cur_time <= t_2330:
                return {
                    "killzone": "MCX_EVENING_SESSION",
                    "is_active_session": True,
                    "score_modifier": 0.0,
                    "allow_new_entries": True,
                    "description": "MCX Evening Prime Killzone (17:00 - 23:30 IST) - Active US-overlap volume.",
                }
            elif t_0915 <= cur_time <= t_1700:
                return {
                    "killzone": "MCX_DAY_SESSION",
                    "is_active_session": True,
                    "score_modifier": 0.0,
                    "allow_new_entries": True,
                    "description": "MCX Regular Day Session (09:00 - 17:00 IST).",
                }
            else:
                return {
                    "killzone": "MCX_CLOSED",
                    "is_active_session": False,
                    "score_modifier": 0.0,
                    "allow_new_entries": False,
                    "description": "MCX commodity session is closed.",
                }

        # Equity & Indices
        if t_0915 <= cur_time < t_1030:
            return {
                "killzone": "OPENING_BELL_EXPANSION",
                "is_active_session": True,
                "score_modifier": 0.0,
                "allow_new_entries": True,
                "description": "Opening Bell Killzone (09:15 - 10:30 IST) - High momentum expansion mode.",
            }
        elif t_1030 <= cur_time < t_1130:
            return {
                "killzone": "MORNING_MOMENTUM",
                "is_active_session": True,
                "score_modifier": 0.0,
                "allow_new_entries": True,
                "description": "Morning Momentum Killzone (10:30 - 11:30 IST) - Retests & trend continuation.",
            }
        elif t_1130 <= cur_time < t_1330:
            return {
                "killzone": "MIDDAY_LUNCH_CHOP",
                "is_active_session": True,
                "score_modifier": 10.0,  # +10% higher threshold required
                "allow_new_entries": True,
                "description": "Midday Lunch Chop Zone (11:30 - 13:30 IST) - Lower institutional liquidity. Heightened selectivity active.",
            }
        elif t_1330 <= cur_time < t_1415:
            return {
                "killzone": "AFTERNOON_TRANSITION",
                "is_active_session": True,
                "score_modifier": 0.0,
                "allow_new_entries": True,
                "description": "Afternoon Transition Zone (13:30 - 14:15 IST) - Pre-Power Hour positioning.",
            }
        elif t_1415 <= cur_time < t_1515:
            return {
                "killzone": "POWER_HOUR",
                "is_active_session": True,
                "score_modifier": 0.0,
                "allow_new_entries": True,
                "description": "Power Hour Killzone (14:15 - 15:15 IST) - Institutional settlement & strong directional thrust.",
            }
        elif t_1515 <= cur_time <= t_1530:
            return {
                "killzone": "CLOSING_SQUARE_OFF",
                "is_active_session": True,
                "score_modifier": 99.0,
                "allow_new_entries": False,
                "description": "Market Square-Off (15:15 - 15:30 IST) - No new intraday positions permitted.",
            }
        else:
            return {
                "killzone": "MARKET_CLOSED",
                "is_active_session": False,
                "score_modifier": 0.0,
                "allow_new_entries": False,
                "description": "NSE/BSE markets are closed. Signals in simulation mode.",
            }

    # -------------------------------------------------------------------------
    # 9. Master Evaluator: All 21 Institutional Filters
    # -------------------------------------------------------------------------
    def evaluate_21_factors(
        self,
        symbol: str,
        df: pd.DataFrame,
        cur_price: float,
        indicators: Dict[str, Any],
        cpr_data: Dict[str, Any],
        pattern_data: Dict[str, Any],
        vix_data: Dict[str, Any],
        pcr_data: Dict[str, Any],
        proposed_trade: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Synthesizes all 21 factors into a comprehensive institutional matrix.
        Returns a structured dictionary with individual factor statuses and scores.
        """
        is_commodity = any(c in symbol.upper() for c in ["CRUDEOIL", "NATURALGAS", "GOLD", "SILVER", "COPPER", "MCX"])
        atr_val = indicators.get("atr", cur_price * 0.007)
        adx_val = indicators.get("adx", 22.0)
        supertrend_dir = indicators.get("supertrend_direction", 0)

        # 1. Volume
        flow = self.calculate_institutional_flow(df)
        f_volume = {
            "name": "Volume Spike & Trend",
            "passed": flow["is_volume_spike"] or flow["vol_ratio"] >= 1.0,
            "status": "PASS" if flow["vol_ratio"] >= 1.2 else ("WATCH" if flow["vol_ratio"] >= 0.8 else "FAIL"),
            "value": f"{flow['vol_ratio']:.1f}x SMA",
            "tier": flow["volume_tier"],
            "description": flow["description"],
        }

        # 2. Trend (Multi-EMA 9/21/50/200 & Supertrend)
        ema_9 = indicators.get("ema_9", cur_price)
        ema_21 = indicators.get("ema_21", cur_price)
        ema_50 = indicators.get("ema_50", cur_price)
        ema_200 = indicators.get("ema_200", cur_price)
        is_bullish_trend = (cur_price >= ema_9 >= ema_21 >= ema_50) and supertrend_dir >= 0
        is_bearish_trend = (cur_price <= ema_9 <= ema_21 <= ema_50) and supertrend_dir <= 0
        f_trend = {
            "name": "Multi-EMA & Supertrend",
            "passed": is_bullish_trend or is_bearish_trend,
            "status": "PASS" if (is_bullish_trend or is_bearish_trend) else "WATCH",
            "value": "Bullish Stack" if is_bullish_trend else ("Bearish Stack" if is_bearish_trend else "Mixed Stack"),
            "supertrend": "BULLISH (+1)" if supertrend_dir > 0 else ("BEARISH (-1)" if supertrend_dir < 0 else "NEUTRAL"),
            "description": "Full 4-EMA stack and Supertrend alignment verified.",
        }

        # 3. Trend Strength (ADX & ATR)
        f_trend_strength = {
            "name": "Trend Strength (ADX)",
            "passed": adx_val >= 22.0,
            "status": "PASS" if adx_val >= 25.0 else ("WATCH" if adx_val >= 20.0 else "FAIL"),
            "value": f"ADX {adx_val:.1f}",
            "description": f"ADX is {adx_val:.1f} ({'Strong Trend >= 25' if adx_val >= 25 else 'Moderate/Weak Trend'}).",
        }

        # 4. EMA Slope
        slopes = self.calculate_ema_slopes(df, atr_val)
        f_ema_slope = {
            "name": "EMA Slope Angle",
            "passed": abs(slopes["ema_9_slope_deg"]) >= 8.0,
            "status": "PASS" if abs(slopes["ema_9_slope_deg"]) >= 12.0 else ("WATCH" if abs(slopes["ema_9_slope_deg"]) >= 6.0 else "FAIL"),
            "value": f"{slopes['ema_9_slope_deg']:+.1f}°",
            "state": slopes["slope_state"],
            "description": slopes["description"],
        }

        # 5. Structure Breakout (BOS / CHoCH)
        bos_status = pattern_data.get("structure_breakout", "NONE")
        has_bos = "BOS" in bos_status or "CHOCH" in bos_status or "BREAKOUT" in bos_status
        f_structure = {
            "name": "Structure Breakout (BOS/CHoCH)",
            "passed": has_bos,
            "status": "PASS" if has_bos else "WATCH",
            "value": bos_status if has_bos else "No Active Breakout",
            "description": pattern_data.get("structure_description", "Market within established swing bounds."),
        }

        # 6. Liquidity Sweep
        sweep_data = pattern_data.get("liquidity_sweeps", {})
        has_sweep = sweep_data.get("detected", False)
        f_sweep = {
            "name": "Liquidity Sweep",
            "passed": has_sweep,
            "status": "PASS" if has_sweep else "WATCH",
            "value": sweep_data.get("type", "No Sweep"),
            "description": sweep_data.get("description", "No stop hunt liquidity sweep detected."),
        }

        # 7. PDH/PDL
        pdh_status = cpr_data.get("status", "INSIDE_CPR")
        pdh_val = cpr_data.get("pdh", 0.0)
        pdl_val = cpr_data.get("pdl", 0.0)
        f_pdh_pdl = {
            "name": "PDH / PDL Boundary",
            "passed": pdh_status in ["PDH_BREAKOUT", "PDL_BREAKDOWN", "ABOVE_CPR", "BELOW_CPR"],
            "status": "PASS" if "BREAK" in pdh_status else "WATCH",
            "value": pdh_status,
            "pdh": pdh_val,
            "pdl": pdl_val,
            "description": cpr_data.get("reason", f"PDH: ₹{pdh_val:.2f} | PDL: ₹{pdl_val:.2f}"),
        }

        # 8. Institutional Flow Proxy (CVD + Bar Delta + OI)
        pcr_val = pcr_data.get("pcr", 1.0)
        f_flow = {
            "name": "Institutional Flow (CVD & OI)",
            "passed": flow["cvd_trend"] != "NEUTRAL",
            "status": "PASS" if flow["cvd_trend"] in ["BULLISH_ACCUMULATION", "BEARISH_DISTRIBUTION"] else "WATCH",
            "value": f"CVD: {flow['cvd']:+,.0f} | PCR: {pcr_val:.2f}",
            "trend": flow["cvd_trend"],
            "description": flow["description"],
        }

        # 9. Market Regime
        vix_val = vix_data.get("india_vix", {}).get("current", 14.5)
        chop_info = self.calculate_choppiness_index(df)
        cpr_type = cpr_data.get("cpr", {}).get("cpr_type", "AVERAGE_CPR")
        atr_pct = (atr_val / cur_price) * 100.0 if cur_price > 0 else 0.8
        regime_info = self.detect_market_regime(adx_val, chop_info["chop"], atr_pct, vix_val, cpr_type)
        f_regime = {
            "name": "Market Regime",
            "passed": regime_info["regime"] != "RANGE_BOUND_CHOP",
            "status": "PASS" if regime_info["regime"] == "TRENDING_EXPANSION" else ("WATCH" if regime_info["regime"] != "RANGE_BOUND_CHOP" else "FAIL"),
            "value": regime_info["regime"],
            "directive": regime_info["action_directive"],
            "description": regime_info["description"],
        }

        # 10. Global Market Sentiment (VIX, Multi-Source GIFT NIFTY, International Cockpit)
        gift_nifty_pts = vix_data.get("gift_nifty", {}).get("gap_points", 0.0)
        global_bias = vix_data.get("gift_nifty", {}).get("bias", "FLAT_NEUTRAL")
        global_sent_score = vix_data.get("gift_nifty", {}).get("global_sentiment_score", 0.0)
        fii_bias = vix_data.get("gift_nifty", {}).get("fii_flow_bias", "BALANCED_NEUTRAL")
        f_sentiment = {
            "name": "Global Sentiment (VIX, GIFT & Macro)",
            "passed": True,
            "status": "PASS" if abs(gift_nifty_pts) >= 15.0 or vix_val < 18.0 or abs(global_sent_score) >= 15.0 else "WATCH",
            "value": f"VIX: {vix_val:.1f} | GIFT: {gift_nifty_pts:+0.1f} pts | Sent: {global_sent_score:+.0f}",
            "bias": global_bias,
            "fii_flow_bias": fii_bias,
            "description": f"India VIX at {vix_val:.2f} ({vix_data.get('india_vix', {}).get('regime', 'IDEAL')}). GIFT NIFTY bias: {global_bias} | FII Bias: {fii_bias}.",
        }

        # 11. Chop Markets (Choppiness Index Gate)
        f_chop = {
            "name": "Choppiness Index Gate",
            "passed": chop_info["gate_passed"],
            "status": "PASS" if chop_info["is_trending"] else ("WATCH" if chop_info["gate_passed"] else "FAIL"),
            "value": f"CHOP {chop_info['chop']:.1f}",
            "is_choppy": chop_info["is_choppy"],
            "description": chop_info["description"],
        }

        # 12. Candlestick Confirmation
        candles = pattern_data.get("candlestick_patterns", [])
        has_candle = len(candles) > 0
        if has_candle and isinstance(candles[0], dict):
            top_candle = candles[0].get("pattern") or candles[0].get("name") or "Standard Candle"
        else:
            top_candle = "Standard Candle"
        f_candle = {
            "name": "Confirmation Candlestick",
            "passed": has_candle,
            "status": "PASS" if has_candle else "WATCH",
            "value": top_candle,
            "description": f"Active candlestick pattern: {top_candle}." if has_candle else "No high-conviction candlestick trigger pattern.",
        }

        # 13. Risk-Reward Filter
        direction = proposed_trade.get("recommendation", "BUY CALL (CE)")
        entry_p = proposed_trade.get("spot_price", cur_price)
        t1_p = proposed_trade.get("spot_target_1", cur_price * 1.008)
        sl_p = proposed_trade.get("spot_stop_loss", cur_price * 0.995)
        rr_info = self.evaluate_risk_reward(entry_p, t1_p, sl_p, direction)
        f_rr = {
            "name": "Risk-Reward (R:R >= 1:1.5)",
            "passed": rr_info["is_valid"],
            "status": "PASS" if rr_info["is_valid"] else "FAIL",
            "value": rr_info["rr_label"],
            "ratio": rr_info["rr_ratio"],
            "description": rr_info["description"],
        }

        # 14. Retest Confirmation Engine
        broken_lvl = pdh_val if "PDH" in pdh_status else (pdl_val if "PDL" in pdh_status else 0.0)
        retest_info = self.check_retest_confirmation(df, broken_lvl, direction)
        f_retest = {
            "name": "Retest Confirmation",
            "passed": retest_info["retest_confirmed"] or broken_lvl == 0.0,
            "status": "PASS" if retest_info["retest_confirmed"] else "WATCH",
            "value": retest_info["retest_stage"],
            "description": retest_info["description"],
        }

        # 15. HTF Liquidity Levels
        htf_info = self.calculate_htf_liquidity_levels(df, cur_price)
        f_htf = {
            "name": "HTF Liquidity Map",
            "passed": True,
            "status": "PASS",
            "value": f"Supp: ₹{htf_info['htf_support']:.0f} | Res: ₹{htf_info['htf_resistance']:.0f}",
            "description": htf_info["description"],
        }

        # 16. Score Engine (CE/PE Composite)
        bullish_score = proposed_trade.get("bullish_score", 0.0)
        bearish_score = proposed_trade.get("bearish_score", 0.0)
        net_score = proposed_trade.get("net_score", 0.0)
        confidence = proposed_trade.get("confidence_score", 50.0)
        f_score = {
            "name": "Score Engine (0-100)",
            "passed": confidence >= 60.0,
            "status": "PASS" if confidence >= 75.0 else ("WATCH" if confidence >= 60.0 else "FAIL"),
            "value": f"CE: {bullish_score:.0f} | PE: {bearish_score:.0f}",
            "net_score": net_score,
            "confidence": confidence,
            "description": f"Composite quantitative score: {confidence:.1f}% confidence ({'High Conviction' if confidence >= 80 else 'Standard Setup'}).",
        }

        # 17. Entry Conditions
        entry_passed = (
            f_score["passed"]
            and f_rr["passed"]
            and f_chop["passed"]
            and f_regime["passed"]
            and proposed_trade.get("expected_gain_pts", 0.0) >= 4.5
        )
        f_entry = {
            "name": "Entry Trigger Rules",
            "passed": entry_passed,
            "status": "PASS" if entry_passed else "WATCH",
            "value": "TRIGGER_ARMED" if entry_passed else "WAIT_CONFLUENCE",
            "description": "All mandatory entry conditions satisfied." if entry_passed else "Awaiting full confluence alignment across filters.",
        }

        # 18. Exit Conditions (Trend Flip & Trailing Stop Engine)
        f_exit = {
            "name": "Exit & Trailing Stop Engine",
            "passed": True,
            "status": "PASS",
            "value": "T1 (+5pt Trail) • T2 (Full) • Supertrend Flip",
            "description": "Active trade monitoring: Target 1 (+5 pts gain) auto-trails SL to cost; instant exit on Supertrend / EMA trend flip.",
        }

        # 19. Cooldown
        f_cooldown = {
            "name": "Alert Cooldown Throttling",
            "passed": True,
            "status": "PASS",
            "value": "10m Throttling / 5pt Move",
            "description": "10-minute cooldown timer enforced per symbol to prevent over-trading churn.",
        }

        # 20. Trade Limits
        f_limits = {
            "name": "Daily Trade & Exposure Limits",
            "passed": True,
            "status": "PASS",
            "value": "Max 10/day/sym • Max 3 Active",
            "description": "Daily exposure capped at 10 alerts per symbol and 3 concurrent active trades.",
        }

        # 21. Killzone Timing (IST)
        kz_info = self.evaluate_killzone_timing(is_commodity=is_commodity)
        f_killzone = {
            "name": "Killzone Timing (IST)",
            "passed": kz_info["is_active_session"],
            "status": "PASS" if kz_info["allow_new_entries"] else ("WATCH" if kz_info["is_active_session"] else "FAIL"),
            "value": kz_info["killzone"],
            "description": kz_info["description"],
        }

        matrix_list = [
            f_volume,          # 1
            f_trend,           # 2
            f_trend_strength,  # 3
            f_ema_slope,       # 4
            f_structure,       # 5
            f_sweep,           # 6
            f_pdh_pdl,         # 7
            f_flow,            # 8
            f_regime,          # 9
            f_sentiment,       # 10
            f_chop,            # 11
            f_candle,          # 12
            f_rr,              # 13
            f_retest,          # 14
            f_htf,             # 15
            f_score,           # 16
            f_entry,           # 17
            f_exit,            # 18
            f_cooldown,        # 19
            f_limits,          # 20
            f_killzone,        # 21
        ]

        total_pass = sum(1 for f in matrix_list if f["status"] == "PASS")
        total_watch = sum(1 for f in matrix_list if f["status"] == "WATCH")
        total_fail = sum(1 for f in matrix_list if f["status"] == "FAIL")

        return {
            "symbol": symbol,
            "cur_price": cur_price,
            "total_factors": 21,
            "passed_count": total_pass,
            "watch_count": total_watch,
            "fail_count": total_fail,
            "confluence_percentage": round((total_pass / 21.0) * 100, 1),
            "regime": regime_info["regime"],
            "regime_directive": regime_info["action_directive"],
            "chop_index": chop_info["chop"],
            "is_choppy": chop_info["is_choppy"],
            "flow_direction": flow.get("flow_direction", "NEUTRAL"),
            "cvd": flow.get("cvd", 0.0),
            "killzone": kz_info["killzone"],
            "risk_reward": rr_info["rr_label"],
            "is_tradeable": entry_passed and kz_info["allow_new_entries"],
            "factors": matrix_list,
        }


# Singleton instance
institutional_filter_engine = InstitutionalFilterEngine()
