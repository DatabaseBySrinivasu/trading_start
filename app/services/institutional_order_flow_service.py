"""
Institutional Order Flow & Order Block Discovery Engine.

Responsibilities:
1. Precise Order Block (OB) Detection (Bullish Demand Zones & Bearish Supply Zones) with exact price bounds [Low - High], 50% Mean Threshold, and mitigation lifecycle tracking (UNMITIGATED_FRESH, TESTED_REJECTED, MITIGATED, BREACHED_BREAKER).
2. Fair Value Gap (FVG) and Liquidity Imbalance detection with Consequent Encroachment (50% midpoint) tracking.
3. Institutional Buyer vs Seller Footprint: Bar-by-bar Volume Delta, Cumulative Volume Delta (CVD), and CVD Absorption/Exhaustion Divergences.
4. Institutional Phase Classification: ACCUMULATION (Buying), MARKUP, DISTRIBUTION (Selling), and MARKDOWN.
5. Timestamped Institutional Activity Timeline: Precise log of WHEN institutions bought or sold with price levels, volume deltas, and narrative descriptions.
6. Option Chain Institutional Pressure (Call Writing/Unwinding vs Put Writing/Unwinding).
"""

import math
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import pytz

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))
IST_TZ = pytz.timezone("Asia/Kolkata")


class InstitutionalOrderFlowService:
    """
    Advanced Institutional Smart Money Concepts (SMC) & Order Flow Engine.
    """

    @staticmethod
    def _clean_float(val: Any, default: float = 0.0) -> float:
        try:
            if val is None or pd.isna(val):
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
    def calculate_bar_delta_and_cvd(
        cls, df: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Calculates bar-by-bar volume delta and Cumulative Volume Delta (CVD).
        Delta = ((Close - Open) / (High - Low + eps)) * Volume
        """
        df = cls._clean_df(df)
        if df is None or len(df) == 0:
            return pd.Series([0.0]), pd.Series([0.0])

        highs = df["high"].values
        lows = df["low"].values
        opens = df["open"].values
        closes = df["close"].values
        volumes = df["volume"].fillna(1000).replace(0, 1000.0).values

        candle_ranges = np.maximum(highs - lows, 1e-5)
        deltas = ((closes - opens) / candle_ranges) * volumes
        delta_series = pd.Series(deltas, index=df.index)
        cvd_series = delta_series.cumsum()
        return delta_series, cvd_series

    @classmethod
    def detect_order_blocks(
        cls, df: pd.DataFrame, lookback: int = 50
    ) -> Dict[str, Any]:
        """
        Scans candle series for institutional Order Blocks (Bullish Demand & Bearish Supply)
        with exact [Low - High] price bounds, 50% Mean Threshold, volume deltas,
        and chronological mitigation lifecycle tracking.
        """
        df = cls._clean_df(df)
        if df is None or len(df) < 6:
            return {
                "bullish_order_blocks": [],
                "bearish_order_blocks": [],
                "nearest_bullish_ob": None,
                "nearest_bearish_ob": None,
                "active_bullish_count": 0,
                "active_bearish_count": 0,
                "breaker_blocks": [],
                "smc_phase": "NEUTRAL_RANGE",
            }

        window = df.tail(lookback).copy()
        cur_price = cls._clean_float(window["close"].iloc[-1])
        vol_sma20 = float(window["volume"].tail(20).mean()) if len(window) >= 20 else float(window["volume"].mean())
        if vol_sma20 <= 0:
            vol_sma20 = 1000.0

        all_bullish_obs: List[Dict[str, Any]] = []
        all_bearish_obs: List[Dict[str, Any]] = []
        breaker_blocks: List[Dict[str, Any]] = []

        # Loop through candles to identify institutional displacement pivots
        # An order block is the last contrary candle prior to an aggressive displacement (2 consecutive strong candles or high volume)
        for i in range(2, len(window) - 2):
            c_ob = window.iloc[i - 1]     # Candidate Order Block candle
            c_disp1 = window.iloc[i]       # Displacement candle 1
            c_disp2 = window.iloc[i + 1]   # Displacement candle 2

            ob_open = cls._clean_float(c_ob["open"])
            ob_close = cls._clean_float(c_ob["close"])
            ob_high = cls._clean_float(c_ob["high"])
            ob_low = cls._clean_float(c_ob["low"])
            ob_vol = cls._clean_float(c_ob["volume"], 1000.0)

            disp1_close = cls._clean_float(c_disp1["close"])
            disp1_open = cls._clean_float(c_disp1["open"])
            disp2_close = cls._clean_float(c_disp2["close"])

            ob_time = str(window.index[i - 1])
            ob_mean_threshold = round((ob_high + ob_low) / 2.0, 2)
            ob_vol_delta = round(((ob_close - ob_open) / max(ob_high - ob_low, 1e-5)) * ob_vol, 1)

            # Subsequent price action after displacement for mitigation check
            subsequent = window.iloc[i + 2:]
            subsequent_lows = subsequent["low"].values if not subsequent.empty else []
            subsequent_highs = subsequent["high"].values if not subsequent.empty else []
            subsequent_closes = subsequent["close"].values if not subsequent.empty else []

            # -----------------------------------------------------------------
            # 1. BULLISH ORDER BLOCK (Institutional Accumulation / Demand Zone)
            # -----------------------------------------------------------------
            # Criteria: Down-candle (close < open), followed by aggressive upward displacement breaking above OB high
            is_down_candle = ob_close < ob_open
            strong_up_disp = disp1_close > ob_high and disp2_close > disp1_close
            has_volume_thrust = (cls._clean_float(c_disp1["volume"]) > 1.2 * vol_sma20) or (disp1_close - disp1_open > (ob_high - ob_low))

            if is_down_candle and strong_up_disp and has_volume_thrust:
                # Assess Mitigation Status
                mitigation_status = "UNMITIGATED_FRESH"
                is_active = True
                touch_count = 0

                if len(subsequent_lows) > 0:
                    min_sub_low = float(np.min(subsequent_lows))
                    min_sub_close = float(np.min(subsequent_closes))

                    # If price closed below OB low -> Breached / Invalidation
                    if min_sub_close < ob_low:
                        mitigation_status = "BREACHED_INVALIDATED"
                        is_active = False
                        # Converted to Bearish Breaker Block
                        breaker_blocks.append({
                            "type": "BEARISH_BREAKER",
                            "zone_top": round(ob_high, 2),
                            "zone_bottom": round(ob_low, 2),
                            "mean_threshold": ob_mean_threshold,
                            "formation_time": ob_time,
                            "description": "Bullish Order Block breached downwards; now acts as institutional resistance.",
                        })
                    elif min_sub_low <= ob_high and min_sub_low >= ob_low:
                        # Price tested the zone and bounced
                        touch_count = sum(1 for l in subsequent_lows if ob_low <= l <= ob_high)
                        mitigation_status = "TESTED_REJECTED" if cur_price > ob_high else "REACTING_IN_ZONE"
                        is_active = True
                    elif min_sub_low < ob_low and min_sub_close >= ob_low:
                        # Liquidity sweep below OB with close inside
                        mitigation_status = "LIQUIDITY_SWEPT_HELD"
                        is_active = True

                distance_pts = round(cur_price - ob_high, 2)
                distance_pct = round((distance_pts / cur_price) * 100, 2) if cur_price > 0 else 0.0

                ob_entry = {
                    "type": "BULLISH_ORDER_BLOCK",
                    "role": "DEMAND_ZONE (INSTITUTIONAL BUYING POOL)",
                    "zone_top": round(ob_high, 2),
                    "zone_bottom": round(ob_low, 2),
                    "mean_threshold": ob_mean_threshold,
                    "volume": int(ob_vol),
                    "volume_delta": ob_vol_delta,
                    "formation_time": ob_time,
                    "bars_ago": len(window) - i,
                    "mitigation_status": mitigation_status,
                    "is_active": is_active,
                    "touch_count": touch_count,
                    "distance_pts": distance_pts,
                    "distance_pct": distance_pct,
                    "in_zone": ob_low <= cur_price <= ob_high,
                }
                all_bullish_obs.append(ob_entry)

            # -----------------------------------------------------------------
            # 2. BEARISH ORDER BLOCK (Institutional Distribution / Supply Zone)
            # -----------------------------------------------------------------
            # Criteria: Up-candle (close > open), followed by aggressive downward displacement breaking below OB low
            is_up_candle = ob_close > ob_open
            strong_down_disp = disp1_close < ob_low and disp2_close < disp1_close
            has_vol_down_thrust = (cls._clean_float(c_disp1["volume"]) > 1.2 * vol_sma20) or (disp1_open - disp1_close > (ob_high - ob_low))

            if is_up_candle and strong_down_disp and has_vol_down_thrust:
                mitigation_status = "UNMITIGATED_FRESH"
                is_active = True
                touch_count = 0

                if len(subsequent_highs) > 0:
                    max_sub_high = float(np.max(subsequent_highs))
                    max_sub_close = float(np.max(subsequent_closes))

                    if max_sub_close > ob_high:
                        mitigation_status = "BREACHED_INVALIDATED"
                        is_active = False
                        # Converted to Bullish Breaker Block
                        breaker_blocks.append({
                            "type": "BULLISH_BREAKER",
                            "zone_top": round(ob_high, 2),
                            "zone_bottom": round(ob_low, 2),
                            "mean_threshold": ob_mean_threshold,
                            "formation_time": ob_time,
                            "description": "Bearish Order Block breached upwards; now acts as institutional support.",
                        })
                    elif max_sub_high >= ob_low and max_sub_high <= ob_high:
                        touch_count = sum(1 for h in subsequent_highs if ob_low <= h <= ob_high)
                        mitigation_status = "TESTED_REJECTED" if cur_price < ob_low else "REACTING_IN_ZONE"
                        is_active = True
                    elif max_sub_high > ob_high and max_sub_close <= ob_high:
                        mitigation_status = "LIQUIDITY_SWEPT_HELD"
                        is_active = True

                distance_pts = round(ob_low - cur_price, 2)
                distance_pct = round((distance_pts / cur_price) * 100, 2) if cur_price > 0 else 0.0

                ob_entry = {
                    "type": "BEARISH_ORDER_BLOCK",
                    "role": "SUPPLY_ZONE (INSTITUTIONAL SELLING POOL)",
                    "zone_top": round(ob_high, 2),
                    "zone_bottom": round(ob_low, 2),
                    "mean_threshold": ob_mean_threshold,
                    "volume": int(ob_vol),
                    "volume_delta": ob_vol_delta,
                    "formation_time": ob_time,
                    "bars_ago": len(window) - i,
                    "mitigation_status": mitigation_status,
                    "is_active": is_active,
                    "touch_count": touch_count,
                    "distance_pts": distance_pts,
                    "distance_pct": distance_pct,
                    "in_zone": ob_low <= cur_price <= ob_high,
                }
                all_bearish_obs.append(ob_entry)

        # Filter active demand and supply order blocks
        active_bullish = [b for b in all_bullish_obs if b["is_active"]]
        active_bearish = [b for b in all_bearish_obs if b["is_active"]]

        # Find nearest Demand and Supply zones to current spot price
        # Nearest Bullish OB is the closest active demand zone below or around current price
        nearest_bull = None
        bull_below = [b for b in active_bullish if b["zone_bottom"] <= cur_price or b["in_zone"]]
        if bull_below:
            nearest_bull = max(bull_below, key=lambda x: x["zone_top"])
        elif active_bullish:
            nearest_bull = min(active_bullish, key=lambda x: abs(x["zone_top"] - cur_price))

        # Nearest Bearish OB is the closest active supply zone above or around current price
        nearest_bear = None
        bear_above = [b for b in active_bearish if b["zone_top"] >= cur_price or b["in_zone"]]
        if bear_above:
            nearest_bear = min(bear_above, key=lambda x: x["zone_bottom"])
        elif active_bearish:
            nearest_bear = min(active_bearish, key=lambda x: abs(x["zone_bottom"] - cur_price))

        # Determine overall SMC Bias
        smc_bias = "NEUTRAL RANGE"
        if nearest_bull and nearest_bull.get("in_zone"):
            smc_bias = "DEMAND_ORDER_BLOCK_TEST (BULLISH ACCUMULATION)"
        elif nearest_bear and nearest_bear.get("in_zone"):
            smc_bias = "SUPPLY_ORDER_BLOCK_TEST (BEARISH DISTRIBUTION)"
        elif nearest_bull and (cur_price - nearest_bull.get("zone_top", 0)) < 0.005 * cur_price:
            smc_bias = "APPROACHING DEMAND OB (BULLISH)"
        elif nearest_bear and (nearest_bear.get("zone_bottom", 0) - cur_price) < 0.005 * cur_price:
            smc_bias = "APPROACHING SUPPLY OB (BEARISH)"
        elif len(active_bullish) > len(active_bearish):
            smc_bias = "BULLISH DEMAND DOMINANCE"
        elif len(active_bearish) > len(active_bullish):
            smc_bias = "BEARISH SUPPLY DOMINANCE"

        demand_zone = {
            "top": nearest_bull["zone_top"],
            "bottom": nearest_bull["zone_bottom"],
            "mean_threshold": nearest_bull["mean_threshold"],
            "status": nearest_bull["mitigation_status"],
        } if nearest_bull else None

        supply_zone = {
            "top": nearest_bear["zone_top"],
            "bottom": nearest_bear["zone_bottom"],
            "mean_threshold": nearest_bear["mean_threshold"],
            "status": nearest_bear["mitigation_status"],
        } if nearest_bear else None

        return {
            "smc_bias": smc_bias,
            "demand_zone": demand_zone,
            "supply_zone": supply_zone,
            "bullish_order_blocks": all_bullish_obs,
            "bearish_order_blocks": all_bearish_obs,
            "active_bullish_order_blocks": active_bullish,
            "active_bearish_order_blocks": active_bearish,
            "nearest_bullish_ob": nearest_bull,
            "nearest_bearish_ob": nearest_bear,
            "active_bullish_count": len(active_bullish),
            "active_bearish_count": len(active_bearish),
            "breaker_blocks": breaker_blocks,
        }

    @classmethod
    def detect_fair_value_gaps(
        cls, df: pd.DataFrame, lookback: int = 40
    ) -> List[Dict[str, Any]]:
        """
        Detects 3-bar institutional Fair Value Gaps (FVG) / Liquidity Imbalances
        and evaluates whether they are unfilled, partially filled, or mitigated.
        """
        df = cls._clean_df(df)
        if df is None or len(df) < 5:
            return []

        window = df.tail(lookback).copy().reset_index(drop=True)
        cur_price = cls._clean_float(window["close"].iloc[-1])
        fvgs = []

        for i in range(2, len(window)):
            c1 = window.iloc[i - 2]
            c2 = window.iloc[i - 1]
            c3 = window.iloc[i]

            c1_high = cls._clean_float(c1["high"])
            c1_low = cls._clean_float(c1["low"])
            c3_high = cls._clean_float(c3["high"])
            c3_low = cls._clean_float(c3["low"])

            # 1. Bullish FVG (Imbalance to the upside: Candle 3 low > Candle 1 high)
            if c3_low > c1_high:
                gap_top = c3_low
                gap_bottom = c1_high
                midpoint_ce = round((gap_top + gap_bottom) / 2.0, 2)
                subsequent = window.iloc[i + 1:]
                sub_lows = subsequent["low"].values if not subsequent.empty else []
                is_filled = False
                if len(sub_lows) > 0:
                    min_sub = float(np.min(sub_lows))
                    is_filled = min_sub <= gap_bottom

                fvgs.append({
                    "type": "BULLISH_FVG",
                    "zone_top": round(gap_top, 2),
                    "zone_bottom": round(gap_bottom, 2),
                    "gap_top": round(gap_top, 2),
                    "gap_bottom": round(gap_bottom, 2),
                    "consequent_encroachment": midpoint_ce,
                    "midpoint_ce": midpoint_ce,
                    "gap_size": round(gap_top - gap_bottom, 2),
                    "candle_idx": i - 1,
                    "is_unfilled": not is_filled,
                    "fill_status": "FULLY_MITIGATED" if is_filled else "UNFILLED_OPEN",
                    "distance_pts": round(cur_price - gap_top, 2),
                })

            # 2. Bearish FVG (Imbalance to the downside: Candle 3 high < Candle 1 low)
            if c3_high < c1_low:
                gap_top = c1_low
                gap_bottom = c3_high
                midpoint_ce = round((gap_top + gap_bottom) / 2.0, 2)
                subsequent = window.iloc[i + 1:]
                sub_highs = subsequent["high"].values if not subsequent.empty else []
                is_filled = False
                if len(sub_highs) > 0:
                    max_sub = float(np.max(sub_highs))
                    is_filled = max_sub >= gap_top

                fvgs.append({
                    "type": "BEARISH_FVG",
                    "zone_top": round(gap_top, 2),
                    "zone_bottom": round(gap_bottom, 2),
                    "gap_top": round(gap_top, 2),
                    "gap_bottom": round(gap_bottom, 2),
                    "consequent_encroachment": midpoint_ce,
                    "midpoint_ce": midpoint_ce,
                    "gap_size": round(gap_top - gap_bottom, 2),
                    "candle_idx": i - 1,
                    "is_unfilled": not is_filled,
                    "fill_status": "FULLY_MITIGATED" if is_filled else "UNFILLED_OPEN",
                    "distance_pts": round(gap_bottom - cur_price, 2),
                })

        return fvgs

    @classmethod
    def analyze_institutional_flow_and_timeline(
        cls, df: pd.DataFrame, symbol: str = "^NSEI"
    ) -> Dict[str, Any]:
        """
        Analyzes institutional buyer vs seller order flow:
        - Real-time Cumulative Volume Delta (CVD)
        - Institutional Buying vs Selling Phase (Accumulation, Markup, Distribution, Markdown)
        - Delta Divergence (Absorption vs Distribution)
        - Chronological Institutional Activity Timeline
        """
        df = cls._clean_df(df)
        if df is None or len(df) < 10:
            return {
                "institutional_phase": "NEUTRAL_CONSOLIDATION",
                "phase_badge": "NEUTRAL FLOW",
                "phase_description": "Normal market flow; no distinct institutional imbalance.",
                "buyer_dominance_pct": 50.0,
                "seller_dominance_pct": 50.0,
                "cumulative_volume_delta": 0.0,
                "cvd_trend": "FLAT",
                "delta_divergence": "NONE",
                "timeline_events": [],
            }

        delta_series, cvd_series = cls.calculate_bar_delta_and_cvd(df)
        window = df.tail(30).copy()
        recent_deltas = delta_series.tail(30).values
        recent_cvd = cvd_series.tail(30).values
        closes = window["close"].values
        volumes = window["volume"].values

        cur_delta = float(recent_deltas[-1])
        cur_cvd = float(recent_cvd[-1])
        cvd_start = float(recent_cvd[0])
        cvd_change = cur_cvd - cvd_start

        # Buyer vs Seller volume estimation
        pos_deltas = np.sum(recent_deltas[recent_deltas > 0]) if len(recent_deltas[recent_deltas > 0]) > 0 else 1.0
        neg_deltas = abs(np.sum(recent_deltas[recent_deltas < 0])) if len(recent_deltas[recent_deltas < 0]) > 0 else 1.0
        total_flow = pos_deltas + neg_deltas
        buyer_pct = round((pos_deltas / max(total_flow, 1.0)) * 100, 1)
        seller_pct = round(100.0 - buyer_pct, 1)

        # ---------------------------------------------------------------------
        # 1. Delta Divergence Detection
        # ---------------------------------------------------------------------
        # Bullish Absorption Divergence: Price making Lower Low, but CVD making Higher High
        price_trend = closes[-1] - closes[0]
        cvd_trend = cur_cvd - cvd_start

        if price_trend < 0 and cvd_trend > 0:
            divergence = "BULLISH_ABSORPTION_DIVERGENCE"
            div_desc = "Price dropped while Cumulative Volume Delta rose -> Institutional smart money passively absorbing supply."
        elif price_trend > 0 and cvd_trend < 0:
            divergence = "BEARISH_EXHAUSTION_DIVERGENCE"
            div_desc = "Price rose while Cumulative Volume Delta fell -> Institutional smart money distributing into retail FOMO."
        elif cvd_trend > 0:
            divergence = "BULLISH_CONVERGENCE"
            div_desc = "Price and CVD expanding in tandem to the upside -> Strong aggressive institutional buying."
        else:
            divergence = "BEARISH_CONVERGENCE"
            div_desc = "Price and CVD declining in tandem -> Strong aggressive institutional selling."

        # ---------------------------------------------------------------------
        # 2. Institutional Market Phase State
        # ---------------------------------------------------------------------
        vol_sma = float(np.mean(volumes))
        last_vol_ratio = float(volumes[-1]) / max(vol_sma, 1.0)
        is_high_volume = last_vol_ratio >= 1.4

        if buyer_pct >= 60.0 and cvd_change > 0:
            if is_high_volume and closes[-1] >= closes[-3]:
                phase = "INSTITUTIONAL_MARKUP"
                phase_badge = "🚀 INSTITUTIONAL BUYING EXPANSION (MARKUP)"
                phase_desc = f"Smart money aggressively buying and marking up price (+{buyer_pct}% buying dominance, CVD +{cur_cvd:,.0f})."
            else:
                phase = "INSTITUTIONAL_ACCUMULATION"
                phase_badge = "🟢 INSTITUTIONAL ACCUMULATION (DIP BUYING)"
                phase_desc = f"Institutions absorbing liquidity on pullbacks ({buyer_pct}% buyer volume, CVD +{cur_cvd:,.0f})."
        elif seller_pct >= 60.0 and cvd_change < 0:
            if is_high_volume and closes[-1] <= closes[-3]:
                phase = "INSTITUTIONAL_MARKDOWN"
                phase_badge = "🚨 INSTITUTIONAL SELLING EXPANSION (MARKDOWN)"
                phase_desc = f"Smart money dumping inventory and pushing price down ({seller_pct}% selling dominance, CVD {cur_cvd:,.0f})."
            else:
                phase = "INSTITUTIONAL_DISTRIBUTION"
                phase_badge = "🔴 INSTITUTIONAL DISTRIBUTION (PROFIT SELLING)"
                phase_desc = f"Institutions liquidating long positions into resistance ({seller_pct}% seller volume, CVD {cur_cvd:,.0f})."
        else:
            phase = "NEUTRAL_CONSOLIDATION"
            phase_badge = "⚪ EQUILIBRIUM / BALANCED ORDER FLOW"
            phase_desc = "Two-sided balanced trade between institutional buyers and sellers."

        # ---------------------------------------------------------------------
        # 3. Chronological Institutional Footprint Timeline
        # ---------------------------------------------------------------------
        # Scan recent bars for institutional footprint events (Volume surges, major Delta spikes, Rejections)
        timeline_events: List[Dict[str, Any]] = []

        for idx in range(max(0, len(window) - 15), len(window)):
            bar = window.iloc[idx]
            bar_time = str(window.index[idx])
            bar_open = cls._clean_float(bar["open"])
            bar_close = cls._clean_float(bar["close"])
            bar_high = cls._clean_float(bar["high"])
            bar_low = cls._clean_float(bar["low"])
            bar_vol = cls._clean_float(bar["volume"], 1000.0)
            bar_delta_val = delta_series.iloc[idx]
            bar_ratio = bar_vol / max(vol_sma, 1.0)

            event_type = None
            action = "NEUTRAL"
            desc = ""

            # Check for institutional volume surge
            if bar_ratio >= 1.8:
                if bar_close > bar_open and bar_delta_val > 0:
                    event_type = "INSTITUTIONAL_BUYING_SURGE"
                    action = "BUYING_SPIKE"
                    desc = f"Institutional buying surge ({bar_ratio:.1f}x avg volume, +{bar_delta_val:,.0f} delta) executed at ₹{bar_close:,.2f}"
                elif bar_close < bar_open and bar_delta_val < 0:
                    event_type = "INSTITUTIONAL_SELLING_SURGE"
                    action = "SELLING_SPIKE"
                    desc = f"Institutional selling surge ({bar_ratio:.1f}x avg volume, {bar_delta_val:,.0f} delta) dumped at ₹{bar_close:,.2f}"

            # Check for liquidity wick rejection
            candle_len = max(bar_high - bar_low, 1e-5)
            lower_wick = min(bar_open, bar_close) - bar_low
            upper_wick = bar_high - max(bar_open, bar_close)

            if lower_wick >= 0.55 * candle_len and not event_type:
                event_type = "DEMAND_ABSORPTION_WICK"
                action = "BUYING_ABSORPTION"
                desc = f"Institutional buyer defense: long lower rejection wick at ₹{bar_low:,.2f} absorbed sell orders."
            elif upper_wick >= 0.55 * candle_len and not event_type:
                event_type = "SUPPLY_DISTRIBUTION_WICK"
                action = "SELLING_EXHAUSTION"
                desc = f"Institutional seller rejection: heavy supply capping upside at ₹{bar_high:,.2f} triggered sell-off."

            if event_type:
                actor = "INSTITUTIONAL_BUYER" if "BUY" in action else "INSTITUTIONAL_SELLER"
                timeline_events.append({
                    "timestamp": bar_time,
                    "event_type": event_type,
                    "action": action,
                    "actor": actor,
                    "price": round(bar_close, 2),
                    "price_level": round(bar_close, 2),
                    "volume_delta": round(bar_delta_val, 1),
                    "volume_ratio": round(bar_ratio, 2),
                    "description": desc,
                    "narrative": desc,
                })

        return {
            "institutional_phase": phase,
            "phase_badge": phase_badge,
            "phase_description": phase_desc,
            "buyer_dominance_pct": buyer_pct,
            "seller_dominance_pct": seller_pct,
            "cumulative_volume_delta": round(cur_cvd, 1),
            "current_bar_delta": round(cur_delta, 1),
            "cvd_trend": "EXPANDING_POSITIVE" if cvd_change > 0 else "EXPANDING_NEGATIVE",
            "delta_divergence": divergence,
            "delta_divergence_description": div_desc,
            "timeline_events": timeline_events,
        }

    @classmethod
    def get_comprehensive_institutional_snapshot(
        cls,
        symbol: str,
        df: pd.DataFrame,
        live_price: float = 0.0,
        pcr_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Synthesizes complete Institutional Order Block & Order Flow Intelligence:
        1. Bullish & Bearish Order Blocks with exact price bounds & mitigation.
        2. Unfilled Fair Value Gaps (FVGs).
        3. Real-time CVD and Institutional Phase (Accumulation / Distribution).
        4. Chronological Timeline of WHEN institutions bought and sold.
        5. Option Chain Institutional Writing Footprint.
        """
        df = cls._clean_df(df)
        if df is None or len(df) < 6:
            return {
                "symbol": symbol,
                "spot_price": live_price,
                "nearest_bullish_ob": None,
                "nearest_bearish_ob": None,
                "active_bullish_obs_count": 0,
                "active_bearish_obs_count": 0,
                "all_order_blocks": [],
                "fair_value_gaps": [],
                "institutional_phase": "NEUTRAL",
                "phase_badge": "NEUTRAL",
                "phase_description": "Insufficient candle depth",
                "buyer_dominance_pct": 50.0,
                "seller_dominance_pct": 50.0,
                "cumulative_volume_delta": 0.0,
                "delta_divergence": "NONE",
                "timeline_events": [],
                "option_chain_institutional_pressure": "BALANCED",
                "timestamp": datetime.now(IST).isoformat(),
            }

        cur_price = live_price if live_price > 0 else cls._clean_float(df["close"].iloc[-1])

        # 1. Order Blocks
        ob_res = cls.detect_order_blocks(df, lookback=60)
        nearest_bull = ob_res.get("nearest_bullish_ob")
        nearest_bear = ob_res.get("nearest_bearish_ob")

        # Combine and sort all active Order Blocks by proximity to current spot price
        active_obs = ob_res.get("active_bullish_order_blocks", []) + ob_res.get("active_bearish_order_blocks", [])
        sorted_obs = sorted(active_obs, key=lambda x: abs(x["zone_top"] - cur_price))

        # 2. Fair Value Gaps
        fvgs = cls.detect_fair_value_gaps(df, lookback=40)
        unfilled_fvgs = [f for f in fvgs if f.get("is_unfilled")]

        # 3. Institutional Flow & Timeline
        flow_res = cls.analyze_institutional_flow_and_timeline(df, symbol=symbol)

        # 4. Option Chain Institutional Pressure
        opt_pressure = "BALANCED"
        opt_desc = "Neutral open interest positioning."
        if pcr_data:
            pcr_val = cls._clean_float(pcr_data.get("pcr_oi", 1.0))
            if pcr_val >= 1.25:
                opt_pressure = "HEAVY_PUT_WRITING_INSTITUTIONAL_FLOOR"
                opt_desc = f"Institutions aggressively writing Puts (PCR {pcr_val:.2f}) building solid support floor."
            elif pcr_val <= 0.75:
                opt_pressure = "HEAVY_CALL_WRITING_INSTITUTIONAL_CEILING"
                opt_desc = f"Institutions aggressively writing Calls (PCR {pcr_val:.2f}) capping upside resistance."
            elif pcr_val >= 1.05:
                opt_pressure = "MODERATE_BULLISH_OI_BIAS"
                opt_desc = f"Mild institutional Put writing dominance (PCR {pcr_val:.2f})."
            elif pcr_val <= 0.90:
                opt_pressure = "MODERATE_BEARISH_OI_BIAS"
                opt_desc = f"Mild institutional Call writing dominance (PCR {pcr_val:.2f})."

        # 5. Order Block Strategic Directives
        strategy_directive = "NEUTRAL"
        if nearest_bull and nearest_bull.get("in_zone"):
            strategy_directive = f"BUYING OPPORTUNITY: Price is reacting inside Bullish Order Block Demand Zone (₹{nearest_bull['zone_bottom']} - ₹{nearest_bull['zone_top']}). Target: ₹{nearest_bear['zone_bottom'] if nearest_bear else cur_price * 1.01:.2f}"
        elif nearest_bear and nearest_bear.get("in_zone"):
            strategy_directive = f"SELLING OPPORTUNITY: Price is reacting inside Bearish Order Block Supply Zone (₹{nearest_bear['zone_bottom']} - ₹{nearest_bear['zone_top']}). Target: ₹{nearest_bull['zone_top'] if nearest_bull else cur_price * 0.99:.2f}"
        elif flow_res.get("institutional_phase") in ["INSTITUTIONAL_ACCUMULATION", "INSTITUTIONAL_MARKUP"]:
            strategy_directive = f"BULLISH CONTINUATION: Institutional buying dominance ({flow_res.get('buyer_dominance_pct')}%) with positive CVD (+{flow_res.get('cumulative_volume_delta'):,.0f}). Favor CALL pullbacks to demand zones."
        elif flow_res.get("institutional_phase") in ["INSTITUTIONAL_DISTRIBUTION", "INSTITUTIONAL_MARKDOWN"]:
            strategy_directive = f"BEARISH CONTINUATION: Institutional selling dominance ({flow_res.get('seller_dominance_pct')}%) with negative CVD ({flow_res.get('cumulative_volume_delta'):,.0f}). Favor PUT rallies to supply zones."

        return {
            "symbol": symbol,
            "spot_price": round(cur_price, 2),
            "order_blocks": ob_res,
            "order_flow": flow_res,
            "smart_money_summary": strategy_directive,
            "nearest_bullish_ob": nearest_bull,
            "nearest_bearish_ob": nearest_bear,
            "active_bullish_obs_count": ob_res.get("active_bullish_count", 0),
            "active_bearish_obs_count": ob_res.get("active_bearish_count", 0),
            "all_order_blocks": sorted_obs,
            "fair_value_gaps": unfilled_fvgs,
            "breaker_blocks": ob_res.get("breaker_blocks", []),
            "institutional_phase": flow_res.get("institutional_phase"),
            "phase_badge": flow_res.get("phase_badge"),
            "phase_description": flow_res.get("phase_description"),
            "buyer_dominance_pct": flow_res.get("buyer_dominance_pct"),
            "seller_dominance_pct": flow_res.get("seller_dominance_pct"),
            "cumulative_volume_delta": flow_res.get("cumulative_volume_delta"),
            "current_bar_delta": flow_res.get("current_bar_delta"),
            "cvd_trend": flow_res.get("cvd_trend"),
            "delta_divergence": flow_res.get("delta_divergence"),
            "delta_divergence_description": flow_res.get("delta_divergence_description"),
            "timeline_events": flow_res.get("timeline_events", []),
            "option_chain_institutional_pressure": opt_pressure,
            "option_chain_description": opt_desc,
            "strategy_directive": strategy_directive,
            "timestamp": datetime.now(IST).isoformat(),
        }


# Global singleton instance
institutional_order_flow_service = InstitutionalOrderFlowService()
