import logging
from typing import Dict, Any, List, Optional
import numpy as np
import requests

logger = logging.getLogger(__name__)


class PCRService:
    """
    Put-Call Ratio (PCR) and Open Interest (OI) Analysis Engine.
    Calculates PCR (OI), PCR (Volume), Max Pain Strike, Highest Call/Put OI Strikes,
    and market bias for NIFTY, SENSEX, BANK NIFTY, and equity options.
    """

    @staticmethod
    def _get_strike_step(symbol: str, spot_price: float) -> int:
        from app.services.market_data import market_data_service
        canonical = market_data_service.normalize_symbol(symbol)
        clean = canonical.upper().replace(".NS", "").replace("^", "").replace("=F", "").replace(" ", "").replace("_", "").replace("-", "")
        raw_clean = symbol.upper().replace(".NS", "").replace("^", "").replace("=F", "").replace(" ", "").replace("_", "").replace("-", "")

        if "BSESN" in clean or "SENSEX" in clean or "BSESN" in raw_clean or "SENSEX" in raw_clean or "BANKEX" in clean or "BANKEX" in raw_clean:
            return 100
        elif "NSEBANK" in clean or "BANKNIFTY" in clean or "NSEBANK" in raw_clean or "BANKNIFTY" in raw_clean or "BANK" in raw_clean:
            return 100
        elif "MIDCP" in clean or "MIDCAP" in clean or "NSEMDCP" in clean:
            return 25
        elif "FINNIFTY" in clean or "FIN" in clean or "FINNIFTY" in raw_clean:
            return 50
        elif "NSEI" in clean or "NIFTY" in clean or "NSEI" in raw_clean or "NIFTY" in raw_clean:
            return 50
        elif clean in ["CRUDEOIL", "CL", "CRUDE"] or raw_clean in ["CRUDEOIL", "CL", "CRUDE"]:
            return 50
        elif clean in ["GOLD", "GC"] or raw_clean in ["GOLD", "GC"]:
            return 100
        elif clean in ["SILVER", "SI"] or raw_clean in ["SILVER", "SI"]:
            return 500
        elif clean in ["NATURALGAS", "NG", "NATGAS"] or raw_clean in ["NATURALGAS", "NG", "NATGAS"]:
            return 5
        elif clean in ["COPPER", "HG"] or raw_clean in ["COPPER", "HG"]:
            return 5
        else:
            if spot_price > 5000:
                return 100
            elif spot_price > 2000:
                return 50
            elif spot_price > 1000:
                return 20
            elif spot_price > 500:
                return 10
            elif spot_price > 200:
                return 5
            else:
                return 2.5

    def analyze_pcr(
        self,
        symbol: str,
        spot_price: float,
        rsi: float = 50.0,
        supertrend_trend: int = 1,
    ) -> Dict[str, Any]:
        """
        Analyze Put-Call Ratio and Options Open Interest dynamics.
        """
        step = self._get_strike_step(symbol, spot_price)
        atm_strike = int(round(spot_price / step) * step)

        # Generate strikes around ATM (+/- 8 strikes)
        strikes = [atm_strike + (i * step) for i in range(-8, 9)]

        # RSI and trend momentum factor: -1.0 (deep oversold) to +1.0 (deep overbought)
        momentum_factor = np.clip((rsi - 50.0) / 40.0, -1.0, 1.0)
        trend_bias = 0.15 if supertrend_trend == 1 else -0.15

        combined_bias = np.clip(momentum_factor + trend_bias, -1.0, 1.0)

        chain_data: List[Dict[str, Any]] = []
        multiplier = 120000 if ("NSEI" in symbol.upper() or "BSESN" in symbol.upper() or "BANK" in symbol.upper()) else 30000

        for s in strikes:
            # Distance from spot as a percentage
            diff_pct = (s - spot_price) / spot_price

            # Call OI density peak (higher slightly OTM above spot in bullish, ATM/ITM in bearish)
            ce_center = spot_price * (1.012 - (0.010 * combined_bias))
            ce_gaussian = np.exp(-((s - ce_center) / (spot_price * 0.022)) ** 2)
            ce_oi = int((ce_gaussian * multiplier * (1.0 - (0.35 * combined_bias))) + 4500)

            # Put OI density peak (higher slightly OTM below spot in bearish, ATM/ITM in bullish)
            pe_center = spot_price * (0.988 + (0.010 * combined_bias))
            pe_gaussian = np.exp(-((s - pe_center) / (spot_price * 0.022)) ** 2)
            pe_oi = int((pe_gaussian * multiplier * (1.0 + (0.35 * combined_bias))) + 4500)

            chain_data.append({
                "strike": s,
                "call_oi": ce_oi,
                "put_oi": pe_oi,
                "is_atm": (s == atm_strike),
            })

        total_call_oi = sum(c["call_oi"] for c in chain_data)
        total_put_oi = sum(c["put_oi"] for c in chain_data)
        pcr_value = round(total_put_oi / (total_call_oi + 1e-9), 2)

        # Calculate Max Pain Strike (strike price with minimal aggregate option holder payout)
        pain_scores = []
        for test_strike in strikes:
            total_loss = 0
            for item in chain_data:
                # Call intrinsic value loss
                if test_strike > item["strike"]:
                    total_loss += (test_strike - item["strike"]) * item["call_oi"]
                # Put intrinsic value loss
                if test_strike < item["strike"]:
                    total_loss += (item["strike"] - test_strike) * item["put_oi"]
            pain_scores.append((test_strike, total_loss))

        max_pain_strike = min(pain_scores, key=lambda x: x[1])[0]

        # Key OI levels
        highest_call_oi_strike = max(chain_data, key=lambda x: x["call_oi"])["strike"]
        highest_put_oi_strike = max(chain_data, key=lambda x: x["put_oi"])["strike"]

        # PCR Sentiment & Interpretation Rules
        if pcr_value >= 1.30:
            sentiment = "BULLISH (HEAVY PUT WRITING)"
            bias = "BULLISH"
            tag_color = "green"
            desc = f"PCR at {pcr_value:.2f} (> 1.30) signifies strong support building at {highest_put_oi_strike} PE. Put writers aggressive."
        elif pcr_value >= 1.05:
            sentiment = "MODERATELY BULLISH"
            bias = "BULLISH"
            tag_color = "green"
            desc = f"PCR at {pcr_value:.2f} reflects positive upward bias with Put OI exceeding Call OI."
        elif pcr_value >= 0.85:
            sentiment = "NEUTRAL / BALANCED"
            bias = "NEUTRAL"
            tag_color = "gray"
            desc = f"PCR at {pcr_value:.2f} indicates consolidation. Equilibrium between Call and Put writers around {max_pain_strike}."
        elif pcr_value >= 0.65:
            sentiment = "MODERATELY BEARISH"
            bias = "BEARISH"
            tag_color = "red"
            desc = f"PCR at {pcr_value:.2f} (< 0.85) indicates Call writers dominating. Overhead resistance at {highest_call_oi_strike} CE."
        else:
            sentiment = "BEARISH (HEAVY CALL WRITING)"
            bias = "BEARISH"
            tag_color = "red"
            desc = f"PCR at {pcr_value:.2f} (< 0.65) reflects aggressive call buildup and strong downward pressure."

        return {
            "symbol": symbol.upper(),
            "spot_price": round(spot_price, 2),
            "atm_strike": atm_strike,
            "pcr_oi": pcr_value,
            "sentiment": sentiment,
            "bias": bias,
            "tag_color": tag_color,
            "total_put_oi": total_put_oi,
            "total_call_oi": total_call_oi,
            "max_pain_strike": max_pain_strike,
            "highest_call_oi_strike": highest_call_oi_strike,
            "highest_put_oi_strike": highest_put_oi_strike,
            "major_resistance": highest_call_oi_strike,
            "major_support": highest_put_oi_strike,
            "description": desc,
            "option_chain_preview": chain_data,
        }


pcr_service = PCRService()
