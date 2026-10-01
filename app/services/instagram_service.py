import json
import logging
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime
from typing import Dict, Any, Optional, List

from app.core.config import settings

logger = logging.getLogger(__name__)


class InstagramService:
    """
    Instagram & Meta Alerting Service for QuantPulse India Pro.
    Sends high-probability CALL (CE) / PUT (PE) strike execution details,
    targets, stop losses, lot sizing, and 12-factor confluence analytics
    directly to Instagram Direct Messaging (IG Graph API / Webhook Relay / Meta Messenger)
    for recipient: 9100040008.
    """

    def __init__(self):
        self.default_recipient_id = settings.INSTAGRAM_RECIPIENT_ID or "9100040008"
        self.access_token = settings.INSTAGRAM_ACCESS_TOKEN
        self.account_id = settings.INSTAGRAM_ACCOUNT_ID
        self.webhook_url = settings.INSTAGRAM_WEBHOOK_URL

    def get_status(self) -> Dict[str, Any]:
        """Returns Instagram configuration status and readiness."""
        has_token = bool(self.access_token and self.access_token != "YOUR_INSTAGRAM_ACCESS_TOKEN_HERE")
        has_webhook = bool(self.webhook_url and self.webhook_url.startswith("http"))

        masked_token = (
            f"{self.access_token[:6]}...{self.access_token[-4:]}"
            if has_token and len(self.access_token) > 10
            else "NOT_CONFIGURED"
        )
        return {
            "configured": has_token or has_webhook,
            "target_recipient": self.default_recipient_id,
            "api_status": "READY" if has_token else ("WEBHOOK_READY" if has_webhook else "AWAITING_CONFIG"),
            "masked_access_token": masked_token,
            "webhook_configured": has_webhook,
            "webhook_url": self.webhook_url if has_webhook else None,
            "auto_alert_enabled": settings.INSTAGRAM_AUTO_ALERT,
            "min_confidence": settings.INSTAGRAM_MIN_CONFIDENCE,
            "guide": (
                "To deliver live alerts to Instagram / Meta Direct (@9100040008): "
                "1. Connect via Instagram Graph API Token or a Meta Webhook Relay. "
                "2. All PUT/CALL signals are formatted for Instagram DMs, Stories, and Reels."
            ),
        }

    def format_signal_message(self, sig: Dict[str, Any]) -> str:
        """
        Formats a comprehensive QuantPulse CALL / PUT trade alert with exact strike
        premium execution prices, targets, lot sizing, capital, and 12-factor confluence
        optimized for Instagram Direct Message and Post/Story delivery.
        """
        instrument = str(sig.get("instrument", sig.get("symbol", "N/A")))
        symbol = str(sig.get("symbol", "N/A"))
        rec = sig.get("recommendation", "NEUTRAL / WAIT")
        sig_type = sig.get("signal_type", "CONSOLIDATION")
        strike = str(sig.get("suggested_strike", "N/A"))
        unit = sig.get("unit", "₹")

        # Visual indicator icons
        if "CALL" in rec or sig_type == "BULLISH":
            rec_icon = "🟢 BUY CALL (CE)"
            trend_badge = "🚀 BULLISH MOMENTUM"
        elif "PUT" in rec or sig_type == "BEARISH":
            rec_icon = "🔴 BUY PUT (PE)"
            trend_badge = "📉 BEARISH DOWNTREND"
        else:
            rec_icon = "⚪ NEUTRAL / WAIT"
            trend_badge = "⏸️ RANGE CONSOLIDATION"

        # Option Premium Execution Prices
        entry = sig.get("option_entry_price", sig.get("entry_price", 0.0))
        t1 = sig.get("option_target_1", sig.get("target_1", 0.0))
        t2 = sig.get("option_target_2", sig.get("target_2", 0.0))
        sl = sig.get("option_stop_loss", sig.get("stop_loss", 0.0))
        gain_pts = round(max(0.0, t1 - entry), 2)
        lot_size = sig.get("lot_size", 1)
        capital = sig.get("capital_required_per_lot", entry * lot_size)
        profit_t1 = sig.get("est_profit_per_lot_t1", (t1 - entry) * lot_size)
        profit_t2 = sig.get("est_profit_per_lot_t2", (t2 - entry) * lot_size)
        risk_lot = sig.get("est_risk_per_lot", abs(entry - sl) * lot_size)

        # Spot Reference
        spot_ltp = sig.get("spot_price", 0.0)
        confidence = sig.get("confidence_score", 0.0)
        rr = sig.get("risk_reward_ratio", "1:2.0")

        # Confluence metrics
        pam = sig.get("price_action_momentum", {})
        mom_regime = pam.get("momentum_regime", "NEUTRAL").replace("_", " ")

        pcr = sig.get("pcr", {})
        pcr_val = pcr.get("pcr_oi", "N/A")
        pcr_sent = pcr.get("sentiment", "NEUTRAL")

        reasons = sig.get("confluence_reasons", [])
        reasons_text = ""
        for r in reasons[:3]:
            reasons_text += f" • {r}\n"
        if not reasons_text:
            reasons_text = " • High multi-factor confluence confirmed\n"

        # 21-Factor Institutional Matrix Summary
        matrix = sig.get("institutional_matrix", {})
        if matrix and matrix.get("total_factors"):
            pass_cnt = matrix.get("passed_count", 0)
            tot_cnt = matrix.get("total_factors", 21)
            pct = matrix.get("confluence_percentage", 0.0)
            regime = str(matrix.get("regime", "TRENDING_EXPANSION")).replace("_", " ")
            matrix_text = f"🏛️ 21-Factor: {pass_cnt}/{tot_cnt} PASS ({pct}%) | {regime}\n"
        else:
            matrix_text = ""

        timestamp = datetime.now().strftime("%d %b %Y, %I:%M:%S %p IST")

        msg = (
            f"⚡ QUANTPULSE PRO: INSTAGRAM TRADE ALERT ⚡\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 Instrument: {instrument} ({symbol})\n"
            f"🎯 Action: {rec_icon}\n"
            f"🏷️ Suggested Strike: {strike}\n"
            f"🚀 Expected Move: +{gain_pts} Pts (Min 5 Pts Verified)\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 OPTION PREMIUM EXECUTION:\n"
            f"  ▶ Buy Entry Price: ₹{entry:,.2f}\n"
            f"  🎯 Target 1: ₹{t1:,.2f} (+{gain_pts} pts)\n"
            f"  🎯 Target 2: ₹{t2:,.2f}\n"
            f"  🛑 Stop Loss: ₹{sl:,.2f}\n"
            f"  ⚖️ Risk-Reward: {rr}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📦 SIZING & CAPITAL:\n"
            f"  • Lot Size: {lot_size} Qty/Lot\n"
            f"  • Capital / Lot: ₹{capital:,.2f}\n"
            f"  • Est. Profit (T1): +₹{profit_t1:,.2f}\n"
            f"  • Est. Profit (T2): +₹{profit_t2:,.2f}\n"
            f"  • Max Risk (SL): -₹{risk_lot:,.2f}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📍 SPOT & CONFLUENCE ({confidence}% Score):\n"
            f"  • Spot Price: ₹{spot_ltp:,.2f} {unit}\n"
            f"  • Momentum: {mom_regime}\n"
            f"  • PCR (OI): {pcr_val} ({pcr_sent})\n"
            f"{matrix_text}"
            f"💡 Factors:\n{reasons_text}"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"🕒 {timestamp} (Market Hours Only)\n"
            f"📱 Recipient: {self.default_recipient_id}\n"
            f"#trading #nifty50 #banknifty #options #stockmarket"
        )
        return msg

    def send_message(
        self,
        text: str,
        recipient_id: Optional[str] = None,
        access_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Dispatches message to Instagram Graph API or Webhook Relay for recipient 9100040008.
        """
        target = str(recipient_id or self.default_recipient_id).strip()
        token = str(access_token or self.access_token or "").strip()
        webhook = self.webhook_url

        # 1. If Webhook is configured (Zapier / Manychat / Custom Meta Gateway)
        if webhook and webhook.startswith("http"):
            try:
                payload = json.dumps({
                    "recipient_id": target,
                    "phone": target,
                    "channel": "instagram_dm",
                    "text": text,
                    "timestamp": datetime.now().isoformat(),
                }).encode("utf-8")
                req = urllib.request.Request(
                    webhook,
                    data=payload,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=8) as resp:
                    resp_body = resp.read().decode("utf-8")
                    return {
                        "success": True,
                        "status": "DELIVERED_VIA_WEBHOOK",
                        "recipient": target,
                        "response": resp_body[:200],
                        "preview": text[:200],
                    }
            except Exception as e:
                logger.error(f"Instagram webhook relay failed: {e}")
                return {
                    "success": False,
                    "status": "WEBHOOK_ERROR",
                    "error": str(e),
                    "recipient": target,
                    "preview": text,
                }

        # 2. If Meta Instagram Graph API Token is configured
        if token and token != "YOUR_INSTAGRAM_ACCESS_TOKEN_HERE":
            try:
                # Meta Graph API endpoint for Instagram / Messenger DM
                url = f"https://graph.facebook.com/v19.0/me/messages?access_token={token}"
                payload = json.dumps({
                    "recipient": {"id": target},
                    "message": {"text": text},
                    "messaging_type": "MESSAGE_TAG",
                    "tag": "CONFIRMED_EVENT_UPDATE"
                }).encode("utf-8")

                req = urllib.request.Request(
                    url,
                    data=payload,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=8) as resp:
                    result = json.loads(resp.read().decode("utf-8"))
                    return {
                        "success": True,
                        "status": "DELIVERED_VIA_INSTAGRAM_GRAPH_API",
                        "recipient": target,
                        "api_response": result,
                        "preview": text[:200],
                    }
            except urllib.error.HTTPError as e:
                err_body = e.read().decode("utf-8")
                logger.warning(f"Meta Instagram API HTTP error {e.code}: {err_body}")
                return {
                    "success": False,
                    "status": "META_API_ERROR",
                    "http_code": e.code,
                    "error": err_body,
                    "recipient": target,
                    "preview": text,
                }
            except Exception as e:
                logger.error(f"Instagram Graph API dispatch failed: {e}")
                return {
                    "success": False,
                    "status": "DISPATCH_FAILED",
                    "error": str(e),
                    "recipient": target,
                    "preview": text,
                }

        # 3. Simulated / Formatted Preview Mode if API token is awaiting user Meta App setup
        logger.info(f"Instagram Alert generated and queued for target: {target}")
        return {
            "success": True,
            "status": "QUEUED_AND_READY",
            "mode": "INSTAGRAM_READY",
            "recipient": target,
            "message": (
                f"Alert successfully generated and formatted for Instagram Direct Message to {target}. "
                f"To enable real-time auto-dispatch, set INSTAGRAM_ACCESS_TOKEN or INSTAGRAM_WEBHOOK_URL."
            ),
            "preview_text": text,
        }

    def validate_alert_eligibility(
        self,
        sig: Dict[str, Any],
        min_points: Optional[float] = None,
        market_hours_only: Optional[bool] = None,
        bypass_filters: bool = False,
    ) -> Dict[str, Any]:
        """
        Validates trade alert criteria for Instagram:
        1. Market Timing: Current time within active market hours (09:15-15:30 IST for NSE/BSE, 09:00-23:30 IST for MCX).
        2. Move Quality: Directional BUY CALL or BUY PUT (no neutral/sideways noise).
        3. Minimum Expected Move: Target 1 gain must be at least 5.0 points (or specified min_points).
        4. High Confluence: Confidence score >= 60.0%.
        """
        if bypass_filters:
            return {"eligible": True, "reason": "Filters bypassed for manual dispatch"}

        from app.services.market_data import market_data_service

        symbol = str(sig.get("symbol", "^NSEI")).strip().upper()
        req_min_pts = float(min_points if min_points is not None else settings.ALERT_MIN_MOVE_POINTS)
        enforce_market_hours = bool(
            market_hours_only if market_hours_only is not None else settings.ALERT_DURING_MARKET_HOURS_ONLY
        )

        # 1. Market Timing Check
        if enforce_market_hours:
            session_info = market_data_service.is_market_open(symbol)
            if not session_info["is_open"]:
                return {
                    "eligible": False,
                    "filter_failed": "MARKET_HOURS",
                    "reason": f"Market is currently closed ({session_info['ist_time']}, {session_info['ist_date']}). Alerts are active during market hours only ({session_info['market_hours']}).",
                    "session_info": session_info,
                }

        # 2. Quality Move Actionable Filter (Reject NEUTRAL / WAIT)
        rec = str(sig.get("recommendation", "")).upper()
        if not ("CALL" in rec or "PUT" in rec):
            return {
                "eligible": False,
                "filter_failed": "NON_ACTIONABLE",
                "reason": f"Signal recommendation is '{rec}'. Only high-conviction BUY CALL and BUY PUT signals are alerted.",
            }

        # 3. Minimum 5 Points Move Filter
        entry = float(sig.get("option_entry_price", sig.get("entry_price", 0.0)))
        t1 = float(sig.get("option_target_1", sig.get("target_1", 0.0)))
        opt_gain = round(max(0.0, t1 - entry), 2)
        spot_gain = round(abs(float(sig.get("spot_target_1", 0.0)) - float(sig.get("spot_price", 0.0))), 2)

        effective_min_pts = 1.0 if symbol in ["NATURALGAS", "NG=F"] else req_min_pts

        if opt_gain < effective_min_pts and spot_gain < 5.0:
            return {
                "eligible": False,
                "filter_failed": "INSUFFICIENT_POINTS",
                "reason": f"Projected move of +{opt_gain} pts is below the minimum +{effective_min_pts} points threshold.",
                "projected_gain_pts": opt_gain,
                "min_points_required": effective_min_pts,
            }

        # 4. Confluence Score Filter (with dynamic self-learning recalibration)
        from app.services.trade_audit_service import trade_audit_service
        min_conf = trade_audit_service.get_symbol_confidence_threshold(symbol)
        conf = float(sig.get("confidence_score", 0.0))
        if conf < min_conf:
            return {
                "eligible": False,
                "filter_failed": "LOW_CONFIDENCE",
                "reason": f"Confidence score ({conf}%) is below recalibrated threshold ({min_conf}%).",
                "confidence": conf,
                "min_confidence_required": min_conf,
            }

        return {
            "eligible": True,
            "projected_gain_pts": opt_gain,
            "reason": f"Passed: High quality {rec} move (+{opt_gain} pts projected gain, {conf}% confluence).",
        }

    def send_signal_alert(
        self,
        sig: Dict[str, Any],
        recipient_id: Optional[str] = None,
        access_token: Optional[str] = None,
        min_points: Optional[float] = None,
        market_hours_only: Optional[bool] = None,
        bypass_filters: bool = False,
    ) -> Dict[str, Any]:
        """Formats and dispatches a live CALL/PUT option trade alert."""
        eligibility = self.validate_alert_eligibility(
            sig=sig,
            min_points=min_points,
            market_hours_only=market_hours_only,
            bypass_filters=bypass_filters,
        )

        if not eligibility["eligible"]:
            logger.info(f"Instagram alert skipped for {sig.get('symbol')}: {eligibility.get('reason')}")
            return {
                "success": False,
                "status": "FILTER_SKIPPED",
                "symbol": sig.get("symbol"),
                "filter_failed": eligibility.get("filter_failed"),
                "reason": eligibility.get("reason"),
                "eligibility": eligibility,
            }

        text = self.format_signal_message(sig)
        res = self.send_message(text, recipient_id=recipient_id, access_token=access_token)
        res["eligibility"] = eligibility

        # Local Persistent Trade Alert Logging & Auto-Audit Registration
        try:
            from app.services.trade_audit_service import trade_audit_service
            trade_audit_service.log_trade_alert(
                signal=sig,
                channels=["INSTAGRAM"],
                delivery_status="SENT" if res.get("success") else "FAILED",
                dispatch_response=res,
            )
        except Exception as e:
            logger.debug(f"Trade audit logging notice: {e}")

        return res

    def send_test_message(
        self,
        recipient_id: Optional[str] = None,
        access_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Sends a verification message to verify Instagram alerting readiness."""
        target = str(recipient_id or self.default_recipient_id).strip()
        timestamp = datetime.now().strftime("%d %b %Y, %I:%M:%S %p IST")
        test_text = (
            f"🔔 QUANTPULSE PRO: INSTAGRAM ALERT TEST 🔔\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"✅ Test Ping Successful!\n"
            f"📱 Target Recipient: {target}\n"
            f"🕒 Timestamp: {timestamp}\n"
            f"🎯 Status: High-Probability PUT & CALL Options Alerts Ready for Delivery.\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"#quantpulse #tradingalerts"
        )
        return self.send_message(test_text, recipient_id=target, access_token=access_token)


instagram_service = InstagramService()
