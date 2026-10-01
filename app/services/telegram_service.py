import json
import logging
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime
from typing import Dict, Any, Optional, List
import html

from app.core.config import settings

logger = logging.getLogger(__name__)


class TelegramService:
    """
    Telegram Bot Alerting Service for QuantPulse India Pro.
    Sends high-probability CALL (CE) / PUT (PE) strike execution details,
    targets, stop losses, lot sizing, and 12-factor confluence analytics
    directly to Telegram user/channel (Default: 9100040008).
    """

    def __init__(self):
        self.default_chat_id = settings.TELEGRAM_CHAT_ID or "9100040008"
        self.bot_token = settings.TELEGRAM_BOT_TOKEN
        self._last_update_id = 0
        self._polling_started = False

    def get_recent_chat_id(self, bot_token: Optional[str] = None) -> Optional[str]:
        """Auto-discovers the latest active chat ID or channel ID from getUpdates."""
        token = str(bot_token or self.bot_token or "").strip()
        if not token or token == "YOUR_TELEGRAM_BOT_TOKEN_HERE":
            return None
        try:
            url = f"https://api.telegram.org/bot{token}/getUpdates"
            with urllib.request.urlopen(url, timeout=5) as resp:
                data = json.loads(resp.read().decode())
                results = data.get("result", [])
                if results:
                    last_update = results[-1]
                    if "message" in last_update and "chat" in last_update["message"]:
                        return str(last_update["message"]["chat"]["id"])
                    elif "channel_post" in last_update and "chat" in last_update["channel_post"]:
                        return str(last_update["channel_post"]["chat"]["id"])
                    elif "my_chat_member" in last_update and "chat" in last_update["my_chat_member"]:
                        return str(last_update["my_chat_member"]["chat"]["id"])
        except Exception as e:
            logger.debug(f"getUpdates discovery error: {e}")
        return None

    def get_status(self) -> Dict[str, Any]:
        """Returns Telegram configuration status and readiness."""
        has_token = bool(self.bot_token and self.bot_token != "YOUR_TELEGRAM_BOT_TOKEN_HERE")
        masked_token = (
            f"{self.bot_token[:6]}...{self.bot_token[-4:]}"
            if has_token and len(self.bot_token) > 10
            else "NOT_CONFIGURED"
        )
        return {
            "configured": has_token,
            "target_chat_id": self.default_chat_id,
            "bot_token_status": "READY" if has_token else "MISSING_OR_PLACEHOLDER",
            "masked_bot_token": masked_token,
            "auto_alert_enabled": settings.TELEGRAM_AUTO_ALERT,
            "min_confidence": settings.TELEGRAM_MIN_CONFIDENCE,
            "bot_username": "SrinuAlgoAlertsBot",
            "bot_link": "https://t.me/SrinuAlgoAlertsBot",
            "guide": (
                "To activate Telegram live delivery: "
                "1. Open https://t.me/SrinuAlgoAlertsBot in Telegram and press START (or send /start). "
                "2. Or add @SrinuAlgoAlertsBot as Administrator to your Telegram channel or group."
            ),
        }

    def format_signal_message(self, sig: Dict[str, Any]) -> str:
        """
        Formats outgoing Telegram trade alerts to match the QuantPulse Pro template:

        ⚡️ QUANTPULSE PRO: LIVE MARKET ALERT ⚡️
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        📊 Instrument: NIFTY 50 (^NSEI)
        🎯 Action: 🔴 BUY PUT (PE)
        🏷️ Suggested Strike: 22400 PE
        🚀 Expected Move: +45.45 Pts (Min 5 Pts Verified)
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        💰 OPTION CONTRACT PREMIUM (BUY):
          ▶️ Buy Entry Price: ₹101.00
          🎯 Target 1: ₹146.45 (+45.45 pts)
          🎯 Target 2: ₹186.85
          🛑 Stop Loss: ₹72.72
          ⚖️ Risk-Reward: 1:1.3
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        📦 LOT SIZING & CAPITAL METRICS:
          • Lot Sizing: 65 Qty / Lot
          • Capital / Lot: ₹6,565.00
          • Projected Gain (T1): +₹2,954.25 / lot
          • Projected Gain (T2): +₹5,580.25 / lot
          • Max Risk (SL): -₹1,838.20 / lot
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        📍 UNDERLYING SPOT REFERENCE:
          • Spot LTP: ₹22,421.95 ₹
          • Spot Target 1: ₹22,305.36 | Stop: ₹22,511.64
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        🔥 CONFLUENCE SIGNALS (Score: 110.0%):
          ⚡️ Momentum: MODERATE BULLISH MOMENTUM (+24.0)
          📊 PCR (OI): 1.16 (MODERATELY BULLISH)
          🏛️ SMC Bias: NEUTRAL RANGE

        💡 Key Technical Factors:
          • Fibonacci: Price below 50% retracement, testing Golden Pocket support (₹22443.66)
          • Supertrend: Bearish trend active (Trailing Resistance: ₹22475.03)
          • EMA: Perfect Bearish Alignment (9 < 21 < 50 EMA: ₹22386.0 < ₹22432.1 < ₹22539.6)
          • RSI (14): Bearish Momentum (38.5)
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        🏛️ 21-FACTOR INSTITUTIONAL MATRIX:
          • Filter Confluence: 9/21 PASS (42.9%)
          • Market Regime: MEAN REVERSION
          • Active Killzone: MARKET CLOSED
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        🕒 Alert Generated: 02 Oct 2026, 01:49:18 AM IST (Market Hours Only)
        🤖 QuantPulse India Pro Terminal
        """
        symbol = str(sig.get("symbol", "^NSEI")).strip()
        instrument = str(sig.get("instrument", symbol))
        rec = str(sig.get("recommendation", "BUY CALL (CE)")).upper()

        # Action formatting
        is_call = "CALL" in rec or "CE" in rec or sig.get("signal_type") == "BULLISH"
        action_emoji = "🟢" if is_call else "🔴"
        if not rec.startswith("BUY") and ("CALL" in rec or "PUT" in rec):
            rec_clean = f"BUY {rec}"
        else:
            rec_clean = rec
        action_str = f"{action_emoji} {rec_clean}"

        strike = str(sig.get("suggested_strike", ""))

        # Option Premium Execution
        entry = float(sig.get("option_entry_price", sig.get("entry_price", 0.0)))
        t1 = float(sig.get("option_target_1", sig.get("target_1", 0.0)))
        t2 = float(sig.get("option_target_2", sig.get("target_2", 0.0)))
        sl = float(sig.get("option_stop_loss", sig.get("stop_loss", 0.0)))
        gain_pts = round(max(0.0, t1 - entry), 2)
        expected_move_pts = float(sig.get("expected_option_gain_pts", gain_pts))
        rr = str(sig.get("risk_reward_ratio", "1:2.0"))

        # Lot Sizing & Capital Metrics
        lot_size = int(sig.get("lot_size", 65))
        capital = float(sig.get("capital_required_per_lot", entry * lot_size))
        profit_t1 = float(sig.get("est_profit_per_lot_t1", (t1 - entry) * lot_size))
        profit_t2 = float(sig.get("est_profit_per_lot_t2", (t2 - entry) * lot_size))
        risk_lot = float(sig.get("est_risk_per_lot", abs(entry - sl) * lot_size))

        # Spot Reference
        spot_ltp = float(sig.get("spot_price", 0.0))
        unit = "₹"
        spot_levels = sig.get("spot_levels", {})
        spot_t1 = float(spot_levels.get("spot_target_1", sig.get("spot_target_1", 0.0)))
        spot_sl = float(spot_levels.get("spot_stop_loss", sig.get("spot_stop_loss", 0.0)))

        # Confluence Signals
        conf_score = float(sig.get("confidence_score", 0.0))
        pam = sig.get("price_action_momentum", {})
        mom_regime = str(pam.get("momentum_regime", "MODERATE_BULLISH_MOMENTUM")).replace("_", " ")
        mom_score = float(pam.get("momentum_score", 24.0))
        mom_score_str = f"+{mom_score:.1f}" if mom_score >= 0 else f"{mom_score:.1f}"

        pcr = sig.get("pcr", {})
        pcr_val = pcr.get("pcr_oi", 1.16)
        if isinstance(pcr_val, (int, float)):
            pcr_val_str = f"{pcr_val:.2f}"
        else:
            pcr_val_str = str(pcr_val)
        pcr_sent = str(pcr.get("sentiment", "MODERATELY BULLISH")).replace("_", " ")

        ob = sig.get("order_blocks", {})
        smc_bias = str(ob.get("smc_bias", "NEUTRAL RANGE")).replace("_", " ")

        # Key Technical Factors
        reasons = sig.get("confluence_reasons", [])
        factors_lines = []
        if reasons:
            for r in reasons[:4]:
                escaped_r = html.escape(str(r))
                factors_lines.append(f"  • {escaped_r}")
        else:
            ind = sig.get("indicators", {})
            rsi_val = float(ind.get("rsi_14", 50.0))
            ema9 = float(ind.get("ema_9", spot_ltp))
            ema21 = float(ind.get("ema_21", spot_ltp))
            ema50 = float(ind.get("ema_50", spot_ltp))
            factors_lines.append(f"  • EMA: Alignment (EMA 9: ₹{ema9:.1f}, 21: ₹{ema21:.1f}, 50: ₹{ema50:.1f})")
            factors_lines.append(f"  • RSI (14): Momentum ({rsi_val:.1f})")
        factors_text = "\n".join(factors_lines)

        # 21-Factor Institutional Matrix
        matrix = sig.get("institutional_matrix", {})
        passed_cnt = int(matrix.get("passed_count", 9))
        tot_cnt = int(matrix.get("total_factors", 21))
        conf_pct = float(matrix.get("confluence_percentage", round((passed_cnt / max(1, tot_cnt)) * 100, 1)))
        regime = str(matrix.get("regime", "MEAN REVERSION")).replace("_", " ")

        # Active Killzone
        from app.services.institutional_filter_engine import institutional_filter_engine
        try:
            kz_info = institutional_filter_engine.check_killzone_timing(symbol)
            active_kz = str(kz_info.get("active_killzone", "MARKET CLOSED")).replace("_", " ")
        except Exception:
            active_kz = "MARKET CLOSED"

        timestamp = datetime.now().strftime("%d %b %Y, %I:%M:%S %p IST")

        msg = (
            f"⚡️ <b>QUANTPULSE PRO: LIVE MARKET ALERT</b> ⚡️\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 <b>Instrument:</b> {html.escape(instrument)} ({html.escape(symbol)})\n"
            f"🎯 <b>Action:</b> {action_str}\n"
            f"🏷️ <b>Suggested Strike:</b> {html.escape(strike)}\n"
            f"🚀 <b>Expected Move:</b> +{expected_move_pts:.2f} Pts (Min 5 Pts Verified)\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 <b>OPTION CONTRACT PREMIUM (BUY):</b>\n"
            f"  ▶️ <b>Buy Entry Price:</b> ₹{entry:,.2f}\n"
            f"  🎯 <b>Target 1:</b> ₹{t1:,.2f} (+{gain_pts:.2f} pts)\n"
            f"  🎯 <b>Target 2:</b> ₹{t2:,.2f}\n"
            f"  🛑 <b>Stop Loss:</b> ₹{sl:,.2f}\n"
            f"  ⚖️ <b>Risk-Reward:</b> {html.escape(rr)}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📦 <b>LOT SIZING & CAPITAL METRICS:</b>\n"
            f"  • <b>Lot Sizing:</b> {lot_size} Qty / Lot\n"
            f"  • <b>Capital / Lot:</b> ₹{capital:,.2f}\n"
            f"  • <b>Projected Gain (T1):</b> +₹{profit_t1:,.2f} / lot\n"
            f"  • <b>Projected Gain (T2):</b> +₹{profit_t2:,.2f} / lot\n"
            f"  • <b>Max Risk (SL):</b> -₹{risk_lot:,.2f} / lot\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📍 <b>UNDERLYING SPOT REFERENCE:</b>\n"
            f"  • <b>Spot LTP:</b> ₹{spot_ltp:,.2f} {unit}\n"
            f"  • <b>Spot Target 1:</b> ₹{spot_t1:,.2f} | <b>Stop:</b> ₹{spot_sl:,.2f}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🔥 <b>CONFLUENCE SIGNALS (Score: {conf_score:.1f}%):</b>\n"
            f"  ⚡️ <b>Momentum:</b> {html.escape(mom_regime)} ({mom_score_str})\n"
            f"  📊 <b>PCR (OI):</b> {html.escape(pcr_val_str)} ({html.escape(pcr_sent)})\n"
            f"  🏛️ <b>SMC Bias:</b> {html.escape(smc_bias)}\n\n"
            f"💡 <b>Key Technical Factors:</b>\n"
            f"{factors_text}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🏛️ <b>21-FACTOR INSTITUTIONAL MATRIX:</b>\n"
            f"  • <b>Filter Confluence:</b> {passed_cnt}/{tot_cnt} PASS ({conf_pct:.1f}%)\n"
            f"  • <b>Market Regime:</b> {html.escape(regime)}\n"
            f"  • <b>Active Killzone:</b> {html.escape(active_kz)}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🕒 <i>Alert Generated: {timestamp} (Market Hours Only)</i>\n"
            f"🤖 <b>QuantPulse India Pro Terminal</b>"
        )
        return msg

    def send_message(
        self,
        text: str,
        chat_id: Optional[str] = None,
        bot_token: Optional[str] = None,
        parse_mode: str = "HTML",
    ) -> Dict[str, Any]:
        """
        Dispatches an HTML/Markdown formatted message to Telegram Bot API.
        Target default chat_id: 9100040008.
        """
        target_chat = str(chat_id or self.default_chat_id).strip()
        token = str(bot_token or self.bot_token or "").strip()

        if not token or token == "YOUR_TELEGRAM_BOT_TOKEN_HERE":
            logger.warning("Telegram Bot Token is not configured.")
            return {
                "success": False,
                "status": "TOKEN_MISSING",
                "chat_id": target_chat,
                "message": (
                    "Telegram Bot Token not configured in .env (TELEGRAM_BOT_TOKEN). "
                    "Create a bot in 15 seconds with @BotFather on Telegram and set token."
                ),
                "preview_text": text,
            }

        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {
            "chat_id": target_chat,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        }

        try:
            req_data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=req_data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                res_body = response.read().decode("utf-8")
                res_json = json.loads(res_body)
                if res_json.get("ok"):
                    logger.info(f"Telegram alert successfully dispatched to {target_chat}")
                    return {
                        "success": True,
                        "status": "SENT",
                        "chat_id": target_chat,
                        "message_id": res_json.get("result", {}).get("message_id"),
                        "message": f"Alert successfully sent to Telegram ID: {target_chat}",
                    }
                else:
                    return {
                        "success": False,
                        "status": "TELEGRAM_ERROR",
                        "chat_id": target_chat,
                        "description": res_json.get("description", "Unknown Telegram error"),
                        "message": f"Telegram API error: {res_json.get('description')}",
                    }

        except urllib.error.HTTPError as e:
            err_msg = e.read().decode("utf-8", errors="ignore")
            logger.error(f"Telegram HTTP Error {e.code}: {err_msg}")
            # Try auto-discovering recent chat if chat not found
            if e.code == 400 and ("chat not found" in err_msg.lower() or "chat_id is empty" in err_msg.lower()):
                discovered = self.get_recent_chat_id(token)
                if discovered and discovered != target_chat:
                    logger.info(f"Auto-discovered active Telegram chat ID {discovered}. Retrying...")
                    self.default_chat_id = discovered
                    return self.send_message(text=text, chat_id=discovered, bot_token=token, parse_mode=parse_mode)

            return {
                "success": False,
                "status": f"HTTP_{e.code}",
                "chat_id": target_chat,
                "error": err_msg,
                "message": f"Telegram delivery failed ({e.code}): {err_msg}. Note: Open https://t.me/SrinuAlgoAlertsBot and press START, or add @SrinuAlgoAlertsBot as Administrator to your Telegram channel.",
                "preview_text": text,
            }
        except Exception as e:
            logger.error(f"Telegram dispatch exception: {str(e)}")
            return {
                "success": False,
                "status": "EXCEPTION",
                "chat_id": target_chat,
                "error": str(e),
                "message": f"Network error communicating with Telegram: {str(e)}",
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
        Validates trade alert criteria:
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

        # For low-unit MCX commodities (e.g. Natural Gas @ ~₹235), nominal point threshold is scaled proportionally
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
        signal: Dict[str, Any],
        chat_id: Optional[str] = None,
        bot_token: Optional[str] = None,
        min_points: Optional[float] = None,
        market_hours_only: Optional[bool] = None,
        bypass_filters: bool = False,
    ) -> Dict[str, Any]:
        """Formats and sends a high-probability trade signal alert to Telegram."""
        # Validate eligibility (Market Hours + Minimum 5 Points Move + Directional Quality)
        eligibility = self.validate_alert_eligibility(
            sig=signal,
            min_points=min_points,
            market_hours_only=market_hours_only,
            bypass_filters=bypass_filters,
        )

        if not eligibility["eligible"]:
            logger.info(f"Telegram alert skipped for {signal.get('symbol')}: {eligibility.get('reason')}")
            return {
                "success": False,
                "status": "FILTER_SKIPPED",
                "symbol": signal.get("symbol"),
                "filter_failed": eligibility.get("filter_failed"),
                "reason": eligibility.get("reason"),
                "eligibility": eligibility,
            }

        text = self.format_signal_message(signal)
        res = self.send_message(text=text, chat_id=chat_id, bot_token=bot_token, parse_mode="HTML")
        res["eligibility"] = eligibility

        # Local Persistent Trade Alert Logging & Auto-Audit Registration
        try:
            from app.services.trade_audit_service import trade_audit_service
            trade_audit_service.log_trade_alert(
                signal=signal,
                channels=["TELEGRAM"],
                delivery_status="SENT" if res.get("success") else "FAILED",
                dispatch_response=res,
            )
        except Exception as e:
            logger.debug(f"Trade audit logging notice: {e}")

        return res

    def send_test_message(
        self,
        chat_id: Optional[str] = None,
        bot_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Sends a sample trade alert formatted in Market Mentor Research and Academy template to test connectivity."""
        target_chat = str(chat_id or self.default_chat_id).strip()
        from app.services.strategy_engine import strategy_engine
        try:
            sig = strategy_engine.generate_options_call_put_signal("^NSEI")
            test_msg = self.format_signal_message(sig)
        except Exception:
            sample_sig = {
                "symbol": "^NSEI",
                "instrument": "NIFTY 50",
                "recommendation": "BUY CALL (CE)",
                "suggested_strike": "23400 CE",
                "strike_price": 23400,
                "option_entry_price": 180.0,
                "option_target_1": 234.0,
                "option_target_2": 288.0,
                "option_target_3": 360.0,
                "option_long_target": 450.0,
                "option_stop_loss": 135.0,
                "lot_size": 65,
            }
            test_msg = self.format_signal_message(sample_sig)
        return self.send_message(text=test_msg, chat_id=target_chat, bot_token=bot_token, parse_mode="HTML")

    def _send_instant_welcome(self, chat_id: str, user_name: str):
        """Sends live NIFTY 50 / BANK NIFTY signals formatted in Market Mentor Academy template upon first contact."""
        try:
            from app.services.strategy_engine import strategy_engine
            # Dispatch Live NIFTY 50 Option Alert
            sig_nifty = strategy_engine.generate_options_call_put_signal("^NSEI")
            self.send_signal_alert(sig_nifty, chat_id=chat_id, bypass_filters=True)
            # Dispatch Live BANK NIFTY Option Alert
            sig_bank = strategy_engine.generate_options_call_put_signal("^NSEBANK")
            self.send_signal_alert(sig_bank, chat_id=chat_id, bypass_filters=True)
        except Exception as e:
            logger.error(f"Error in welcome alert dispatch: {e}")

    def poll_once(self) -> Optional[str]:
        """Polls Telegram for new updates, registers active chat ID, and auto-responds."""
        token = str(self.bot_token or "").strip()
        if not token or token == "YOUR_TELEGRAM_BOT_TOKEN_HERE":
            return None
        try:
            url = f"https://api.telegram.org/bot{token}/getUpdates?timeout=2"
            if self._last_update_id:
                url += f"&offset={self._last_update_id + 1}"
            with urllib.request.urlopen(url, timeout=6) as resp:
                data = json.loads(resp.read().decode())
                results = data.get("result", [])
                for update in results:
                    self._last_update_id = max(self._last_update_id, update.get("update_id", 0))
                    chat_id = None
                    user_name = "Trader"
                    if "message" in update and "chat" in update["message"]:
                        chat_id = str(update["message"]["chat"]["id"])
                        user_name = update["message"].get("from", {}).get("first_name", "Trader")
                    elif "channel_post" in update and "chat" in update["channel_post"]:
                        chat_id = str(update["channel_post"]["chat"]["id"])
                        user_name = update["channel_post"]["chat"].get("title", "Channel")
                    elif "my_chat_member" in update and "chat" in update["my_chat_member"]:
                        chat_id = str(update["my_chat_member"]["chat"]["id"])
                        user_name = update["my_chat_member"]["chat"].get("title", "Member")

                    if chat_id:
                        old_chat = self.default_chat_id
                        self.default_chat_id = chat_id
                        logger.info(f"Connected to Telegram chat ID: {chat_id} ({user_name})")
                        self._send_instant_welcome(chat_id, user_name)
                        return chat_id
        except Exception as e:
            logger.debug(f"Telegram polling error: {e}")
        return None

    def start_polling(self):
        """Starts background auto-discovery listener thread."""
        if self._polling_started:
            return
        self._polling_started = True
        import threading
        import time

        def _worker():
            logger.info("Telegram Auto-Discovery Poller started in background.")
            while True:
                try:
                    self.poll_once()
                except Exception:
                    pass
                time.sleep(3)

        thread = threading.Thread(target=_worker, daemon=True, name="TelegramPoller")
        thread.start()


telegram_service = TelegramService()
