import os
import json
import time
import logging
import threading
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional

from app.core.config import settings

logger = logging.getLogger(__name__)


class TradeAuditService:
    """
    Local Trade Alert Logger & Autonomous Self-Correction Engine.

    Responsibilities:
    1. Local Persistent Logging: Logs every generated, dispatched, and filtered trade alert to local disk (`logs/trade_alerts.jsonl` and `logs/trade_audit.log`).
    2. Active Trade Tracking: Continuously tracks live price action (spot and option premium) against active trade alerts.
    3. Autonomous Mistake & Invalidation Detection:
       - Stop-loss breaches / adverse trend shifts.
       - False breakouts / momentum reversal traps (Bearish/Bullish BOS, reversal patterns).
       - Profit locking & trailing stop loss activations upon Target 1 (+5 pts) attainment.
       - Target 2 full-profit execution.
    4. Autonomous Self-Correction Dispatch: Dispatches prominent correction/exit alerts to Telegram and Instagram for recipient 9100040008.
    5. Self-Learning Recalibration: Dynamically adjusts required confluence thresholds if consecutive errors or market chop are detected.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.root_dir = Path(__file__).resolve().parent.parent.parent
        self.logs_dir = self.root_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.alerts_file = self.logs_dir / "trade_alerts.jsonl"
        self.corrections_file = self.logs_dir / "trade_corrections.jsonl"
        self.audit_log_file = self.logs_dir / "trade_audit.log"
        self.technical_factors_file = self.logs_dir / "trade_technical_factors.log"
        self.stats_file = self.logs_dir / "audit_stats.json"

        self.active_alerts: Dict[str, Dict[str, Any]] = {}
        self.completed_alerts: List[Dict[str, Any]] = []
        self.corrections_history: List[Dict[str, Any]] = []

        # Self-learning symbol penalty tracking (symbol -> consecutive loss count)
        self.symbol_loss_streak: Dict[str, int] = {}
        self.symbol_confidence_boost: Dict[str, float] = {}

        # Load historical state from disk
        self._load_state_from_disk()

    @staticmethod
    def get_ist_datetime() -> datetime:
        """Returns current datetime in Indian Standard Time (IST, UTC+5:30)."""
        ist = timezone(timedelta(hours=5, minutes=30))
        return datetime.now(ist)

    def _get_ist_time_str(self) -> str:
        return self.get_ist_datetime().strftime("%d %b %Y, %I:%M:%S %p IST")

    def _load_state_from_disk(self):
        """Loads past alerts and corrections from local jsonl files into memory."""
        try:
            if self.alerts_file.exists():
                with open(self.alerts_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                record = json.loads(line)
                                alert_id = record.get("alert_id")
                                if alert_id:
                                    if record.get("status") == "ACTIVE":
                                        self.active_alerts[alert_id] = record
                                    else:
                                        self.completed_alerts.append(record)
                            except Exception:
                                continue

            if self.corrections_file.exists():
                with open(self.corrections_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                self.corrections_history.append(json.loads(line))
                            except Exception:
                                continue

            self._log_text(f"Audit engine initialized. Active alerts: {len(self.active_alerts)}, History: {len(self.completed_alerts)}")
        except Exception as e:
            logger.error(f"Failed to load audit state: {e}")

    def _log_text(self, message: str):
        """Appends formatted human-readable log entry to logs/trade_audit.log."""
        try:
            ist_str = self._get_ist_time_str()
            log_line = f"[{ist_str}] {message}\n"
            with open(self.audit_log_file, "a", encoding="utf-8") as f:
                f.write(log_line)
        except Exception as e:
            logger.debug(f"Audit text log write error: {e}")

    def _append_jsonl(self, file_path: Path, data: Dict[str, Any]):
        """Appends structured JSON record to a jsonl file."""
        try:
            with open(file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Error appending to {file_path.name}: {e}")

    def _log_technical_factors(self, alert_id: str, signal: Dict[str, Any], record: Dict[str, Any]):
        """
        Logs complete technical factor diagnostic snapshot for the trade alert to logs/trade_technical_factors.log.
        Maintains complete audit trail of 21-factor institutional matrix, Greeks, indicators, and market context.
        """
        try:
            ist_str = self._get_ist_time_str()
            symbol = str(signal.get("symbol", record.get("symbol", "")))
            instrument = str(signal.get("instrument", record.get("instrument", symbol)))
            rec = str(signal.get("recommendation", record.get("recommendation", "")))
            strike = str(signal.get("suggested_strike", record.get("suggested_strike", "")))
            conf = float(signal.get("confidence_score", record.get("confidence_score", 0.0)))
            spot = float(signal.get("spot_price", record.get("spot_entry", 0.0)))

            opt_entry = float(signal.get("option_entry_price", record.get("option_entry_price", 0.0)))
            opt_t1 = float(signal.get("option_target_1", record.get("option_target_1", 0.0)))
            opt_t2 = float(signal.get("option_target_2", record.get("option_target_2", 0.0)))
            opt_t3 = float(signal.get("option_target_3", 0.0))
            opt_long_tgt = float(signal.get("option_long_target", 0.0))
            opt_sl = float(signal.get("option_stop_loss", record.get("option_stop_loss", 0.0)))
            lot_size = signal.get("lot_size", record.get("lot_size", 1))
            rr = signal.get("risk_reward_ratio", "1:2.0")

            lines = [
                "=" * 80,
                f"TRADE TECHNICAL FACTORS AUDIT REPORT — [{alert_id}]",
                f"Timestamp (IST)   : {ist_str}",
                f"Instrument        : {instrument} ({symbol})",
                f"Action / Signal   : {rec} | Strike: {strike}",
                f"Confidence Score  : {conf}% | R:R Ratio: {rr} | Lot Size: {lot_size}",
                "-" * 80,
                "1. EXECUTION & REFERENCE PRICING:",
                f"   • Underlying Spot Price : ₹{spot:,.2f}",
                f"   • Spot Levels (T1/T2/SL): ₹{signal.get('spot_target_1', 0):,.2f} / ₹{signal.get('spot_target_2', 0):,.2f} / ₹{signal.get('spot_stop_loss', 0):,.2f}",
                f"   • Option Entry Price    : ₹{opt_entry:.2f}",
                f"   • Option Targets (1/2/3): ₹{opt_t1:.2f} / ₹{opt_t2:.2f} / ₹{opt_t3:.2f}",
                f"   • Option Long Target    : ₹{opt_long_tgt:.2f}",
                f"   • Option Stop Loss      : ₹{opt_sl:.2f}",
                "-" * 80,
            ]

            # 2. 21-Factor Institutional Matrix
            inst_matrix = signal.get("institutional_matrix")
            if isinstance(inst_matrix, dict) and "factors" in inst_matrix:
                c_pct = inst_matrix.get("confluence_percentage", 0.0)
                p_cnt = inst_matrix.get("passed_count", 0)
                w_cnt = inst_matrix.get("watch_count", 0)
                f_cnt = inst_matrix.get("fail_count", 0)
                lines.append(f"2. 21-FACTOR INSTITUTIONAL MATRIX (Confluence: {c_pct:.1f}% | Pass: {p_cnt} | Watch: {w_cnt} | Fail: {f_cnt}):")
                for f in inst_matrix.get("factors", []):
                    f_num = f.get("id", "")
                    f_name = f.get("name", "")
                    f_status = f.get("status", "")
                    f_val = f.get("value", "")
                    f_desc = f.get("description", "")
                    lines.append(f"   [{f_status:5}] Factor {str(f_num):2}: {f_name:28} | Val: {str(f_val):22} | {f_desc}")
                lines.append("-" * 80)

            # 3. Technical Indicators
            ind = signal.get("indicators", {})
            if ind:
                lines.append("3. CORE TECHNICAL INDICATORS:")
                lines.append(f"   • RSI (14)          : {ind.get('rsi_14', 'N/A')}")
                lines.append(f"   • EMAs (9/21/50)    : {ind.get('ema_9', 'N/A')} / {ind.get('ema_21', 'N/A')} / {ind.get('ema_50', 'N/A')}")
                lines.append(f"   • Supertrend        : {ind.get('supertrend', 'N/A')} ({ind.get('supertrend_signal', 'N/A')})")
                lines.append(f"   • MACD / Signal     : {ind.get('macd', 'N/A')} / {ind.get('macd_signal', 'N/A')}")
                lines.append(f"   • Bollinger Bands   : Upper: {ind.get('bollinger_upper', 'N/A')} | Mid: {ind.get('bollinger_middle', 'N/A')} | Lower: {ind.get('bollinger_lower', 'N/A')}")
                lines.append(f"   • ATR (14)          : {ind.get('atr', 'N/A')}")
                lines.append("-" * 80)

            # 4. Derivatives & Options Order Flow
            pcr = signal.get("pcr", {})
            if pcr:
                lines.append("4. DERIVATIVES & OPTIONS FLOW (PCR & MAX PAIN):")
                lines.append(f"   • PCR (OI Ratio)    : {pcr.get('pcr_oi', 'N/A')} ({pcr.get('sentiment', 'N/A')})")
                lines.append(f"   • Max Pain Strike   : ₹{pcr.get('max_pain_strike', 'N/A')}")
                p_oi = pcr.get('total_put_oi', 'N/A')
                c_oi = pcr.get('total_call_oi', 'N/A')
                if isinstance(p_oi, (int, float)) and isinstance(c_oi, (int, float)):
                    lines.append(f"   • Total Put/Call OI : {p_oi:,.0f} / {c_oi:,.0f}")
                else:
                    lines.append(f"   • Total Put/Call OI : {p_oi} / {c_oi}")
                lines.append("-" * 80)

            # 5. Price Action & Smart Money Concepts (SMC)
            pam = signal.get("price_action_momentum", {})
            dl = signal.get("day_levels", {})
            ob = signal.get("order_blocks", {})
            lines.append("5. PRICE ACTION & SMART MONEY CONCEPTS (SMC):")
            if pam:
                lines.append(f"   • Momentum Score    : {pam.get('momentum_score', 'N/A')} ({pam.get('momentum_regime', 'N/A')})")
                lines.append(f"   • Trend Structure   : {pam.get('trend_structure', 'N/A')}")
                lines.append(f"   • BOS Status        : {pam.get('bos_status', 'N/A')}")
                lines.append(f"   • Wick Rejection    : {pam.get('wick_rejection', 'N/A')}")
            if dl:
                cpr = dl.get("cpr", {})
                lines.append(f"   • PDH / PDL / PDC   : ₹{dl.get('pdh', 'N/A')} / ₹{dl.get('pdl', 'N/A')} / ₹{dl.get('pdc', 'N/A')}")
                lines.append(f"   • CPR (TC/Pivot/BC) : ₹{cpr.get('tc', 'N/A')} / ₹{cpr.get('pivot', 'N/A')} / ₹{cpr.get('bc', 'N/A')}")
            if ob:
                lines.append(f"   • SMC Order Blocks  : Bias: {ob.get('smc_bias', 'N/A')} | Bullish OB: {ob.get('nearest_bullish_ob', 'N/A')} | Bearish OB: {ob.get('nearest_bearish_ob', 'N/A')}")
            lines.append("-" * 80)

            # 6. Macro Cues & Volatility
            vix = signal.get("vix_intel", {})
            gift = signal.get("gift_nifty_intel", {})
            news = signal.get("news_intel", {})
            lines.append("6. MACRO SENTIMENT & VOLATILITY CONTEXT:")
            if vix:
                lines.append(f"   • India VIX         : {vix.get('current_vix', 'N/A')} ({vix.get('regime', 'N/A')}, Bias: {vix.get('trend_bias', 'N/A')})")
            if gift:
                lines.append(f"   • GIFT NIFTY Cues   : Price: {gift.get('gift_nifty_price', 'N/A')} | Gap: {gift.get('projected_gap_pts', 'N/A')} pts ({gift.get('opening_bias', 'N/A')})")
            if news:
                lines.append(f"   • Financial News    : Sentiment: {news.get('overall_sentiment', 'N/A')} (Score: {news.get('average_sentiment_score', 'N/A')})")
            lines.append("=" * 80 + "\n")

            log_block = "\n".join(lines)
            with open(self.technical_factors_file, "a", encoding="utf-8") as f:
                f.write(log_block)
        except Exception as e:
            logger.error(f"Error logging technical factors to {self.technical_factors_file.name}: {e}")

    def _update_alert_in_jsonl(self, updated_record: Dict[str, Any]):
        """Rewrites active alerts file when record status is modified."""
        try:
            # We rewrite the alerts file periodically or on state change
            all_records = list(self.active_alerts.values()) + self.completed_alerts[-300:]
            with open(self.alerts_file, "w", encoding="utf-8") as f:
                for r in all_records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Error updating alerts file: {e}")

    def log_trade_alert(
        self,
        signal: Dict[str, Any],
        channels: Optional[List[str]] = None,
        delivery_status: str = "DISPATCHED",
        dispatch_response: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Logs a generated or dispatched trade alert locally on disk and in memory.
        Enqueues the alert for live tracking and mistake self-correction.
        """
        with self.lock:
            symbol = str(signal.get("symbol", "^NSEI")).strip().upper()
            instrument = str(signal.get("instrument", symbol))
            rec = str(signal.get("recommendation", "NEUTRAL")).strip()
            strike = str(signal.get("suggested_strike", "N/A"))

            # Only track directional actionable alerts (BUY CALL / BUY PUT)
            is_actionable = "CALL" in rec or "PUT" in rec
            now_ts = time.time()
            now_ist = self._get_ist_time_str()
            alert_id = f"ALT_{symbol.replace('^', '').replace('=', '')}_{int(now_ts)}"

            spot_price = float(signal.get("spot_price", 0.0))
            spot_levels = signal.get("spot_levels", {})
            spot_t1 = float(spot_levels.get("spot_target_1") or signal.get("spot_target_1") or spot_price)
            spot_t2 = float(spot_levels.get("spot_target_2") or signal.get("spot_target_2") or spot_price)
            spot_sl = float(spot_levels.get("spot_stop_loss") or signal.get("spot_stop_loss") or spot_price)

            opt_entry = float(signal.get("option_entry_price", signal.get("entry_price", 0.0)))
            opt_t1 = float(signal.get("option_target_1", signal.get("target_1", 0.0)))
            opt_t2 = float(signal.get("option_target_2", signal.get("target_2", 0.0)))
            opt_sl = float(signal.get("option_stop_loss", signal.get("stop_loss", 0.0)))

            opt_gain = round(max(0.0, opt_t1 - opt_entry), 2)
            spot_gain = round(abs(spot_t1 - spot_price), 2)
            conf = float(signal.get("confidence_score", 0.0))

            # Check if there is an existing ACTIVE alert for the same symbol
            # If so, close/supersede previous one
            for existing_id, existing_alert in list(self.active_alerts.items()):
                if existing_alert.get("symbol") == symbol and existing_alert.get("status") == "ACTIVE":
                    existing_alert["status"] = "SUPERSEDED"
                    existing_alert["closed_at_ist"] = now_ist
                    existing_alert["closed_reason"] = f"Superseded by new signal {alert_id} ({rec})"
                    self.completed_alerts.append(existing_alert)
                    del self.active_alerts[existing_id]

            record = {
                "alert_id": alert_id,
                "timestamp": now_ts,
                "created_at_ist": now_ist,
                "last_checked_at_ist": now_ist,
                "symbol": symbol,
                "instrument": instrument,
                "recommendation": rec,
                "signal_type": signal.get("signal_type", "CONSOLIDATION"),
                "suggested_strike": strike,
                "spot_entry": spot_price,
                "spot_target_1": spot_t1,
                "spot_target_2": spot_t2,
                "spot_stop_loss": spot_sl,
                "option_entry_price": opt_entry,
                "option_target_1": opt_t1,
                "option_target_2": opt_t2,
                "option_stop_loss": opt_sl,
                "expected_option_gain_pts": opt_gain,
                "expected_spot_gain_pts": spot_gain,
                "lot_size": signal.get("lot_size", 1),
                "confidence_score": conf,
                "confluence_reasons": signal.get("confluence_reasons", []),
                "channels": channels or ["TELEGRAM", "INSTAGRAM"],
                "delivery_status": delivery_status,
                "dispatch_response": dispatch_response,
                "status": "ACTIVE" if is_actionable else "FILTERED",
                "highest_spot_reached": spot_price,
                "lowest_spot_reached": spot_price,
                "highest_opt_reached": opt_entry,
                "lowest_opt_reached": opt_entry,
                "current_spot": spot_price,
                "current_opt_estimated": opt_entry,
                "target_1_hit": False,
                "target_2_hit": False,
                "stop_loss_hit": False,
                "corrections_dispatched": [],
            }

            if is_actionable:
                self.active_alerts[alert_id] = record
            else:
                self.completed_alerts.append(record)

            # Persist to JSONL & text logs
            self._append_jsonl(self.alerts_file, record)
            self._log_text(
                f"TRADE ALERT LOGGED: [{alert_id}] {instrument} ({symbol}) -> {rec} [{strike}] "
                f"Entry: ₹{opt_entry:.2f} | T1: ₹{opt_t1:.2f} (+{opt_gain} pts) | SL: ₹{opt_sl:.2f} | Conf: {conf}%"
            )
            self._log_technical_factors(alert_id, signal, record)

            return record

    def check_and_self_correct(
        self,
        symbol: Optional[str] = None,
        force_dispatch: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Audits all active trade alerts against live price action and momentum.
        Detects mistakes, stop loss breaches, adverse reversals, or target milestones,
        and automatically dispatches self-correction notices to Telegram and Instagram.
        """
        from app.services.market_data import market_data_service
        from app.services.telegram_service import telegram_service
        from app.services.instagram_service import instagram_service

        corrections_generated: List[Dict[str, Any]] = []
        now_ts = time.time()
        now_ist = self._get_ist_time_str()

        with self.lock:
            alerts_to_check = list(self.active_alerts.values())

        for alert in alerts_to_check:
            sym = alert.get("symbol")
            if symbol and sym != symbol:
                continue

            alert_id = alert.get("alert_id")
            rec = alert.get("recommendation", "")
            is_call = "CALL" in rec
            is_put = "PUT" in rec
            strike = alert.get("suggested_strike", "N/A")
            instrument = alert.get("instrument", sym)

            # 1. Fetch live market price
            live_data = market_data_service.get_live_price(sym)
            if not live_data or "price" not in live_data:
                continue

            cur_spot = float(live_data["price"])
            spot_entry = float(alert.get("spot_entry", cur_spot))
            spot_t1 = float(alert.get("spot_target_1", cur_spot))
            spot_t2 = float(alert.get("spot_target_2", cur_spot))
            spot_sl = float(alert.get("spot_stop_loss", cur_spot))

            opt_entry = float(alert.get("option_entry_price", 0.0))
            opt_t1 = float(alert.get("option_target_1", 0.0))
            opt_t2 = float(alert.get("option_target_2", 0.0))
            opt_sl = float(alert.get("option_stop_loss", 0.0))

            # Approximate current option premium move
            spot_move = cur_spot - spot_entry
            delta_est = 0.50 if is_call or is_put else 0.0
            cur_opt_est = round(max(0.50, opt_entry + (spot_move * delta_est if is_call else -spot_move * delta_est)), 2)

            # Track peak / trough excursion
            alert["highest_spot_reached"] = max(alert.get("highest_spot_reached", cur_spot), cur_spot)
            alert["lowest_spot_reached"] = min(alert.get("lowest_spot_reached", cur_spot), cur_spot)
            alert["highest_opt_reached"] = max(alert.get("highest_opt_reached", cur_opt_est), cur_opt_est)
            alert["lowest_opt_reached"] = min(alert.get("lowest_opt_reached", cur_opt_est), cur_opt_est)
            alert["current_spot"] = cur_spot
            alert["current_opt_estimated"] = cur_opt_est
            alert["last_checked_at_ist"] = now_ist

            correction_type: Optional[str] = None
            correction_reason: str = ""
            action_directive: str = ""
            mark_closed = False
            is_success = False

            # 2. Check Conditions

            # A. TARGET 2 HIT (Max Profit Achieved)
            if not alert.get("target_2_hit"):
                if (is_call and (cur_spot >= spot_t2 or cur_opt_est >= opt_t2)) or \
                   (is_put and (cur_spot <= spot_t2 or cur_opt_est >= opt_t2)):
                    correction_type = "TARGET_2_ACHIEVED"
                    correction_reason = f"Target 2 reached! Current Spot: ₹{cur_spot:,.2f} | Est Option Premium: ₹{cur_opt_est:,.2f}."
                    action_directive = "🏆 BOOK FULL PROFIT NOW. Target 2 objective completely fulfilled."
                    alert["target_2_hit"] = True
                    alert["status"] = "TARGET_2_HIT"
                    mark_closed = True
                    is_success = True

            # B. TARGET 1 HIT (+5 Pts Move Achieved) -> Issue Profit Booking & Trail SL Correction
            if not correction_type and not alert.get("target_1_hit"):
                if (is_call and (cur_spot >= spot_t1 or cur_opt_est >= opt_t1)) or \
                   (is_put and (cur_spot <= spot_t1 or cur_opt_est >= opt_t1)):
                    pts_gained = round(max(0.0, cur_opt_est - opt_entry), 2)
                    correction_type = "TARGET_1_HIT_TRAIL_SL"
                    correction_reason = f"Target 1 reached (+{pts_gained} pts)! Current Spot: ₹{cur_spot:,.2f} (Entry: ₹{spot_entry:,.2f})."
                    action_directive = f"🎯 BOOK 70% PROFIT & TRAIL STOP LOSS TO COST (₹{opt_entry:,.2f}) to lock in gains risk-free."
                    alert["target_1_hit"] = True
                    alert["status"] = "TARGET_1_HIT"
                    # Tighten stop loss to breakeven entry
                    alert["spot_stop_loss"] = spot_entry
                    alert["option_stop_loss"] = opt_entry
                    is_success = True

            # C. STOP LOSS BREACH (Adverse Market Invalidation)
            if not correction_type:
                sl_breached = False
                if is_call and (cur_spot <= spot_sl or cur_opt_est <= opt_sl):
                    sl_breached = True
                elif is_put and (cur_spot >= spot_sl or cur_opt_est <= opt_sl):
                    sl_breached = True

                if sl_breached:
                    pts_loss = round(abs(opt_entry - cur_opt_est), 2)
                    correction_type = "STOP_LOSS_EXIT"
                    correction_reason = f"Adverse move breached Stop Loss level (₹{spot_sl:,.2f}). Current Spot: ₹{cur_spot:,.2f}."
                    action_directive = f"🛑 EXIT POSITION IMMEDIATELY to protect capital and prevent drawdown."
                    alert["stop_loss_hit"] = True
                    alert["status"] = "STOP_LOSS_HIT"
                    mark_closed = True
                    is_success = False

            # D. SHARP TREND REVERSAL / BREAKOUT TRAP (SMC / Momentum Invalidation)
            if not correction_type and not mark_closed:
                # Fetch fresh pattern & momentum confluence
                try:
                    df = market_data_service.get_historical_candles(sym, period="5d", interval="15m")
                    if df is not None and not df.empty:
                        from app.services.pattern_service import pattern_service
                        pam = pattern_service.analyze_price_action_momentum(df)
                        mom_score = float(pam.get("momentum_score", 0.0))
                        bos = str(pam.get("bos_status", ""))

                        # Call invalidated by sharp bearish breakout
                        if is_call and (mom_score <= -30.0 or "BEARISH_BOS" in bos):
                            correction_type = "TREND_REVERSAL_EXIT"
                            correction_reason = f"Bearish trend reversal confirmed: {bos} with strong selling momentum ({mom_score})."
                            action_directive = "⚠️ EXIT CALL POSITION IMMEDIATELY. Bullish structure invalidated."
                            alert["status"] = "REVERSED_INVALIDATED"
                            mark_closed = True
                            is_success = False

                        # Put invalidated by sharp bullish breakout
                        elif is_put and (mom_score >= 30.0 or "BULLISH_BOS" in bos):
                            correction_type = "TREND_REVERSAL_EXIT"
                            correction_reason = f"Bullish trend reversal confirmed: {bos} with strong buying momentum (+{mom_score})."
                            action_directive = "⚠️ EXIT PUT POSITION IMMEDIATELY. Bearish structure invalidated."
                            alert["status"] = "REVERSED_INVALIDATED"
                            mark_closed = True
                            is_success = False
                except Exception as ex:
                    logger.debug(f"Reversal check skipped for {sym}: {ex}")

            # E. TIME-BASED STALENESS (3+ Hours of compression without progress)
            if not correction_type and not mark_closed:
                age_hours = (now_ts - alert.get("timestamp", now_ts)) / 3600.0
                if age_hours >= 3.0 and not alert.get("target_1_hit"):
                    correction_type = "TIME_DECAY_EXPIRY"
                    correction_reason = f"Signal held for {age_hours:.1f} hours in tight range without hitting T1. Theta decay risk elevated."
                    action_directive = "⏸️ CLOSE POSITION AT MARKET / SQUARE OFF to avoid option premium decay."
                    alert["status"] = "EXPIRED_STALE"
                    mark_closed = True

            # 3. If a Correction Event Occurred, Record & Dispatch
            if correction_type:
                corr_id = f"CORR_{sym.replace('^', '')}_{int(now_ts)}"
                corr_event = {
                    "correction_id": corr_id,
                    "alert_id": alert_id,
                    "symbol": sym,
                    "instrument": instrument,
                    "original_recommendation": rec,
                    "suggested_strike": strike,
                    "type": correction_type,
                    "timestamp": now_ts,
                    "ist_time": now_ist,
                    "current_spot": cur_spot,
                    "spot_entry": spot_entry,
                    "current_opt_estimated": cur_opt_est,
                    "option_entry": opt_entry,
                    "reason": correction_reason,
                    "action_directive": action_directive,
                    "is_success": is_success,
                }

                alert["corrections_dispatched"].append(corr_event)
                self.corrections_history.append(corr_event)
                self._append_jsonl(self.corrections_file, corr_event)
                corrections_generated.append(corr_event)

                # Update self-learning feedback for this symbol
                self._update_self_learning_feedback(sym, is_success)

                # Log human-readable audit entry
                self._log_text(
                    f"SELF-CORRECTION TRIGGERED: [{corr_id}] for {instrument} ({rec}) -> {correction_type}: {correction_reason}"
                )

                # Dispatch Correction Alerts to Telegram and Instagram
                if force_dispatch:
                    self._dispatch_correction_to_channels(
                        corr_event=corr_event,
                        telegram_service=telegram_service,
                        instagram_service=instagram_service,
                    )

            # 4. If alert is closed, move from active to completed
            if mark_closed:
                alert["closed_at_ist"] = now_ist
                with self.lock:
                    if alert_id in self.active_alerts:
                        self.completed_alerts.append(alert)
                        del self.active_alerts[alert_id]

            # Update alert in memory and file
            self._update_alert_in_jsonl(alert)

        return corrections_generated

    def _update_self_learning_feedback(self, symbol: str, is_success: bool):
        """
        Dynamically adjusts symbol confidence requirements based on outcome history.
        If consecutive losses occur on choppy markets, boosts required confluence score.
        """
        current_streak = self.symbol_loss_streak.get(symbol, 0)
        if is_success:
            self.symbol_loss_streak[symbol] = 0
            self.symbol_confidence_boost[symbol] = 0.0
            self._log_text(f"LEARNING UPDATE: {symbol} win recorded. Confluence threshold reset to baseline.")
        else:
            new_streak = current_streak + 1
            self.symbol_loss_streak[symbol] = new_streak
            if new_streak >= 2:
                # Add +15% penalty to required confluence per consecutive error to avoid chop
                boost = min(30.0, (new_streak - 1) * 15.0)
                self.symbol_confidence_boost[symbol] = boost
                self._log_text(
                    f"LEARNING RECALIBRATION: {symbol} loss streak={new_streak}. "
                    f"Requiring +{boost}% higher confluence (min threshold: {settings.ALERT_MIN_CONFIDENCE + boost}%) for next alerts."
                )

    def get_symbol_confidence_threshold(self, symbol: str) -> float:
        """Returns baseline + dynamic penalty threshold for a symbol."""
        boost = self.symbol_confidence_boost.get(symbol, 0.0)
        return min(95.0, settings.ALERT_MIN_CONFIDENCE + boost)

    def _dispatch_correction_to_channels(
        self,
        corr_event: Dict[str, Any],
        telegram_service: Any,
        instagram_service: Any,
    ):
        """Dispatches formatted correction messages to Telegram and Instagram."""
        instrument = corr_event["instrument"]
        sym = corr_event["symbol"]
        rec = corr_event["original_recommendation"]
        strike = corr_event["suggested_strike"]
        reason = corr_event["reason"]
        directive = corr_event["action_directive"]
        ist_time = corr_event["ist_time"]
        corr_type = corr_event["type"]

        badge_icon = "🎯 PROFIT TARGET REACHED" if corr_event["is_success"] else "⚠️ TRADE RE-CORRECTION / EXIT NOTICE"

        # 1. Telegram HTML Message
        tg_html = (
            f"⚡ <b>QUANTPULSE PRO: {badge_icon}</b> ⚡\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🚨 <b>ALERT ID:</b> <code>{corr_event['alert_id']}</code>\n"
            f"📊 <b>Instrument:</b> {instrument} ({sym})\n"
            f"🏷️ <b>Original Recommendation:</b> <b>{rec}</b> [{strike}]\n"
            f"📍 <b>Spot Entry:</b> ₹{corr_event['spot_entry']:,.2f} | <b>Current Spot:</b> ₹{corr_event['current_spot']:,.2f}\n"
            f"💰 <b>Est. Option Premium:</b> ₹{corr_event['current_opt_estimated']:,.2f} (Entry: ₹{corr_event['option_entry']:,.2f})\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🔍 <b>DIAGNOSIS:</b>\n"
            f"• <i>{reason}</i>\n\n"
            f"🎯 <b>ACTION DIRECTIVE:</b>\n"
            f"👉 <b>{directive}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🕒 <i>{ist_time}</i>\n"
            f"📱 Recipient: <code>9100040008</code> (Verified)"
        )

        try:
            telegram_service.send_message(text=tg_html, parse_mode="HTML")
        except Exception as e:
            logger.warning(f"Telegram correction dispatch failed: {e}")

        # 2. Instagram Message
        ig_text = (
            f"⚡ QUANTPULSE PRO: {badge_icon} ⚡\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"🚨 ALERT ID: {corr_event['alert_id']}\n"
            f"📊 Instrument: {instrument} ({sym})\n"
            f"🏷️ Original Signal: {rec} [{strike}]\n"
            f"📍 Spot: ₹{corr_event['current_spot']:,.2f} (Entry: ₹{corr_event['spot_entry']:,.2f})\n"
            f"💰 Est Premium: ₹{corr_event['current_opt_estimated']:,.2f} (Entry: ₹{corr_event['option_entry']:,.2f})\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"🔍 DIAGNOSIS:\n"
            f"• {reason}\n\n"
            f"🎯 ACTION DIRECTIVE:\n"
            f"👉 {directive}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"🕒 {ist_time}\n"
            f"📱 Target: 9100040008\n"
            f"#tradecorrection #quantpulse #capitalprotection #tradingalerts"
        )

        try:
            instagram_service.send_message(text=ig_text)
        except Exception as e:
            logger.warning(f"Instagram correction dispatch failed: {e}")

    def get_audit_summary(self) -> Dict[str, Any]:
        """Returns consolidated audit, win rate, mistake correction stats, and learning status."""
        with self.lock:
            all_alerts = list(self.active_alerts.values()) + self.completed_alerts
            total_logged = len(all_alerts)
            active_count = len(self.active_alerts)
            corrections_count = len(self.corrections_history)

            target_hits = sum(1 for a in all_alerts if a.get("target_1_hit") or a.get("status") in ["TARGET_1_HIT", "TARGET_2_HIT"])
            sl_hits = sum(1 for a in all_alerts if a.get("stop_loss_hit") or a.get("status") == "STOP_LOSS_HIT")
            reversals = sum(1 for a in all_alerts if a.get("status") == "REVERSED_INVALIDATED")

            completed_count = target_hits + sl_hits + reversals
            win_rate = round((target_hits / completed_count * 100), 1) if completed_count > 0 else 100.0

            return {
                "total_alerts_logged": total_logged,
                "active_alerts_count": active_count,
                "total_corrections_dispatched": corrections_count,
                "target_1_and_2_hits": target_hits,
                "stop_loss_exits": sl_hits,
                "trend_reversals_caught": reversals,
                "accuracy_win_rate_pct": win_rate,
                "active_alerts": list(self.active_alerts.values()),
                "recent_corrections": self.corrections_history[-20:],
                "recent_alerts": all_alerts[-25:],
                "learning_recalibrations": {
                    "symbol_loss_streaks": self.symbol_loss_streak,
                    "confidence_boosts": self.symbol_confidence_boost,
                },
                "log_files": {
                    "alerts_jsonl": str(self.alerts_file),
                    "corrections_jsonl": str(self.corrections_file),
                    "audit_text_log": str(self.audit_log_file),
                    "technical_factors_log": str(self.technical_factors_file),
                },
            }

    def get_recent_logs(self, limit: int = 50) -> List[str]:
        """Returns the most recent lines from the trade audit text log."""
        if not self.audit_log_file.exists():
            return []
        try:
            with open(self.audit_log_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                return [line.strip() for line in lines[-limit:]]
        except Exception as e:
            logger.error(f"Error reading audit log: {e}")
            return []

    def get_recent_technical_factor_logs(self, limit: int = 100) -> List[str]:
        """Returns the most recent lines from the trade technical factors text log."""
        if not self.technical_factors_file.exists():
            return []
        try:
            with open(self.technical_factors_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                return [line.strip() for line in lines[-limit:]]
        except Exception as e:
            logger.error(f"Error reading technical factors log: {e}")
            return []


# Global Singleton Instance
trade_audit_service = TradeAuditService()
