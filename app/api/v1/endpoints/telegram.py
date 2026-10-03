from fastapi import APIRouter, HTTPException, Query, Body
from typing import Optional, Dict, Any, List
from pydantic import BaseModel

from app.services.telegram_service import telegram_service
from app.services.strategy_engine import strategy_engine
from app.api.v1.endpoints.market import POPULAR_INDICES, POPULAR_COMMODITIES

router = APIRouter()


class CustomMessageRequest(BaseModel):
    message: str
    chat_id: Optional[str] = None
    bot_token: Optional[str] = None


class TelegramConfigRequest(BaseModel):
    bot_token: str
    chat_id: Optional[str] = "9100040008"


@router.get("/status", summary="Get Telegram Bot alerting status")
def get_telegram_status():
    """Check Telegram bot configuration, default destination (9100040008), and readiness."""
    return telegram_service.get_status()


@router.post("/test", summary="Send test verification ping to Telegram (9100040008)")
def send_telegram_test(
    chat_id: Optional[str] = Query(None, description="Optional target Telegram Chat ID (defaults to 9100040008)"),
    bot_token: Optional[str] = Query(None, description="Optional Telegram Bot Token override"),
):
    """Sends a verification message to Telegram to test bot delivery and chat access."""
    res = telegram_service.send_test_message(chat_id=chat_id, bot_token=bot_token)
    return res


@router.post("/send-signal/{symbol}", summary="Generate live CALL / PUT signal and send to Telegram")
def send_signal_to_telegram(
    symbol: str,
    chat_id: Optional[str] = Query(None, description="Target Telegram Chat ID (defaults to 9100040008 / registered ID)"),
    bot_token: Optional[str] = Query(None, description="Optional Bot Token override"),
    min_points: float = Query(5.0, description="Minimum expected move in points (default: 5.0 pts)"),
    market_hours_only: bool = Query(False, description="Enforce market hours only (set False for manual direct trigger)"),
    bypass_filters: bool = Query(True, description="Bypass filters for manual one-click dispatch"),
    interval: str = Query("15m", description="Candle interval (5m, 15m, 1h, 1d)"),
    period: str = Query("5d", description="Lookback period (5d, 1mo)"),
):
    """
    Computes 12-factor confluence CALL / PUT signal with exact option strike prices,
    targets, and stop-loss levels, then dispatches the formatted alert to Telegram.
    """
    sig = strategy_engine.generate_options_call_put_signal(symbol, interval=interval, period=period)
    res = telegram_service.send_signal_alert(
        sig,
        chat_id=chat_id,
        bot_token=bot_token,
        min_points=min_points,
        market_hours_only=market_hours_only,
        bypass_filters=bypass_filters,
    )
    return {
        "delivery_result": res,
        "signal_sent": {
            "instrument": sig.get("instrument"),
            "symbol": sig.get("symbol"),
            "recommendation": sig.get("recommendation"),
            "suggested_strike": sig.get("suggested_strike"),
            "option_entry_price": sig.get("option_entry_price"),
            "option_target_1": sig.get("option_target_1"),
            "option_stop_loss": sig.get("option_stop_loss"),
            "expected_option_gain_pts": sig.get("expected_option_gain_pts"),
            "is_good_move": sig.get("is_good_move"),
            "confidence_score": sig.get("confidence_score"),
        },
    }


@router.post("/broadcast-all", summary="Broadcast all active BUY CALL / BUY PUT signals to Telegram")
def broadcast_all_signals(
    chat_id: Optional[str] = Query(None, description="Target Telegram Chat ID (defaults to 9100040008 / registered ID)"),
    bot_token: Optional[str] = Query(None, description="Optional Bot Token override"),
    include_commodities: bool = Query(True, description="Include MCX Commodities"),
    min_points: float = Query(5.0, description="Minimum expected move in points (default: 5.0 pts)"),
    market_hours_only: bool = Query(True, description="Deliver alerts during active market hours only"),
    bypass_filters: bool = Query(False, description="Bypass filters for testing/manual force broadcast"),
    min_confidence: float = Query(60.0, description="Minimum confluence confidence score to alert"),
):
    """
    Scans major Indices (NIFTY 50, BANK NIFTY, SENSEX) and MCX Commodities,
    and sends high-confidence CALL / PUT recommendations to Telegram (Market Hours & Min 5 Pts).
    """
    targets = ["^NSEI", "^NSEBANK", "^BSESN"]
    if include_commodities:
        targets.extend(["CRUDEOIL", "GOLD", "SILVER", "NATURALGAS", "COPPER"])

    dispatched = []
    skipped = []

    for sym in targets:
        try:
            sig = strategy_engine.generate_options_call_put_signal(sym)
            res = telegram_service.send_signal_alert(
                sig,
                chat_id=chat_id,
                bot_token=bot_token,
                min_points=min_points,
                market_hours_only=market_hours_only,
                bypass_filters=bypass_filters,
            )
            if res.get("success"):
                dispatched.append({
                    "symbol": sym,
                    "instrument": sig.get("instrument"),
                    "recommendation": sig.get("recommendation"),
                    "strike": sig.get("suggested_strike"),
                    "expected_move_pts": sig.get("expected_option_gain_pts"),
                    "delivery": res,
                })
            else:
                skipped.append({
                    "symbol": sym,
                    "status": res.get("status"),
                    "reason": res.get("reason", "Filtered out"),
                })
        except Exception as e:
            skipped.append({"symbol": sym, "reason": str(e)})

    return {
        "status": "COMPLETED",
        "market_hours_only": market_hours_only,
        "min_points_filter": min_points,
        "total_scanned": len(targets),
        "total_dispatched": len(dispatched),
        "total_skipped": len(skipped),
        "dispatched": dispatched,
        "skipped": skipped,
    }


@router.post("/custom-message", summary="Send custom message or trading note to Telegram")
def send_custom_telegram_message(payload: CustomMessageRequest):
    """Sends arbitrary trading notice or update to Telegram."""
    return telegram_service.send_message(
        text=payload.message,
        chat_id=payload.chat_id,
        bot_token=payload.bot_token,
        parse_mode="HTML",
    )


@router.post("/config", summary="Dynamically set Telegram Bot Token and Chat ID")
def update_telegram_config(payload: TelegramConfigRequest):
    """Updates Telegram bot token and default destination chat ID in runtime settings."""
    if payload.bot_token:
        telegram_service.bot_token = payload.bot_token.strip()
    if payload.chat_id:
        telegram_service.default_chat_id = payload.chat_id.strip()
    return {
        "status": "success",
        "message": f"Telegram configured. Destination Chat ID: {telegram_service.default_chat_id}",
        "details": telegram_service.get_status(),
    }
