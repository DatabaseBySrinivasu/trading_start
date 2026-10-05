from fastapi import APIRouter, HTTPException, Query, Body
from typing import Optional, Dict, Any, List
from pydantic import BaseModel

from app.services.instagram_service import instagram_service
from app.services.strategy_engine import strategy_engine

router = APIRouter()


class CustomMessageRequest(BaseModel):
    message: str
    recipient_id: Optional[str] = "9100040008"
    access_token: Optional[str] = None


class InstagramConfigRequest(BaseModel):
    access_token: Optional[str] = None
    account_id: Optional[str] = None
    recipient_id: Optional[str] = "9100040008"
    webhook_url: Optional[str] = None


@router.get("/status", summary="Get Instagram Alerting status")
def get_instagram_status():
    """Check Instagram / Meta alert configuration, target recipient (9100040008), and readiness."""
    return instagram_service.get_status()


@router.post("/test", summary="Send test verification ping to Instagram (9100040008)")
def send_instagram_test(
    recipient_id: Optional[str] = Query("9100040008", description="Target Instagram Recipient ID / Phone (default: 9100040008)"),
    access_token: Optional[str] = Query(None, description="Optional Meta Instagram Graph API Access Token override"),
):
    """Sends a verification trade alert to Instagram for recipient 9100040008."""
    return instagram_service.send_test_message(recipient_id=recipient_id, access_token=access_token)


@router.post("/send-signal/{symbol}", summary="Generate live CALL / PUT signal and send to Instagram")
def send_signal_to_instagram(
    symbol: str,
    recipient_id: Optional[str] = Query("9100040008", description="Target Instagram Recipient ID / Phone (defaults to 9100040008)"),
    access_token: Optional[str] = Query(None, description="Optional Access Token override"),
    min_points: float = Query(5.0, description="Minimum expected move in points (default: 5.0 pts)"),
    market_hours_only: bool = Query(False, description="Enforce market hours only (set False for manual direct trigger)"),
    bypass_filters: bool = Query(True, description="Bypass filters for manual one-click dispatch"),
    interval: str = Query("15m", description="Candle interval (5m, 15m, 1h, 1d)"),
    period: str = Query("5d", description="Lookback period (5d, 1mo)"),
):
    """
    Computes 12-factor confluence CALL / PUT signal with exact option strike prices,
    targets, and stop-loss levels, then dispatches the formatted alert to Instagram.
    """
    sig = strategy_engine.generate_options_call_put_signal(symbol, interval=interval, period=period)
    res = instagram_service.send_signal_alert(
        sig,
        recipient_id=recipient_id,
        access_token=access_token,
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


@router.post("/broadcast-all", summary="Broadcast all active BUY CALL / BUY PUT signals to Instagram")
def broadcast_all_signals_instagram(
    recipient_id: Optional[str] = Query("9100040008", description="Target Instagram Recipient ID / Phone (defaults to 9100040008)"),
    access_token: Optional[str] = Query(None, description="Optional Access Token override"),
    include_commodities: bool = Query(True, description="Include MCX Commodities"),
    min_points: float = Query(5.0, description="Minimum expected move in points (default: 5.0 pts)"),
    market_hours_only: bool = Query(True, description="Deliver alerts during active market hours only"),
    bypass_filters: bool = Query(False, description="Bypass filters for testing/manual force broadcast"),
    min_confidence: float = Query(60.0, description="Minimum confluence confidence score to alert"),
):
    """
    Scans major Indices (NIFTY 50, BANK NIFTY, SENSEX) and MCX Commodities,
    and sends high-confidence CALL / PUT recommendations to Instagram recipient (9100040008).
    """
    targets = ["^NSEI", "^NSEBANK", "^BSESN"]
    if include_commodities:
        targets.extend(["CRUDEOIL", "GOLD", "SILVER", "NATURALGAS", "COPPER"])

    dispatched = []
    skipped = []

    for sym in targets:
        try:
            sig = strategy_engine.generate_options_call_put_signal(sym)
            res = instagram_service.send_signal_alert(
                sig,
                recipient_id=recipient_id,
                access_token=access_token,
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
            skipped.append({"symbol": sym, "error": str(e)})

    return {
        "status": "COMPLETED",
        "recipient": recipient_id,
        "market_hours_only": market_hours_only,
        "min_points_filter": min_points,
        "total_scanned": len(targets),
        "total_dispatched": len(dispatched),
        "total_skipped": len(skipped),
        "dispatched_count": len(dispatched),
        "skipped_count": len(skipped),
        "dispatched": dispatched,
        "skipped": skipped,
    }


@router.post("/send-custom", summary="Send custom trading message to Instagram")
def send_custom_instagram_message(body: CustomMessageRequest):
    """Sends custom message text to Instagram."""
    if not body.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    return instagram_service.send_message(
        text=body.message,
        recipient_id=body.recipient_id,
        access_token=body.access_token,
    )
