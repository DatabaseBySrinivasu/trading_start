from fastapi import APIRouter, HTTPException, Query, Body
from typing import Optional, Dict, Any, List
from pydantic import BaseModel

from app.services.trade_audit_service import trade_audit_service

router = APIRouter()


class AuditCheckRequest(BaseModel):
    symbol: Optional[str] = None
    force_dispatch: bool = True


@router.get("/summary", summary="Get comprehensive Trade Alert Audit & Self-Correction Summary")
def get_audit_summary():
    """
    Returns real-time trade alert auditing stats, active alert tracking status,
    historical mistake corrections, target hit rate, and self-learning recalibrations.
    """
    return trade_audit_service.get_audit_summary()


@router.get("/alerts", summary="Get active and historical trade alerts from local JSONL logs")
def get_trade_alerts(
    status: Optional[str] = Query(None, description="Filter by status (ACTIVE, TARGET_1_HIT, TARGET_2_HIT, STOP_LOSS_HIT, REVERSED_INVALIDATED)"),
    symbol: Optional[str] = Query(None, description="Filter by ticker symbol (e.g. ^NSEI, ^NSEBANK, CRUDEOIL)"),
    limit: int = Query(50, description="Max records to return", ge=1, le=200),
):
    """
    Retrieves logged trade alerts with target milestones, stop losses, and live excursion tracking.
    """
    summary = trade_audit_service.get_audit_summary()
    alerts = summary.get("recent_alerts", [])

    if symbol:
        alerts = [a for a in alerts if a.get("symbol", "").upper() == symbol.upper()]
    if status:
        alerts = [a for a in alerts if a.get("status", "").upper() == status.upper()]

    return {
        "total_returned": len(alerts[-limit:]),
        "active_count": summary.get("active_alerts_count", 0),
        "alerts": list(reversed(alerts[-limit:])),
    }


@router.get("/corrections", summary="Get all autonomous trade mistake corrections & exit notices")
def get_trade_corrections(
    limit: int = Query(30, description="Max corrections to return", ge=1, le=100)
):
    """
    Retrieves chronological history of autonomous mistake detections, stop loss breaches,
    false breakout exits, and profit-locking trailing stop loss activations.
    """
    summary = trade_audit_service.get_audit_summary()
    corrections = summary.get("recent_corrections", [])
    return {
        "total_corrections": len(corrections),
        "corrections": list(reversed(corrections[-limit:])),
    }


@router.get("/logs", summary="Get recent entries from persistent local trade audit text log")
def get_trade_audit_logs(
    limit: int = Query(50, description="Number of log lines to retrieve", ge=10, le=200)
):
    """
    Reads the tail of `logs/trade_audit.log` recorded locally on disk.
    """
    lines = trade_audit_service.get_recent_logs(limit=limit)
    return {
        "log_file": str(trade_audit_service.audit_log_file),
        "lines_count": len(lines),
        "logs": lines,
    }


@router.get("/technical-factors", summary="Get recent entries from persistent local technical factors diagnostic log")
def get_trade_technical_factors_logs(
    limit: int = Query(100, description="Number of log lines to retrieve", ge=10, le=500)
):
    """
    Reads the tail of `logs/trade_technical_factors.log` recorded locally on disk.
    """
    lines = trade_audit_service.get_recent_technical_factor_logs(limit=limit)
    return {
        "log_file": str(trade_audit_service.technical_factors_file),
        "lines_count": len(lines),
        "logs": lines,
    }


@router.post("/check-now", summary="Trigger immediate audit & autonomous mistake self-correction cycle")
def trigger_audit_check(
    symbol: Optional[str] = Query(None, description="Optional symbol to audit specifically"),
    force_dispatch: bool = Query(True, description="Whether to dispatch detected corrections to Telegram and Instagram"),
):
    """
    Scans all active trade alerts against current market prices and momentum.
    If an adverse move, stop loss hit, or reversal is detected, immediately dispatches
    self-correction notices to Telegram and Instagram for recipient 9100040008.
    """
    corrections = trade_audit_service.check_and_self_correct(
        symbol=symbol,
        force_dispatch=force_dispatch,
    )
    return {
        "status": "COMPLETED",
        "audited_symbol": symbol or "ALL_ACTIVE",
        "corrections_triggered_count": len(corrections),
        "corrections_triggered": corrections,
        "active_alerts_remaining": len(trade_audit_service.active_alerts),
    }
