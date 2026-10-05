from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from app.services.paper_trader import paper_trader, LOT_SIZES

router = APIRouter()


class SimpleOrderRequest(BaseModel):
    symbol: str = Field(..., description="Stock or index symbol, e.g. SBIN, RELIANCE, NIFTY, ^NSEI")
    quantity: int = Field(..., gt=0, description="Number of shares/units")
    price: Optional[float] = Field(None, gt=0, description="Optional limit price. If omitted, uses live market price.")


class UnifiedOrderRequest(BaseModel):
    symbol: str = Field(..., description="Index, Commodity, Option or Stock symbol (e.g. ^NSEI, NIFTY, ^NSEBANK, ^BSESN, CRUDEOIL, RELIANCE)")
    side: Optional[str] = Field("BUY", description="Direction: 'BUY'/'LONG' or 'SELL'/'SHORT'")
    asset_type: Optional[str] = Field(None, description="'INDEX', 'COMMODITY', 'OPTION', or 'EQUITY'")
    lots: Optional[int] = Field(None, ge=1, description="Number of lots (e.g. 1 lot = 65 for Nifty, 20 for Sensex, 30 for Bank Nifty)")
    quantity: Optional[int] = Field(None, ge=1, description="Total units/shares (overridden if lots is provided)")
    price: Optional[float] = Field(None, gt=0, description="Execution price (defaults to live spot / premium price)")
    target_1: Optional[float] = Field(None, gt=0, description="Target 1 exit level")
    target_2: Optional[float] = Field(None, gt=0, description="Target 2 exit level")
    stop_loss: Optional[float] = Field(None, gt=0, description="Stop Loss exit level")
    notes: Optional[str] = Field(None, description="Optional trade rationale or setup notes")


class ClosePositionRequest(BaseModel):
    symbol: str = Field(..., description="Symbol or position key to square off")
    price: Optional[float] = Field(None, gt=0, description="Optional exit price (defaults to live price)")


class ResetPortfolioRequest(BaseModel):
    initial_balance: Optional[float] = Field(500000.0, gt=0, description="Initial virtual capital in INR")


@router.get("/portfolio", summary="Get simulated portfolio summary")
def get_portfolio():
    """Returns cash balance, margin utilized, open positions, total portfolio value, and P&L."""
    return paper_trader.get_portfolio_summary()


@router.post("/order", summary="Place multi-asset simulated order (Index Levels, Options, Commodities, Equities)")
def place_order(order: UnifiedOrderRequest):
    """
    Execute simulated paper trade on Index Levels (Long/Short), Options, Commodities, or Equities.
    Supports lot-based sizing (Nifty 65, Sensex 20, Bank Nifty 30, Crude Oil 100, Natural Gas 1250) and 10% margin leverage for index contracts.
    """
    res = paper_trader.place_order(
        symbol=order.symbol,
        side=order.side or "BUY",
        lots=order.lots,
        quantity=order.quantity,
        asset_type=order.asset_type,
        price=order.price,
        target_1=order.target_1,
        target_2=order.target_2,
        stop_loss=order.stop_loss,
        notes=order.notes,
    )
    if not res["success"]:
        raise HTTPException(status_code=400, detail=res["message"])
    return res


@router.post("/buy", summary="Place simulated BUY order (Backward Compatible)")
def buy_order(order: SimpleOrderRequest):
    """Buy shares / contracts at live market price with virtual funds."""
    res = paper_trader.buy(order.symbol, order.quantity, price=order.price)
    if not res["success"]:
        raise HTTPException(status_code=400, detail=res["message"])
    return res


@router.post("/sell", summary="Place simulated SELL order (Backward Compatible)")
def sell_order(order: SimpleOrderRequest):
    """Sell shares / contracts at live market price."""
    res = paper_trader.sell(order.symbol, order.quantity, price=order.price)
    if not res["success"]:
        raise HTTPException(status_code=400, detail=res["message"])
    return res


@router.post("/close", summary="Square off / close active paper position")
def close_position(req: ClosePositionRequest):
    """Squares off an open paper position at market price and realizes P&L."""
    res = paper_trader.close_position(req.symbol, price=req.price)
    if not res["success"]:
        raise HTTPException(status_code=400, detail=res["message"])
    return res


@router.post("/close-all", summary="Square off all active paper positions")
def close_all():
    """Squares off all open paper positions at market price."""
    return paper_trader.close_all_positions()


@router.post("/reverse", summary="Reverse active paper position (Flip Long <-> Short)")
def reverse_position(req: ClosePositionRequest):
    """Closes current position and immediately enters opposite position with same lot size."""
    res = paper_trader.reverse_position(req.symbol)
    if not res["success"]:
        raise HTTPException(status_code=400, detail=res["message"])
    return res


@router.post("/reset", summary="Reset paper portfolio")
def reset_portfolio(req: Optional[ResetPortfolioRequest] = None):
    """Resets paper trading account to fresh balance and clears positions & history."""
    balance = req.initial_balance if req else 100000.0
    return paper_trader.reset(initial_balance=balance)


@router.get("/lot-sizes", summary="Get standardized lot sizes reference")
def get_lot_sizes():
    """Returns exchange-standardized lot sizes for Indian Indices and MCX Commodities."""
    return {
        "lot_sizes": LOT_SIZES,
        "indices": {
            "NIFTY 50": 65,
            "BANK NIFTY": 30,
            "SENSEX": 20,
            "FINNIFTY": 65,
            "MIDCPNIFTY": 120,
        },
        "commodities": {
            "CRUDE OIL (Brent)": {"lot_size": 100, "unit": "bbl"},
            "NATURAL GAS (HH)": {"lot_size": 1250, "unit": "mmBtu"},
            "GOLD 999": {"lot_size": 100, "unit": "g"},
            "SILVER": {"lot_size": 30, "unit": "kg"},
        }
    }
