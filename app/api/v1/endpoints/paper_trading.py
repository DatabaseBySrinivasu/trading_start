from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from app.services.paper_trader import paper_trader

router = APIRouter()


class OrderRequest(BaseModel):
    symbol: str = Field(..., description="Stock symbol, e.g. SBIN, RELIANCE, TCS")
    quantity: int = Field(..., gt=0, description="Number of shares")
    price: Optional[float] = Field(None, gt=0, description="Optional limit price. If omitted, uses live market price.")


@router.get("/portfolio", summary="Get simulated portfolio summary")
def get_portfolio():
    """Returns cash balance, open positions, total portfolio value, and P&L."""
    return paper_trader.get_portfolio_summary()


@router.post("/buy", summary="Place simulated BUY order")
def buy_order(order: OrderRequest):
    """Buy shares at live market price with virtual funds."""
    res = paper_trader.buy(order.symbol, order.quantity, price=order.price)
    if not res["success"]:
        raise HTTPException(status_code=400, detail=res["message"])
    return res


@router.post("/sell", summary="Place simulated SELL order")
def sell_order(order: OrderRequest):
    """Sell shares at live market price."""
    res = paper_trader.sell(order.symbol, order.quantity, price=order.price)
    if not res["success"]:
        raise HTTPException(status_code=400, detail=res["message"])
    return res
