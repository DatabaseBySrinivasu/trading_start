from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class TradeBase(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=10, description="Ticker symbol (e.g. AAPL, BTC-USD)")
    side: OrderSide = Field(..., description="BUY or SELL")
    order_type: OrderType = Field(default=OrderType.MARKET, description="Order execution type")
    quantity: float = Field(..., gt=0, description="Number of shares or units")
    price: Optional[float] = Field(None, gt=0, description="Limit price (required for LIMIT orders)")
    notes: Optional[str] = Field(None, max_length=500, description="Optional trade notes or strategy rationale")


class TradeCreate(TradeBase):
    pass


class TradeUpdate(BaseModel):
    quantity: Optional[float] = Field(None, gt=0)
    price: Optional[float] = Field(None, gt=0)
    status: Optional[OrderStatus] = None
    notes: Optional[str] = Field(None, max_length=500)


class Trade(TradeBase):
    id: int
    status: OrderStatus = OrderStatus.PENDING
    executed_price: Optional[float] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
