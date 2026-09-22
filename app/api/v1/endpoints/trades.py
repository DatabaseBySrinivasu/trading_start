from datetime import datetime, timezone
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query, status
from app.models.trade import Trade, TradeCreate, TradeUpdate, OrderSide, OrderStatus

router = APIRouter()

# In-memory storage for demonstration / starter
_trades_db: dict[int, Trade] = {
    1: Trade(
        id=1,
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=10.0,
        price=185.50,
        executed_price=185.50,
        status=OrderStatus.FILLED,
        notes="Breakout above 50-day moving average",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc)
    ),
    2: Trade(
        id=2,
        symbol="NVDA",
        side=OrderSide.BUY,
        quantity=5.0,
        price=120.00,
        status=OrderStatus.PENDING,
        notes="Momentum entry",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc)
    )
}
_id_counter = 2


@router.get("/", response_model=List[Trade], summary="List trades")
def list_trades(
    symbol: Optional[str] = Query(None, description="Filter trades by ticker symbol"),
    status: Optional[OrderStatus] = Query(None, description="Filter trades by order status"),
    skip: int = Query(0, ge=0, description="Pagination offset"),
    limit: int = Query(20, ge=1, le=100, description="Pagination limit")
):
    """Retrieve trades with optional filtering by symbol and status."""
    trades = list(_trades_db.values())
    if symbol:
        trades = [t for t in trades if t.symbol.upper() == symbol.upper()]
    if status:
        trades = [t for t in trades if t.status == status]
    return trades[skip : skip + limit]


@router.get("/{trade_id}", response_model=Trade, summary="Get trade by ID")
def get_trade(trade_id: int):
    """Retrieve details of a single trade by ID."""
    if trade_id not in _trades_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trade with id {trade_id} not found"
        )
    return _trades_db[trade_id]


@router.post("/", response_model=Trade, status_code=status.HTTP_201_CREATED, summary="Create new trade/order")
def create_trade(trade_in: TradeCreate):
    """Place a new trade or order."""
    global _id_counter
    now = datetime.now(timezone.utc)
    _id_counter += 1
    new_trade = Trade(
        id=_id_counter,
        symbol=trade_in.symbol.upper(),
        side=trade_in.side,
        order_type=trade_in.order_type,
        quantity=trade_in.quantity,
        price=trade_in.price,
        notes=trade_in.notes,
        status=OrderStatus.PENDING,
        created_at=now,
        updated_at=now
    )
    _trades_db[_id_counter] = new_trade
    return new_trade


@router.put("/{trade_id}", response_model=Trade, summary="Update trade")
def update_trade(trade_id: int, trade_in: TradeUpdate):
    """Update fields or status of an existing trade."""
    if trade_id not in _trades_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trade with id {trade_id} not found"
        )
    existing_trade = _trades_db[trade_id]
    update_data = trade_in.model_dump(exclude_unset=True)
    update_data["updated_at"] = datetime.now(timezone.utc)

    updated_trade = existing_trade.model_copy(update=update_data)
    _trades_db[trade_id] = updated_trade
    return updated_trade


@router.delete("/{trade_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Cancel/Delete trade")
def delete_trade(trade_id: int):
    """Cancel and remove a trade by ID."""
    if trade_id not in _trades_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trade with id {trade_id} not found"
        )
    del _trades_db[trade_id]
    return None
