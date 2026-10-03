from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from app.services.market_data import market_data_service


class Position:
    def __init__(self, symbol: str, quantity: int, average_price: float):
        self.symbol = symbol
        self.quantity = quantity
        self.average_price = average_price

    def to_dict(self, current_price: Optional[float] = None) -> Dict[str, Any]:
        cur_price = current_price or self.average_price
        invested = self.quantity * self.average_price
        current_val = self.quantity * cur_price
        unrealized_pnl = current_val - invested
        pnl_pct = (unrealized_pnl / invested * 100) if invested > 0 else 0.0

        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "average_price": round(self.average_price, 2),
            "current_price": round(cur_price, 2),
            "invested_value": round(invested, 2),
            "current_value": round(current_val, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "pnl_percent": round(pnl_pct, 2),
        }


class PaperTrader:
    def __init__(self, initial_balance: float = 100000.0):
        self.initial_balance = initial_balance
        self.cash = initial_balance
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[Dict[str, Any]] = []
        self.realized_pnl: float = 0.0

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get live balance, portfolio value, open positions, and total P&L."""
        total_holdings_value = 0.0
        positions_summary = []

        for symbol, pos in self.positions.items():
            live_data = market_data_service.get_live_price(symbol)
            cur_price = live_data["price"] if live_data else pos.average_price
            pos_dict = pos.to_dict(current_price=cur_price)
            total_holdings_value += pos_dict["current_value"]
            positions_summary.append(pos_dict)

        total_portfolio_value = self.cash + total_holdings_value
        total_pnl = total_portfolio_value - self.initial_balance
        total_pnl_pct = (total_pnl / self.initial_balance) * 100

        return {
            "initial_balance": self.initial_balance,
            "available_cash": round(self.cash, 2),
            "holdings_value": round(total_holdings_value, 2),
            "total_portfolio_value": round(total_portfolio_value, 2),
            "realized_pnl": round(self.realized_pnl, 2),
            "total_pnl": round(total_pnl, 2),
            "total_return_pct": round(total_pnl_pct, 2),
            "open_positions_count": len(self.positions),
            "positions": positions_summary,
        }

    def buy(self, symbol: str, quantity: int, price: Optional[float] = None) -> Dict[str, Any]:
        """Execute a simulated BUY order at live or specified price."""
        symbol = symbol.upper()
        if price is None:
            live = market_data_service.get_live_price(symbol)
            if not live:
                return {"success": False, "message": f"Could not fetch live price for {symbol}"}
            price = live["price"]

        cost = quantity * price
        if cost > self.cash:
            return {
                "success": False,
                "message": f"Insufficient funds. Required: ₹{cost:,.2f}, Available: ₹{self.cash:,.2f}",
            }

        self.cash -= cost

        if symbol in self.positions:
            pos = self.positions[symbol]
            total_qty = pos.quantity + quantity
            new_avg = ((pos.quantity * pos.average_price) + (quantity * price)) / total_qty
            pos.quantity = total_qty
            pos.average_price = new_avg
        else:
            self.positions[symbol] = Position(symbol=symbol, quantity=quantity, average_price=price)

        record = {
            "id": len(self.trade_history) + 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": "BUY",
            "symbol": symbol,
            "quantity": quantity,
            "price": round(price, 2),
            "total_amount": round(cost, 2),
            "remaining_cash": round(self.cash, 2),
        }
        self.trade_history.append(record)

        return {
            "success": True,
            "message": f"Successfully bought {quantity} shares of {symbol} at ₹{price:.2f}",
            "order": record,
        }

    def sell(self, symbol: str, quantity: int, price: Optional[float] = None) -> Dict[str, Any]:
        """Execute a simulated SELL order at live or specified price."""
        symbol = symbol.upper()
        if symbol not in self.positions or self.positions[symbol].quantity < quantity:
            held = self.positions[symbol].quantity if symbol in self.positions else 0
            return {
                "success": False,
                "message": f"Insufficient shares to sell. Requested: {quantity}, Held: {held}",
            }

        if price is None:
            live = market_data_service.get_live_price(symbol)
            if not live:
                return {"success": False, "message": f"Could not fetch live price for {symbol}"}
            price = live["price"]

        pos = self.positions[symbol]
        revenue = quantity * price
        cost_basis = quantity * pos.average_price
        pnl = revenue - cost_basis

        self.cash += revenue
        self.realized_pnl += pnl

        pos.quantity -= quantity
        if pos.quantity == 0:
            del self.positions[symbol]

        record = {
            "id": len(self.trade_history) + 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": "SELL",
            "symbol": symbol,
            "quantity": quantity,
            "price": round(price, 2),
            "total_amount": round(revenue, 2),
            "realized_pnl": round(pnl, 2),
            "remaining_cash": round(self.cash, 2),
        }
        self.trade_history.append(record)

        return {
            "success": True,
            "message": f"Successfully sold {quantity} shares of {symbol} at ₹{price:.2f} (P&L: ₹{pnl:+,.2f})",
            "order": record,
        }


# Global paper trader instance with 1 Lakh INR starting balance
paper_trader = PaperTrader(initial_balance=100000.0)
