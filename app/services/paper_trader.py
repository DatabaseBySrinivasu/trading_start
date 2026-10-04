from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from app.services.market_data import market_data_service


# Standard Exchange Standardized Lot Sizes & Multipliers
LOT_SIZES: Dict[str, int] = {
    "^NSEI": 65,
    "NIFTY": 65,
    "NIFTY 50": 65,
    "NIFTY50": 65,
    "^NSEBANK": 30,
    "BANKNIFTY": 30,
    "BANK NIFTY": 30,
    "^BSESN": 20,
    "SENSEX": 20,
    "BSE SENSEX": 20,
    "FINNIFTY": 65,
    "MIDCPNIFTY": 120,
    "CRUDEOIL": 100,
    "CRUDE OIL": 100,
    "BZ=F": 100,
    "NATURALGAS": 1250,
    "NATURAL GAS": 1250,
    "NG=F": 1250,
    "GOLD": 100,
    "GC=F": 100,
    "SILVER": 30,
    "SI=F": 30,
    "COPPER": 2500,
}

INDEX_DISPLAY_NAMES: Dict[str, str] = {
    "^NSEI": "NIFTY 50",
    "NIFTY": "NIFTY 50",
    "NIFTY 50": "NIFTY 50",
    "^NSEBANK": "BANK NIFTY",
    "BANKNIFTY": "BANK NIFTY",
    "BANK NIFTY": "BANK NIFTY",
    "^BSESN": "SENSEX",
    "SENSEX": "SENSEX",
    "CRUDEOIL": "CRUDE OIL",
    "CRUDE OIL": "CRUDE OIL",
    "BZ=F": "CRUDE OIL",
    "NATURALGAS": "NATURAL GAS",
    "NATURAL GAS": "NATURAL GAS",
    "NG=F": "NATURAL GAS",
    "GOLD": "GOLD 999",
    "GC=F": "GOLD 999",
    "SILVER": "SILVER",
    "SI=F": "SILVER",
}


def normalize_symbol_and_asset(symbol_in: str, explicit_asset: Optional[str] = None) -> tuple[str, str, str, int]:
    """
    Resolves standard ticker, display name, asset type, and standard lot size.
    Returns: (canonical_symbol, display_name, asset_type, lot_size)
    """
    s_upper = symbol_in.strip().upper()

    # Index Mappings
    if s_upper in ["NIFTY", "NIFTY 50", "NIFTY50", "^NSEI"]:
        return "^NSEI", "NIFTY 50", "INDEX", 65
    elif s_upper in ["BANKNIFTY", "BANK NIFTY", "^NSEBANK"]:
        return "^NSEBANK", "BANK NIFTY", "INDEX", 30
    elif s_upper in ["SENSEX", "BSE SENSEX", "^BSESN"]:
        return "^BSESN", "SENSEX", "INDEX", 20
    elif s_upper in ["FINNIFTY", "NIFTY IT"]:
        return s_upper, s_upper, "INDEX", 65
    elif s_upper in ["MIDCPNIFTY"]:
        return s_upper, s_upper, "INDEX", 120

    # Commodity Mappings
    elif s_upper in ["CRUDEOIL", "CRUDE OIL", "BZ=F"]:
        return "CRUDEOIL", "CRUDE OIL (Brent)", "COMMODITY", 100
    elif s_upper in ["NATURALGAS", "NATURAL GAS", "NG=F"]:
        return "NATURALGAS", "NATURAL GAS (HH)", "COMMODITY", 1250
    elif s_upper in ["GOLD", "GC=F"]:
        return "GOLD", "GOLD (MCX)", "COMMODITY", 100
    elif s_upper in ["SILVER", "SI=F"]:
        return "SILVER", "SILVER (MCX)", "COMMODITY", 30

    # Option Contract Check (e.g. NIFTY 24500 CE)
    elif " CE" in s_upper or " PE" in s_upper:
        lot = 65
        for k, v in LOT_SIZES.items():
            if k in s_upper:
                lot = v
                break
        return s_upper, s_upper, "OPTION", lot

    # Default Equity / Stock
    asset = explicit_asset or "EQUITY"
    lot = LOT_SIZES.get(s_upper, 1)
    return s_upper, s_upper, asset, lot


class Position:
    def __init__(
        self,
        symbol: str,
        quantity: int,
        average_price: float,
        side: str = "LONG",
        asset_type: str = "EQUITY",
        lots: int = 1,
        lot_size: int = 1,
        display_name: Optional[str] = None,
        target_1: Optional[float] = None,
        target_2: Optional[float] = None,
        stop_loss: Optional[float] = None,
        margin_used: Optional[float] = None,
        created_at: Optional[str] = None,
    ):
        self.symbol = symbol
        self.quantity = quantity
        self.average_price = average_price
        self.side = side.upper()  # "LONG" or "SHORT"
        self.asset_type = asset_type.upper()  # "INDEX", "COMMODITY", "OPTION", "EQUITY"
        self.lot_size = lot_size
        self.lots = lots if lots > 0 else max(1, quantity // lot_size if lot_size > 0 else 1)
        self.display_name = display_name or symbol
        self.target_1 = target_1
        self.target_2 = target_2
        self.stop_loss = stop_loss
        self.created_at = created_at or datetime.now(timezone.utc).isoformat()

        # Margin computation (Index / Commodities 10% span margin, Option/Equity 100% cash)
        if margin_used is not None:
            self.margin_used = margin_used
        else:
            if self.asset_type in ["INDEX", "COMMODITY"]:
                self.margin_used = (self.quantity * self.average_price) * 0.10  # 10% Margin requirement
            else:
                self.margin_used = self.quantity * self.average_price

    def to_dict(self, current_price: Optional[float] = None) -> Dict[str, Any]:
        cur_price = current_price if (current_price is not None and current_price > 0) else self.average_price

        # Point-Based Calculation
        if self.side in ["LONG", "BUY"]:
            points_pnl = cur_price - self.average_price
        else:  # SHORT / SELL
            points_pnl = self.average_price - cur_price

        # Total Rupee P&L
        unrealized_pnl = points_pnl * self.quantity
        notional_val = self.quantity * cur_price

        # P&L Percentage (Return on Margin)
        pnl_pct = (unrealized_pnl / self.margin_used * 100) if self.margin_used > 0 else 0.0

        # Target & SL Hit Status Evaluation
        target_status = "ACTIVE"
        if self.side in ["LONG", "BUY"]:
            if self.target_2 and cur_price >= self.target_2:
                target_status = "TARGET_2_HIT"
            elif self.target_1 and cur_price >= self.target_1:
                target_status = "TARGET_1_HIT"
            elif self.stop_loss and cur_price <= self.stop_loss:
                target_status = "SL_HIT"
        else:  # SHORT
            if self.target_2 and cur_price <= self.target_2:
                target_status = "TARGET_2_HIT"
            elif self.target_1 and cur_price <= self.target_1:
                target_status = "TARGET_1_HIT"
            elif self.stop_loss and cur_price >= self.stop_loss:
                target_status = "SL_HIT"

        return {
            "symbol": self.symbol,
            "display_name": self.display_name,
            "side": self.side,
            "asset_type": self.asset_type,
            "quantity": self.quantity,
            "lots": self.lots,
            "lot_size": self.lot_size,
            "average_price": round(self.average_price, 2),
            "average_buy_price": round(self.average_price, 2),  # Backward compatibility
            "current_price": round(cur_price, 2),
            "points_pnl": round(points_pnl, 2),
            "margin_used": round(self.margin_used, 2),
            "invested_value": round(self.margin_used, 2),  # Backward compatibility
            "notional_value": round(notional_val, 2),
            "current_value": round(self.margin_used + unrealized_pnl, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "pnl_percent": round(pnl_pct, 2),
            "target_1": round(self.target_1, 2) if self.target_1 else None,
            "target_2": round(self.target_2, 2) if self.target_2 else None,
            "stop_loss": round(self.stop_loss, 2) if self.stop_loss else None,
            "target_status": target_status,
            "created_at": self.created_at,
        }


class PositionsDict(dict):
    """Enhanced dictionary supporting flexible symbol and side resolution."""
    def __getitem__(self, key):
        if super().__contains__(key):
            return super().__getitem__(key)
        k_up = str(key).upper()
        for k, v in self.items():
            if k.upper() == f"{k_up}_LONG" or k.upper() == f"{k_up}_SHORT" or k.upper().startswith(k_up):
                return v
        raise KeyError(key)

    def __contains__(self, key):
        if super().__contains__(key):
            return True
        k_up = str(key).upper()
        for k in self.keys():
            if k.upper() == f"{k_up}_LONG" or k.upper() == f"{k_up}_SHORT" or k.upper().startswith(k_up):
                return True
        return False

    def __delitem__(self, key):
        if super().__contains__(key):
            return super().__delitem__(key)
        k_up = str(key).upper()
        for k in list(self.keys()):
            if k.upper() == f"{k_up}_LONG" or k.upper() == f"{k_up}_SHORT" or k.upper().startswith(k_up):
                return super().__delitem__(k)
        raise KeyError(key)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


class PaperTrader:
    def __init__(self, initial_balance: float = 500000.0):
        self.initial_balance = initial_balance
        self.cash = initial_balance
        self.positions: PositionsDict = PositionsDict()
        self.trade_history: List[Dict[str, Any]] = []
        self.realized_pnl: float = 0.0

    def _get_live_price(self, symbol: str) -> Optional[float]:
        """Fetch live ticker price with commodity & index resolution."""
        live_data = market_data_service.get_live_price(symbol)
        if live_data and "price" in live_data and live_data["price"] > 0:
            return live_data["price"]

        # Fallback mappings for commodities
        if symbol.upper() in ["CRUDEOIL", "CRUDE OIL"]:
            live_bz = market_data_service.get_live_price("BZ=F")
            if live_bz and "price" in live_bz:
                return live_bz["price"]
        elif symbol.upper() in ["NATURALGAS", "NATURAL GAS"]:
            live_ng = market_data_service.get_live_price("NG=F")
            if live_ng and "price" in live_ng:
                return live_ng["price"]
        elif symbol.upper() in ["GOLD"]:
            live_gc = market_data_service.get_live_price("GC=F")
            if live_gc and "price" in live_gc:
                return live_gc["price"]
        elif symbol.upper() in ["SILVER"]:
            live_si = market_data_service.get_live_price("SI=F")
            if live_si and "price" in live_si:
                return live_si["price"]

        return None

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get live balance, margin utilized, open positions, and total P&L."""
        total_margin_used = 0.0
        total_unrealized_pnl = 0.0
        positions_summary = []

        for key, pos in self.positions.items():
            cur_price = self._get_live_price(pos.symbol) or pos.average_price
            pos_dict = pos.to_dict(current_price=cur_price)
            total_margin_used += pos_dict["margin_used"]
            total_unrealized_pnl += pos_dict["unrealized_pnl"]
            positions_summary.append(pos_dict)

        total_portfolio_value = self.cash + total_margin_used + total_unrealized_pnl
        total_pnl = self.realized_pnl + total_unrealized_pnl
        total_pnl_pct = (total_pnl / self.initial_balance) * 100 if self.initial_balance > 0 else 0.0

        # Calculate Win Rate from history
        winning_trades = sum(1 for t in self.trade_history if t.get("realized_pnl", 0) > 0)
        closed_trades = sum(1 for t in self.trade_history if "realized_pnl" in t)
        win_rate = (winning_trades / closed_trades * 100) if closed_trades > 0 else 0.0

        return {
            "initial_balance": round(self.initial_balance, 2),
            "available_cash": round(self.cash, 2),
            "margin_used": round(total_margin_used, 2),
            "holdings_value": round(total_margin_used + total_unrealized_pnl, 2),
            "total_portfolio_value": round(total_portfolio_value, 2),
            "realized_pnl": round(self.realized_pnl, 2),
            "unrealized_pnl": round(total_unrealized_pnl, 2),
            "total_pnl": round(total_pnl, 2),
            "total_return_pct": round(total_pnl_pct, 2),
            "win_rate_pct": round(win_rate, 1),
            "closed_trades_count": closed_trades,
            "open_positions_count": len(self.positions),
            "positions": positions_summary,
            "trade_history": self.trade_history[-20:],  # Last 20 executed orders
        }

    def place_order(
        self,
        symbol: str,
        side: str = "BUY",
        lots: Optional[int] = None,
        quantity: Optional[int] = None,
        asset_type: Optional[str] = None,
        price: Optional[float] = None,
        target_1: Optional[float] = None,
        target_2: Optional[float] = None,
        stop_loss: Optional[float] = None,
        notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Comprehensive order execution for Index Levels (Futures / Spot), Commodities, Options, and Stocks.
        Supports both LONG (BUY) and SHORT (SELL) positions.
        """
        canonical_sym, display_name, resolved_asset, default_lot = normalize_symbol_and_asset(symbol, asset_type)

        # Standardize side
        side_clean = "LONG" if side.upper() in ["BUY", "LONG", "CE", "CALL"] else "SHORT"

        # Resolve quantity and lots
        if lots is not None and lots > 0:
            qty = lots * default_lot
            num_lots = lots
        elif quantity is not None and quantity > 0:
            qty = quantity
            num_lots = max(1, quantity // default_lot if default_lot > 0 else 1)
        else:
            num_lots = 1
            qty = default_lot

        # Live Price Fetching
        if price is None or price <= 0:
            price = self._get_live_price(canonical_sym)
            if price is None or price <= 0:
                # Mock baseline prices if market closed
                if canonical_sym == "^NSEI":
                    price = 24500.0
                elif canonical_sym == "^NSEBANK":
                    price = 52000.0
                elif canonical_sym == "^BSESN":
                    price = 80000.0
                elif canonical_sym == "CRUDEOIL":
                    price = 6150.0
                elif canonical_sym == "NATURALGAS":
                    price = 230.0
                else:
                    return {"success": False, "message": f"Could not fetch live market price for {symbol}"}

        # Calculate Margin Requirement
        if resolved_asset in ["INDEX", "COMMODITY"]:
            required_margin = (qty * price) * 0.10  # 10% span margin leverage
        else:
            required_margin = qty * price

        if required_margin > self.cash:
            return {
                "success": False,
                "message": f"Insufficient funds / virtual cash. Required margin: ₹{required_margin:,.2f}, Available: ₹{self.cash:,.2f}",
            }

        # Deduct margin from available cash
        self.cash -= required_margin

        # Position Key (Track per symbol & side)
        pos_key = f"{canonical_sym}_{side_clean}"

        if pos_key in self.positions:
            pos = self.positions[pos_key]
            total_qty = pos.quantity + qty
            total_lots = pos.lots + num_lots
            new_avg = ((pos.quantity * pos.average_price) + (qty * price)) / total_qty
            pos.quantity = total_qty
            pos.lots = total_lots
            pos.average_price = new_avg
            pos.margin_used += required_margin
            if target_1: pos.target_1 = target_1
            if target_2: pos.target_2 = target_2
            if stop_loss: pos.stop_loss = stop_loss
        else:
            self.positions[pos_key] = Position(
                symbol=canonical_sym,
                display_name=display_name,
                quantity=qty,
                average_price=price,
                side=side_clean,
                asset_type=resolved_asset,
                lots=num_lots,
                lot_size=default_lot,
                target_1=target_1,
                target_2=target_2,
                stop_loss=stop_loss,
                margin_used=required_margin,
            )

        order_record = {
            "id": len(self.trade_history) + 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": f"{side_clean}_ENTRY",
            "side": side_clean,
            "symbol": canonical_sym,
            "display_name": display_name,
            "asset_type": resolved_asset,
            "lots": num_lots,
            "lot_size": default_lot,
            "quantity": qty,
            "price": round(price, 2),
            "margin_used": round(required_margin, 2),
            "target_1": round(target_1, 2) if target_1 else None,
            "target_2": round(target_2, 2) if target_2 else None,
            "stop_loss": round(stop_loss, 2) if stop_loss else None,
            "remaining_cash": round(self.cash, 2),
            "notes": notes or f"Paper trade {resolved_asset} {side_clean} @ ₹{price:,.2f}",
        }
        self.trade_history.append(order_record)

        direction_label = "LONG (BUY)" if side_clean == "LONG" else "SHORT (SELL)"
        return {
            "success": True,
            "message": f"Successfully opened {direction_label} on {display_name} ({num_lots} lot{'s' if num_lots>1 else ''} / {qty} qty) at ₹{price:,.2f}",
            "order": order_record,
            "position": self.positions[pos_key].to_dict(current_price=price),
        }

    def close_position(self, pos_key_or_symbol: str, price: Optional[float] = None) -> Dict[str, Any]:
        """Square off / close an open paper position at market price."""
        # Find position matching key or symbol
        target_key = None
        if pos_key_or_symbol in self.positions:
            target_key = pos_key_or_symbol
        else:
            for k in self.positions.keys():
                if k.startswith(pos_key_or_symbol.upper()) or k == pos_key_or_symbol:
                    target_key = k
                    break

        if not target_key or target_key not in self.positions:
            return {"success": False, "message": f"No active open position found for {pos_key_or_symbol}"}

        pos = self.positions[target_key]
        if price is None or price <= 0:
            price = self._get_live_price(pos.symbol) or pos.average_price

        # Compute Realized P&L
        if pos.side == "LONG":
            points = price - pos.average_price
        else:  # SHORT
            points = pos.average_price - price

        pnl = points * pos.quantity

        # Release Margin + Credit Realized P&L
        self.cash += (pos.margin_used + pnl)
        self.realized_pnl += pnl

        order_record = {
            "id": len(self.trade_history) + 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": f"{pos.side}_EXIT",
            "side": pos.side,
            "symbol": pos.symbol,
            "display_name": pos.display_name,
            "asset_type": pos.asset_type,
            "lots": pos.lots,
            "quantity": pos.quantity,
            "entry_price": round(pos.average_price, 2),
            "exit_price": round(price, 2),
            "points_pnl": round(points, 2),
            "realized_pnl": round(pnl, 2),
            "margin_released": round(pos.margin_used, 2),
            "remaining_cash": round(self.cash, 2),
        }
        self.trade_history.append(order_record)

        del self.positions[target_key]

        return {
            "success": True,
            "message": f"Successfully squared off {pos.display_name} {pos.side} at ₹{price:,.2f} | Points: {points:+,.2f} pts | P&L: ₹{pnl:+,.2f}",
            "realized_pnl": round(pnl, 2),
            "points_pnl": round(points, 2),
            "margin_released": round(pos.margin_used, 2),
            "order": order_record,
        }

    def close_all_positions(self) -> Dict[str, Any]:
        """Square off all currently open paper positions."""
        keys = list(self.positions.keys())
        if not keys:
            return {"success": True, "message": "No open positions to close", "closed_count": 0}

        results = []
        for k in keys:
            res = self.close_position(k)
            results.append(res)

        return {
            "success": True,
            "message": f"Successfully squared off all {len(keys)} positions",
            "closed_count": len(keys),
            "results": results,
        }

    def reverse_position(self, pos_key_or_symbol: str) -> Dict[str, Any]:
        """Square off current position and instantly open opposite side position."""
        target_key = None
        if pos_key_or_symbol in self.positions:
            target_key = pos_key_or_symbol
        else:
            for k in self.positions.keys():
                if k.startswith(pos_key_or_symbol.upper()) or k == pos_key_or_symbol:
                    target_key = k
                    break

        if not target_key:
            return {"success": False, "message": f"No active position to reverse for {pos_key_or_symbol}"}

        pos = self.positions[target_key]
        opp_side = "SHORT" if pos.side == "LONG" else "LONG"
        sym = pos.symbol
        lots = pos.lots
        asset = pos.asset_type

        # 1. Close current
        close_res = self.close_position(target_key)
        if not close_res["success"]:
            return close_res

        # 2. Open opposite
        open_res = self.place_order(symbol=sym, side=opp_side, lots=lots, asset_type=asset)
        return {
            "success": open_res["success"],
            "message": f"Reversed position on {sym} to {opp_side}: {open_res['message']}",
            "new_position": open_res.get("position"),
            "close_details": close_res,
            "open_details": open_res,
        }

    def reset(self, initial_balance: float = 100000.0) -> Dict[str, Any]:
        """Reset paper trader portfolio to fresh initial balance."""
        self.initial_balance = initial_balance
        self.cash = initial_balance
        self.positions.clear()
        self.trade_history.clear()
        self.realized_pnl = 0.0
        return {
            "success": True,
            "message": f"Portfolio reset successfully with virtual balance of ₹{initial_balance:,.2f}",
            "initial_balance": self.initial_balance,
            "available_cash": self.cash,
            "cash": self.cash,
        }

    # Backward compatibility methods for existing endpoints/tests
    def buy(self, symbol: str, quantity: int, price: Optional[float] = None) -> Dict[str, Any]:
        res = self.place_order(symbol=symbol, side="BUY", quantity=quantity, price=price)
        return res

    def sell(self, symbol: str, quantity: int, price: Optional[float] = None) -> Dict[str, Any]:
        # If user holds long position, sell acts as closing/reducing
        pos_key = f"{symbol.upper()}_LONG"
        if pos_key in self.positions:
            pos = self.positions[pos_key]
            if pos.quantity <= quantity:
                return self.close_position(pos_key, price=price)
            else:
                # Partial reduction
                if price is None:
                    price = self._get_live_price(symbol) or pos.average_price
                points = price - pos.average_price
                pnl = points * quantity
                released_margin = (quantity * pos.average_price) * 0.10 if pos.asset_type in ["INDEX", "COMMODITY"] else quantity * pos.average_price
                self.cash += (released_margin + pnl)
                self.realized_pnl += pnl
                pos.quantity -= quantity
                pos.margin_used -= released_margin
                pos.lots = max(1, pos.quantity // pos.lot_size if pos.lot_size > 0 else 1)
                record = {
                    "id": len(self.trade_history) + 1,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "type": "SELL_PARTIAL",
                    "symbol": symbol,
                    "quantity": quantity,
                    "price": round(price, 2),
                    "realized_pnl": round(pnl, 2),
                    "remaining_cash": round(self.cash, 2),
                }
                self.trade_history.append(record)
                return {
                    "success": True,
                    "message": f"Successfully sold {quantity} shares of {symbol} at ₹{price:.2f} (P&L: ₹{pnl:+,.2f})",
                    "order": record,
                }
        else:
            # Cannot sell an unheld asset in simple sell endpoint
            return {
                "success": False,
                "message": f"Cannot sell {symbol}: Insufficient shares or no active position held.",
            }


# Global paper trader instance with 1 Lakh INR starting balance
paper_trader = PaperTrader(initial_balance=100000.0)
