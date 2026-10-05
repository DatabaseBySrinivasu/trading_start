import logging
import time
import pyotp
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from SmartApi import SmartConnect
from app.core.config import settings

logger = logging.getLogger(__name__)

# Standard Angel One Tokens for popular Indian instruments
ANGEL_INSTRUMENT_MAP: Dict[str, Dict[str, Any]] = {
    "^NSEI": {"exchange": "NSE", "tradingsymbol": "Nifty 50", "symboltoken": "99926000", "name": "NIFTY 50"},
    "NIFTY": {"exchange": "NSE", "tradingsymbol": "Nifty 50", "symboltoken": "99926000", "name": "NIFTY 50"},
    "^BSESN": {"exchange": "BSE", "tradingsymbol": "SENSEX", "symboltoken": "99919000", "name": "BSE SENSEX"},
    "SENSEX": {"exchange": "BSE", "tradingsymbol": "SENSEX", "symboltoken": "99919000", "name": "BSE SENSEX"},
    "^NSEBANK": {"exchange": "NSE", "tradingsymbol": "Nifty Bank", "symboltoken": "99926009", "name": "BANK NIFTY"},
    "BANKNIFTY": {"exchange": "NSE", "tradingsymbol": "Nifty Bank", "symboltoken": "99926009", "name": "BANK NIFTY"},
    "FINNIFTY": {"exchange": "NSE", "tradingsymbol": "Nifty Fin Services", "symboltoken": "99926037", "name": "FIN NIFTY"},
    "NIFTY_FIN_SERVICE.NS": {"exchange": "NSE", "tradingsymbol": "Nifty Fin Services", "symboltoken": "99926037", "name": "FIN NIFTY"},
    "^NSEMDCP50": {"exchange": "NSE", "tradingsymbol": "NIFTY MID SELECT", "symboltoken": "99926074", "name": "MIDCAP NIFTY"},
    "MIDCPNIFTY": {"exchange": "NSE", "tradingsymbol": "NIFTY MID SELECT", "symboltoken": "99926074", "name": "MIDCAP NIFTY"},
    "SBIN": {"exchange": "NSE", "tradingsymbol": "SBIN-EQ", "symboltoken": "3045", "name": "State Bank of India"},
    "RELIANCE": {"exchange": "NSE", "tradingsymbol": "RELIANCE-EQ", "symboltoken": "2885", "name": "Reliance Industries"},
    "TCS": {"exchange": "NSE", "tradingsymbol": "TCS-EQ", "symboltoken": "11536", "name": "Tata Consultancy Services"},
    "INFY": {"exchange": "NSE", "tradingsymbol": "INFY-EQ", "symboltoken": "1594", "name": "Infosys"},
    "HDFCBANK": {"exchange": "NSE", "tradingsymbol": "HDFCBANK-EQ", "symboltoken": "1333", "name": "HDFC Bank"},
    "ICICIBANK": {"exchange": "NSE", "tradingsymbol": "ICICIBANK-EQ", "symboltoken": "4963", "name": "ICICI Bank"},
    "LT": {"exchange": "NSE", "tradingsymbol": "LT-EQ", "symboltoken": "11483", "name": "Larsen & Toubro"},
    "ITC": {"exchange": "NSE", "tradingsymbol": "ITC-EQ", "symboltoken": "1660", "name": "ITC Ltd"},
    "TATAMOTORS": {"exchange": "NSE", "tradingsymbol": "TATAMOTORS-EQ", "symboltoken": "3456", "name": "Tata Motors"},
    "BHARTIARTL": {"exchange": "NSE", "tradingsymbol": "BHARTIARTL-EQ", "symboltoken": "10604", "name": "Bharti Airtel"},
    "AXISBANK": {"exchange": "NSE", "tradingsymbol": "AXISBANK-EQ", "symboltoken": "5900", "name": "Axis Bank"},
    "KOTAKBANK": {"exchange": "NSE", "tradingsymbol": "KOTAKBANK-EQ", "symboltoken": "1922", "name": "Kotak Mahindra Bank"},
    "CRUDEOIL": {"exchange": "MCX", "tradingsymbol": "CRUDEOIL", "symboltoken": "254471", "name": "Brent Crude Oil (MCX)"},
    "BRENT": {"exchange": "MCX", "tradingsymbol": "CRUDEOIL", "symboltoken": "254471", "name": "Brent Crude Oil (MCX)"},
    "GOLD": {"exchange": "MCX", "tradingsymbol": "GOLD", "symboltoken": "254483", "name": "Gold 999 (MCX)"},
    "SILVER": {"exchange": "MCX", "tradingsymbol": "SILVER", "symboltoken": "254490", "name": "Silver (MCX)"},
    "NATURALGAS": {"exchange": "MCX", "tradingsymbol": "NATURALGAS", "symboltoken": "254495", "name": "Henry Hub Natural Gas (MCX)"},
    "HENRYHUB": {"exchange": "MCX", "tradingsymbol": "NATURALGAS", "symboltoken": "254495", "name": "Henry Hub Natural Gas (MCX)"},
    "COPPER": {"exchange": "MCX", "tradingsymbol": "COPPER", "symboltoken": "254498", "name": "Copper (MCX)"},
}


class AngelOneClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        client_code: Optional[str] = None,
        mpin: Optional[str] = None,
        totp_secret: Optional[str] = None,
    ):
        self.api_key = api_key or settings.ANGEL_API_KEY
        self.client_code = client_code or settings.ANGEL_CLIENT_CODE
        self.mpin = mpin or settings.ANGEL_MPIN
        self.totp_secret = totp_secret or settings.ANGEL_TOTP_SECRET
        self.smart_api: Optional[SmartConnect] = None
        self.session_data: Optional[Dict[str, Any]] = None
        self.is_connected: bool = False
        self.last_login_time: float = 0
        self.last_error: Optional[str] = None
        self._price_cache: Dict[str, Dict[str, Any]] = {}

    def has_credentials(self) -> bool:
        """Check if all required credentials are provided and valid."""
        if not all([self.api_key, self.client_code, self.mpin, self.totp_secret]):
            return False
        if "YOUR_" in str(self.totp_secret) or len(str(self.totp_secret).strip()) < 8:
            return False
        return True

    def login(self) -> Dict[str, Any]:
        """Authenticate with Angel One SmartAPI using TOTP and create a session."""
        if not self.has_credentials():
            err = "Angel One credentials missing or invalid in .env file."
            self.last_error = err
            self.is_connected = False
            raise ValueError(err)

        try:
            self.smart_api = SmartConnect(api_key=self.api_key)
            totp = pyotp.TOTP(self.totp_secret.strip()).now()
            response = self.smart_api.generateSession(self.client_code.strip(), self.mpin.strip(), totp)

            if not response.get("status"):
                error_msg = response.get("message", "Login failed")
                self.last_error = error_msg
                self.is_connected = False
                logger.warning(f"Angel One Login Failed: {error_msg}")
                raise Exception(f"Angel One Login Error: {error_msg}")

            self.session_data = response.get("data")
            self.is_connected = True
            self.last_login_time = time.time()
            self.last_error = None
            logger.info(f"Angel One Login Successful for {self.client_code}")
            return response
        except Exception as e:
            self.is_connected = False
            self.last_error = str(e)
            logger.warning(f"Angel One connection exception: {e}")
            raise

    def ensure_connected(self) -> bool:
        """Ensure session is active or attempt reconnection."""
        if not self.has_credentials():
            return False
        # If session is older than 6 hours, re-login
        if self.is_connected and self.smart_api and (time.time() - self.last_login_time < 21600):
            return True
        try:
            self.login()
            return True
        except Exception:
            return False

    def get_token_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Look up exchange token for a given symbol."""
        sym = symbol.strip().upper().replace(".NS", "").replace(".BO", "")
        if sym in ANGEL_INSTRUMENT_MAP:
            return ANGEL_INSTRUMENT_MAP[sym]
        for k, v in ANGEL_INSTRUMENT_MAP.items():
            if k.upper() == sym or v["tradingsymbol"].upper().startswith(sym):
                return v
        return None

    def get_ltp(self, exchange: str, tradingsymbol: str, symboltoken: str) -> Optional[float]:
        """Fetch Last Traded Price (LTP) from Angel One."""
        if not self.ensure_connected() or not self.smart_api:
            return None
        try:
            res = self.smart_api.ltpData(exchange, tradingsymbol, symboltoken)
            if isinstance(res, dict) and res.get("status") and res.get("data"):
                return float(res["data"]["ltp"])
            elif isinstance(res, dict) and not res.get("status"):
                msg = str(res.get("message", ""))
                err_code = str(res.get("errorcode", res.get("errorCode", "")))
                # Only mark disconnected on session/auth errors
                if "invalid token" in msg.lower() or "session" in msg.lower() or err_code in ["AG8004", "AB8050"]:
                    self.is_connected = False
        except Exception as e:
            err_str = str(e).lower()
            logger.debug(f"Angel One get_ltp error for {tradingsymbol}: {e}")
            if "invalid token" in err_str or "session" in err_str or "ag8004" in err_str:
                self.is_connected = False
        return None

    def get_live_price(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch real-time LTP and day data via Angel One if connected."""
        token_info = self.get_token_info(symbol)
        if not token_info:
            return None

        ltp = self.get_ltp(
            exchange=token_info["exchange"],
            tradingsymbol=token_info["tradingsymbol"],
            symboltoken=token_info["symboltoken"],
        )
        if ltp is not None and ltp > 0:
            return {
                "symbol": symbol.upper(),
                "name": token_info.get("name", symbol),
                "price": round(ltp, 2),
                "source": "ANGEL_ONE_SMARTAPI",
                "exchange": token_info["exchange"],
                "symboltoken": token_info["symboltoken"],
                "timestamp": time.time(),
            }
        return None

    def get_all_live_prices(self) -> Dict[str, Dict[str, Any]]:
        """Fetch real-time quotes for all mapped instruments in Angel One."""
        if not self.ensure_connected() or not self.smart_api:
            return {}

        results: Dict[str, Dict[str, Any]] = {}
        # Fetch tokens
        for sym, token_info in ANGEL_INSTRUMENT_MAP.items():
            if sym.startswith("^") or sym in ["CRUDEOIL", "GOLD", "SILVER", "NATURALGAS", "COPPER"] or sym in ["SBIN", "RELIANCE", "TCS", "INFY", "HDFCBANK"]:
                ltp = self.get_ltp(
                    exchange=token_info["exchange"],
                    tradingsymbol=token_info["tradingsymbol"],
                    symboltoken=token_info["symboltoken"],
                )
                if ltp is not None and ltp > 0:
                    results[sym] = {
                        "symbol": sym,
                        "name": token_info.get("name", sym),
                        "price": round(ltp, 2),
                        "source": "ANGEL_ONE_SMARTAPI",
                        "exchange": token_info["exchange"],
                        "symboltoken": token_info["symboltoken"],
                        "timestamp": time.time(),
                    }
        return results

    def get_historical_candles(
        self,
        symbol_token: str,
        exchange: str = "NSE",
        interval: str = "FIVE_MINUTE",
        days_back: int = 5,
    ) -> Optional[pd.DataFrame]:
        """Fetch OHLCV candle data from Angel One."""
        if not self.ensure_connected() or not self.smart_api:
            return None

        to_date = datetime.now()
        from_date = to_date - timedelta(days=days_back)

        params = {
            "exchange": exchange,
            "symboltoken": symbol_token,
            "interval": interval,
            "fromdate": from_date.strftime("%Y-%m-%d %H:%M"),
            "todate": to_date.strftime("%Y-%m-%d %H:%M"),
        }

        try:
            res = self.smart_api.getCandleData(params)
            if not res.get("status") or not res.get("data"):
                logger.warning(f"Failed to fetch Angel One candles: {res}")
                return None

            df = pd.DataFrame(
                res["data"],
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)
            return df
        except Exception as e:
            logger.debug(f"Angel One getCandleData error: {e}")
            return None

    def place_order(
        self,
        symbol: str,
        token: str,
        quantity: int,
        transaction_type: str = "BUY",
        order_type: str = "MARKET",
        product_type: str = "INTRADAY",
        price: float = 0.0,
        exchange: str = "NSE",
    ) -> Optional[str]:
        """Place an order with Angel One."""
        if not self.ensure_connected() or not self.smart_api:
            raise Exception("Angel One client is not connected. Please check your credentials.")

        params = {
            "variety": "NORMAL",
            "tradingsymbol": symbol,
            "symboltoken": token,
            "transactiontype": transaction_type.upper(),
            "exchange": exchange,
            "ordertype": order_type.upper(),
            "producttype": product_type.upper(),
            "duration": "DAY",
            "price": str(price) if order_type.upper() == "LIMIT" else "0",
            "quantity": str(quantity),
        }

        order_res = self.smart_api.placeOrder(params)
        logger.info(f"Angel One Order placed: {order_res}")
        return order_res

    def get_status(self) -> Dict[str, Any]:
        """Return Angel One connection and credential status."""
        has_creds = self.has_credentials()
        return {
            "is_configured": has_creds,
            "is_connected": self.is_connected,
            "client_code": self.client_code if has_creds else None,
            "last_login": datetime.fromtimestamp(self.last_login_time).strftime("%Y-%m-%d %H:%M:%S") if self.last_login_time > 0 else None,
            "last_error": self.last_error,
            "supported_instruments": list(ANGEL_INSTRUMENT_MAP.keys()),
        }


# Global instance
angel_client = AngelOneClient()
