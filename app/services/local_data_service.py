import os
import json
import time
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List, Union
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))


class LocalDataService:
    """
    Local Market Data & Historical Candle Persistence Engine.

    Responsibilities:
    1. Persist multi-timeframe historical candle datasets (15m, 1h, 1d, 1wk) to local disk (CSV / JSON).
    2. Incrementally merge incoming live/remote candles with existing historical data without duplicates.
    3. Log real-time live price ticks (1-second frequency) to local JSONL tick archives.
    4. Persist real-time market snapshots and technical factor state for offline strategy execution and replay.
    5. Provide local-first candle retrieval so strategies can execute with sub-millisecond local latency.
    """

    def __init__(self, base_dir: Optional[Union[str, Path]] = None):
        if base_dir is None:
            # Default to data/ directory at project root
            self.base_dir = Path(__file__).resolve().parent.parent.parent / "data"
        else:
            self.base_dir = Path(base_dir)

        self.historical_dir = self.base_dir / "historical"
        self.live_dir = self.base_dir / "live"
        self.snapshots_dir = self.base_dir / "snapshots"

        # Create directories
        for d in [self.base_dir, self.historical_dir, self.live_dir, self.snapshots_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self._tick_buffer: List[Dict[str, Any]] = []
        self._last_tick_flush: float = time.time()

    @staticmethod
    def sanitize_symbol_filename(symbol: str) -> str:
        """Sanitizes ticker symbols for cross-platform safe file names."""
        clean = symbol.strip().upper()
        clean = clean.replace("^", "IDX_").replace("=", "_").replace(".", "_").replace(":", "_").replace("/", "_")
        return clean

    def _get_candle_filepath(self, symbol: str, interval: str, ext: str = "csv") -> Path:
        safe_sym = self.sanitize_symbol_filename(symbol)
        safe_interval = interval.strip().lower()
        return self.historical_dir / f"{safe_sym}_{safe_interval}.{ext}"

    def save_candles(self, symbol: str, interval: str, df: pd.DataFrame) -> str:
        """
        Saves candle DataFrame to local disk in CSV format.
        Guarantees timestamp index, columns (open, high, low, close, volume), and chronological sorting.
        """
        if df is None or df.empty:
            return ""

        file_path = self._get_candle_filepath(symbol, interval, ext="csv")
        save_df = df.copy()

        # Ensure index is datetime
        if not isinstance(save_df.index, pd.DatetimeIndex):
            if "timestamp" in save_df.columns:
                save_df["timestamp"] = pd.to_datetime(save_df["timestamp"])
                save_df = save_df.set_index("timestamp")
            else:
                save_df.index = pd.to_datetime(save_df.index)

        save_df.index.name = "timestamp"

        # Ensure lowercase standard OHLCV columns
        save_df.columns = [str(c).lower() for c in save_df.columns]
        req_cols = ["open", "high", "low", "close"]
        for col in req_cols:
            if col not in save_df.columns:
                logger.warning(f"Candle DataFrame for {symbol} missing required column: {col}")
                return ""

        if "volume" not in save_df.columns:
            save_df["volume"] = 0

        cols_to_save = ["open", "high", "low", "close", "volume"]
        extra_cols = [c for c in save_df.columns if c not in cols_to_save]
        final_cols = cols_to_save + extra_cols

        save_df = save_df[final_cols].sort_index()
        # Remove duplicate timestamps keeping the latest
        save_df = save_df[~save_df.index.duplicated(keep="last")]

        try:
            save_df.to_csv(file_path, index=True)
            logger.debug(f"Saved {len(save_df)} candles to local file: {file_path}")
            return str(file_path)
        except Exception as e:
            logger.error(f"Failed to save candles to {file_path}: {e}")
            return ""

    def load_candles(self, symbol: str, interval: str) -> Optional[pd.DataFrame]:
        """
        Loads stored candle DataFrame from local disk.
        Returns None if no local data exists.
        """
        file_path = self._get_candle_filepath(symbol, interval, ext="csv")
        if not file_path.exists():
            return None

        try:
            df = pd.read_csv(file_path, index_col="timestamp", parse_dates=True)
            if df.empty:
                return None
            df.columns = [str(c).lower() for c in df.columns]
            return df
        except Exception as e:
            logger.error(f"Failed to load local candles from {file_path}: {e}")
            return None

    def merge_and_save_candles(self, symbol: str, interval: str, new_df: pd.DataFrame) -> pd.DataFrame:
        """
        Loads existing local historical data, merges incoming new candles,
        resolves conflicts by prioritizing newer data, and updates the local disk.
        """
        if new_df is None or new_df.empty:
            existing = self.load_candles(symbol, interval)
            return existing if existing is not None else pd.DataFrame()

        clean_new = new_df.copy()
        if not isinstance(clean_new.index, pd.DatetimeIndex):
            if "timestamp" in clean_new.columns:
                clean_new["timestamp"] = pd.to_datetime(clean_new["timestamp"])
                clean_new = clean_new.set_index("timestamp")
            else:
                clean_new.index = pd.to_datetime(clean_new.index)

        clean_new.index.name = "timestamp"
        clean_new.columns = [str(c).lower() for c in clean_new.columns]

        existing_df = self.load_candles(symbol, interval)
        if existing_df is None or existing_df.empty:
            self.save_candles(symbol, interval, clean_new)
            return clean_new

        # Combine old data and new data
        combined_df = pd.concat([existing_df, clean_new])
        combined_df = combined_df.sort_index()
        # Deduplicate timestamps, keeping the newest entry
        combined_df = combined_df[~combined_df.index.duplicated(keep="last")]

        self.save_candles(symbol, interval, combined_df)
        return combined_df

    def log_live_tick(self, symbol: str, tick_data: Dict[str, Any]):
        """
        Appends real-time 1-second price ticks to local daily tick log.
        """
        if not tick_data:
            return

        now_ist = datetime.now(IST)
        date_str = now_ist.strftime("%Y-%m-%d")
        tick_record = {
            "timestamp": now_ist.isoformat(),
            "epoch": time.time(),
            "symbol": symbol,
            "price": tick_data.get("price"),
            "change": tick_data.get("change"),
            "change_percent": tick_data.get("change_percent"),
            "source": tick_data.get("source", "LIVE_FEED"),
            "tick_direction": tick_data.get("tick_direction", "FLAT"),
        }

        self._tick_buffer.append(tick_record)

        # Flush buffer every 5 seconds or when > 50 ticks
        if len(self._tick_buffer) >= 50 or (time.time() - self._last_tick_flush) >= 5.0:
            self.flush_tick_buffer(date_str)

    def flush_tick_buffer(self, date_str: Optional[str] = None):
        """Flushes in-memory tick buffer to daily JSONL file."""
        if not self._tick_buffer:
            return

        if date_str is None:
            date_str = datetime.now(IST).strftime("%Y-%m-%d")

        tick_file = self.live_dir / f"ticks_{date_str}.jsonl"
        try:
            with open(tick_file, "a", encoding="utf-8") as f:
                for item in self._tick_buffer:
                    f.write(json.dumps(item) + "\n")
            self._tick_buffer.clear()
            self._last_tick_flush = time.time()
        except Exception as e:
            logger.error(f"Failed to flush live ticks to {tick_file}: {e}")

    def save_market_snapshot(self, snapshot_data: Dict[str, Any]):
        """
        Saves full market snapshot to local disk for persistence and audit replay.
        """
        if not snapshot_data:
            return

        now_ist = datetime.now(IST)
        date_str = now_ist.strftime("%Y-%m-%d")

        # 1. Update latest snapshot
        latest_file = self.snapshots_dir / "latest_snapshot.json"
        try:
            with open(latest_file, "w", encoding="utf-8") as f:
                json.dump(snapshot_data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to write latest snapshot: {e}")

        # 2. Append to daily snapshot stream (hourly / periodic)
        daily_snap_file = self.snapshots_dir / f"snapshot_{date_str}.jsonl"
        try:
            with open(daily_snap_file, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "timestamp": now_ist.isoformat(),
                    "epoch": snapshot_data.get("timestamp", time.time()),
                    "indices": snapshot_data.get("indices", []),
                    "options_signals": snapshot_data.get("options_signals", []),
                    "commodities": snapshot_data.get("commodities", []),
                }, default=str) + "\n")
        except Exception as e:
            logger.error(f"Failed to append daily snapshot: {e}")

    def load_latest_snapshot(self) -> Optional[Dict[str, Any]]:
        """Loads latest saved market snapshot."""
        latest_file = self.snapshots_dir / "latest_snapshot.json"
        if not latest_file.exists():
            return None
        try:
            with open(latest_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load latest snapshot: {e}")
            return None

    def get_storage_status(self) -> Dict[str, Any]:
        """
        Returns a comprehensive report of all locally stored datasets,
        including symbols, intervals, candle counts, time spans, and disk usage.
        """
        files_info = []
        total_candles = 0
        total_bytes = 0

        # 1. Inspect historical files
        if self.historical_dir.exists():
            for f in self.historical_dir.glob("*.csv"):
                try:
                    size = f.stat().st_size
                    total_bytes += size
                    # Parse symbol and interval from filename: {symbol}_{interval}.csv
                    stem = f.stem
                    parts = stem.rsplit("_", 1)
                    if len(parts) == 2:
                        sym, interval = parts[0], parts[1]
                    else:
                        sym, interval = stem, "unknown"

                    df = pd.read_csv(f, index_col=0, nrows=2)
                    total_rows = sum(1 for _ in open(f, "rb")) - 1 if size > 0 else 0
                    total_candles += max(0, total_rows)

                    files_info.append({
                        "file": f.name,
                        "symbol": sym,
                        "interval": interval,
                        "size_kb": round(size / 1024, 2),
                        "candle_count": max(0, total_rows),
                    })
                except Exception as e:
                    logger.debug(f"Error reading file status for {f}: {e}")

        # 2. Inspect live tick files
        tick_files_count = 0
        if self.live_dir.exists():
            for f in self.live_dir.glob("*.jsonl"):
                size = f.stat().st_size
                total_bytes += size
                tick_files_count += 1

        # 3. Inspect snapshot files
        snapshot_files_count = 0
        if self.snapshots_dir.exists():
            for f in self.snapshots_dir.glob("*.*"):
                size = f.stat().st_size
                total_bytes += size
                snapshot_files_count += 1

        return {
            "status": "ACTIVE",
            "base_directory": str(self.base_dir),
            "total_stored_files": len(files_info),
            "total_historical_candles": total_candles,
            "total_disk_usage_mb": round(total_bytes / (1024 * 1024), 2),
            "live_tick_files_count": tick_files_count,
            "snapshot_files_count": snapshot_files_count,
            "datasets": sorted(files_info, key=lambda x: x["symbol"]),
            "timestamp": datetime.now(IST).isoformat(),
        }

    def sync_all_tracked_symbols(self, market_data_service, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Synchronizes historical and current candles for all popular Indices,
        MCX Commodities, and Watchlist Equities across 15m, 1d, and 1wk intervals.
        """
        tracked_symbols = [
            "^NSEI", "^BSESN", "^NSEBANK", "NIFTY_FIN_SERVICE.NS", "^NSEMDCP50",
            "CRUDEOIL", "NATURALGAS", "GOLD", "SILVER", "COPPER",
            "RELIANCE", "TCS", "HDFCBANK", "SBIN", "INFY", "ICICIBANK", "LT", "ITC"
        ]
        intervals_map = [
            ("15m", "5d"),
            ("1d", "3mo"),
            ("1wk", "1y"),
        ]

        synced_results = []
        for sym in tracked_symbols:
            canon = market_data_service.normalize_symbol(sym)
            for interval, period in intervals_map:
                try:
                    df = market_data_service.get_historical_candles(canon, period=period, interval=interval)
                    if df is not None and not df.empty:
                        merged_df = self.merge_and_save_candles(canon, interval, df)
                        file_p = str(self._get_candle_filepath(canon, interval, ext="csv"))
                        synced_results.append({
                            "symbol": canon,
                            "interval": interval,
                            "period": period,
                            "candles_saved": int(len(merged_df)),
                            "file_path": file_p,
                            "status": "SUCCESS"
                        })
                    else:
                        synced_results.append({
                            "symbol": canon,
                            "interval": interval,
                            "status": "EMPTY"
                        })
                except Exception as e:
                    synced_results.append({
                        "symbol": canon,
                        "interval": interval,
                        "status": f"ERROR: {str(e)}"
                    })

        return {
            "synced_count": int(len([r for r in synced_results if r.get("status") == "SUCCESS"])),
            "total_attempted": int(len(synced_results)),
            "details": synced_results,
            "timestamp": datetime.now(IST).isoformat(),
        }


# Global singleton instance
local_data_service = LocalDataService()
