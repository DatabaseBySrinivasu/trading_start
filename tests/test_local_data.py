import os
import json
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from app.services.local_data_service import LocalDataService, local_data_service
from app.services.market_data import market_data_service


class TestLocalDataPersistenceEngine:
    """
    Comprehensive test suite for LocalDataService:
    1. Multi-timeframe historical candle disk persistence (CSV).
    2. Incremental candle merging without duplicate timestamps.
    3. Real-time live tick archiving (JSONL).
    4. Market snapshot persistence and replay.
    5. Local storage status and metadata reporting.
    6. Background synchronization across tracked symbols.
    7. REST API endpoints for local data inspection and synchronization.
    """

    @pytest.fixture
    def custom_service(self, tmp_path):
        """Returns an isolated LocalDataService instance rooted in a temp directory."""
        return LocalDataService(base_dir=tmp_path / "test_data")

    @pytest.fixture
    def sample_df(self):
        """Generates 20 sample candles."""
        dates = pd.date_range("2026-09-01", periods=20, freq="15min")
        np.random.seed(42)
        base = 24500.0
        closes = base + np.cumsum(np.random.normal(0, 10, 20))
        return pd.DataFrame({
            "open": closes - 5,
            "high": closes + 10,
            "low": closes - 10,
            "close": closes,
            "volume": [10000 + i * 500 for i in range(20)],
        }, index=dates)

    def test_save_and_load_candles(self, custom_service, sample_df):
        # Save candles
        path_str = custom_service.save_candles("^NSEI", "15m", sample_df)
        assert path_str != ""
        assert os.path.exists(path_str)

        # Load candles back
        loaded = custom_service.load_candles("^NSEI", "15m")
        assert loaded is not None
        assert not loaded.empty
        assert len(loaded) == 20
        assert list(loaded.columns) == ["open", "high", "low", "close", "volume"]
        assert loaded["close"].iloc[0] == pytest.approx(sample_df["close"].iloc[0], rel=1e-3)

    def test_merge_and_save_candles_deduplication(self, custom_service, sample_df):
        # Save first batch of 15 candles
        batch1 = sample_df.iloc[:15]
        custom_service.save_candles("^NSEI", "15m", batch1)

        # Create overlapping second batch (candles 10 to 20 with updated volume on candle 10)
        batch2 = sample_df.iloc[10:].copy()
        batch2.loc[batch2.index[0], "volume"] = 999999

        merged = custom_service.merge_and_save_candles("^NSEI", "15m", batch2)

        # Result should be exactly 20 candles, chronological, with updated volume
        assert len(merged) == 20
        assert merged.index.is_monotonic_increasing
        assert merged.loc[sample_df.index[10], "volume"] == 999999

    def test_log_live_ticks_and_flush(self, custom_service):
        for i in range(5):
            custom_service.log_live_tick("^NSEI", {
                "price": 24500.0 + i,
                "change": 10.0 + i,
                "change_percent": 0.05,
                "source": "ANGEL_ONE",
                "tick_direction": "UP",
            })

        assert len(custom_service._tick_buffer) == 5
        custom_service.flush_tick_buffer(date_str="2026-10-03")
        assert len(custom_service._tick_buffer) == 0

        # Verify JSONL content on disk
        tick_file = custom_service.live_dir / "ticks_2026-10-03.jsonl"
        assert tick_file.exists()
        lines = tick_file.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 5
        record = json.loads(lines[0])
        assert record["symbol"] == "^NSEI"
        assert record["price"] == 24500.0

    def test_save_and_load_market_snapshot(self, custom_service):
        snap_data = {
            "timestamp": 1727900000.0,
            "indices": [{"name": "NIFTY 50", "price": 24550.0}],
            "options_signals": [{"symbol": "^NSEI", "recommendation": "BUY CALL"}],
            "commodities": [{"symbol": "CRUDEOIL", "price": 6150.0}],
        }

        custom_service.save_market_snapshot(snap_data)

        # Load latest snapshot
        loaded = custom_service.load_latest_snapshot()
        assert loaded is not None
        assert loaded["timestamp"] == snap_data["timestamp"]
        assert len(loaded["indices"]) == 1
        assert loaded["indices"][0]["name"] == "NIFTY 50"

    def test_storage_status_reporting(self, custom_service, sample_df):
        custom_service.save_candles("^NSEI", "15m", sample_df)
        custom_service.save_candles("CRUDEOIL", "1d", sample_df)
        custom_service.save_market_snapshot({"timestamp": 12345})

        status = custom_service.get_storage_status()
        assert status["status"] == "ACTIVE"
        assert status["total_stored_files"] == 2
        assert status["total_historical_candles"] == 40
        assert status["total_disk_usage_mb"] >= 0.0
        assert len(status["datasets"]) == 2

    def test_sync_all_tracked_symbols(self, custom_service):
        res = custom_service.sync_all_tracked_symbols(market_data_service)
        assert "synced_count" in res
        assert "total_attempted" in res
        assert res["total_attempted"] > 0
        assert res["synced_count"] > 0

    def test_rest_api_local_data_endpoints(self, client):
        # 1. GET /api/v1/market/local-data/status
        res_status = client.get("/api/v1/market/local-data/status")
        assert res_status.status_code == 200
        data_status = res_status.json()
        assert data_status["status"] == "ACTIVE"
        assert "total_stored_files" in data_status
        assert "total_historical_candles" in data_status

        # 2. GET /api/v1/market/local-data/candles/{symbol}
        res_candles = client.get("/api/v1/market/local-data/candles/^NSEI?interval=15m")
        assert res_candles.status_code == 200
        data_candles = res_candles.json()
        assert data_candles["symbol"] == "^NSEI"
        assert data_candles["total_candles"] > 0
        assert len(data_candles["recent_candles"]) > 0

        # 3. GET /api/v1/market/local-data/snapshots/latest
        res_snap = client.get("/api/v1/market/local-data/snapshots/latest")
        assert res_snap.status_code == 200
        data_snap = res_snap.json()
        assert "indices" in data_snap or "timestamp" in data_snap

        # 4. POST /api/v1/market/local-data/sync
        res_sync = client.post("/api/v1/market/local-data/sync")
        assert res_sync.status_code == 200
        data_sync = res_sync.json()
        assert "synced_count" in data_sync
