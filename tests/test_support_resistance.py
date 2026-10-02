import pytest
import pandas as pd
import numpy as np
from app.services.pattern_service import pattern_service
from app.services.strategy_engine import strategy_engine
from app.services.market_data import market_data_service


class TestMultiTimeframeSupportAndResistance:
    """
    Test suite for:
    1. Multi-Timeframe Historical Data Ingestion & Extreme Levels (PDH/PDL, 2DH-20DH, PWH/PWL, PMH/PML)
    2. Daily & Weekly Central Pivot Range (CPR)
    3. Daily & Weekly Classical Pivots (R1-R4, S1-S4)
    4. Camarilla Pivot Points Engine (H1-H5, L1-L5)
    5. Structural Swing High/Low Fractal Clusters (Demand/Supply Zones)
    6. Multi-Day Macro Fibonacci Retracements & Golden Pocket
    7. Consolidated Resistance & Support Hierarchy with Confluence Tagging
    8. Strategy Engine S/R Alignment (Dynamic Targets & Stop Loss Placement)
    9. REST API Endpoints
    """

    @pytest.fixture
    def sample_daily_df(self):
        """Creates 30 days of realistic daily candle data."""
        np.random.seed(42)
        n = 30
        base_price = 24500.0
        prices = [base_price]
        for _ in range(n - 1):
            prices.append(prices[-1] + np.random.uniform(-100, 120))

        dates = pd.date_range(end=pd.Timestamp.now(), periods=n, freq="D")
        highs = [p + np.random.uniform(50, 150) for p in prices]
        lows = [p - np.random.uniform(50, 150) for p in prices]
        opens = [p - np.random.uniform(-30, 30) for p in prices]
        closes = prices
        volumes = [int(np.random.uniform(50000, 200000)) for _ in range(n)]

        return pd.DataFrame({
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }, index=dates)

    def test_camarilla_pivots_calculation(self):
        high, low, close = 25000.0, 24600.0, 24800.0
        cam = pattern_service.calculate_camarilla_pivots(high, low, close, cur_price=24800.0)

        assert "h5" in cam and "h4" in cam and "h3" in cam and "h2" in cam and "h1" in cam
        assert "l1" in cam and "l2" in cam and "l3" in cam and "l4" in cam and "l5" in cam
        assert "status" in cam
        assert "bias" in cam

        # Mathematical relationships
        assert cam["h5"] > cam["h4"] > cam["h3"] > cam["h2"] > cam["h1"]
        assert cam["l1"] > cam["l2"] > cam["l3"] > cam["l4"] > cam["l5"]
        assert cam["h3"] > close > cam["l3"]

        # H4 Breakout Test
        cam_breakout = pattern_service.calculate_camarilla_pivots(high, low, close, cur_price=25100.0)
        assert cam_breakout["status"] == "BULLISH_H4_BREAKOUT"

        # L4 Breakdown Test
        cam_breakdown = pattern_service.calculate_camarilla_pivots(high, low, close, cur_price=24500.0)
        assert cam_breakdown["status"] == "BEARISH_L4_BREAKDOWN"

    def test_swing_clusters_detection(self, sample_daily_df):
        clusters = pattern_service.calculate_swing_clusters(sample_daily_df, lookback=25, cluster_tolerance_pct=0.35)
        assert "supply_clusters" in clusters
        assert "demand_clusters" in clusters
        assert "total_clusters" in clusters
        assert isinstance(clusters["supply_clusters"], list)
        assert isinstance(clusters["demand_clusters"], list)

        if clusters["supply_clusters"]:
            sc = clusters["supply_clusters"][0]
            assert "price" in sc
            assert "touches" in sc
            assert "strength" in sc

    def test_macro_fibonacci(self, sample_daily_df):
        fib = pattern_service.calculate_macro_fibonacci(sample_daily_df, lookback=25)
        assert "levels" in fib
        assert "swing_high" in fib
        assert "swing_low" in fib
        assert "nearest_support" in fib
        assert "nearest_resistance" in fib
        assert "golden_pocket_support" in fib

        levels = fib["levels"]
        assert "fib_0.0" in levels
        assert "fib_23.6" in levels
        assert "fib_38.2" in levels
        assert "fib_50.0" in levels
        assert "fib_61.8" in levels
        assert "fib_78.6" in levels
        assert "fib_100.0" in levels
        assert "fib_161.8" in levels

        assert levels["fib_0.0"] >= levels["fib_61.8"] >= levels["fib_100.0"]

    def test_day_levels_enrichment(self, sample_daily_df):
        dl = pattern_service.calculate_day_levels(sample_daily_df)
        assert "pdh" in dl
        assert "pdl" in dl
        assert "pdc" in dl
        assert "cpr" in dl
        assert "classical_pivots" in dl
        assert "camarilla_pivots" in dl

        pivots = dl["classical_pivots"]
        assert "r1" in pivots and "r2" in pivots and "r3" in pivots and "r4" in pivots
        assert "s1" in pivots and "s2" in pivots and "s3" in pivots and "s4" in pivots
        assert pivots["r4"] > pivots["r3"] > pivots["r2"] > pivots["r1"]
        assert pivots["s1"] > pivots["s2"] > pivots["s3"] > pivots["s4"]

    def test_unified_multi_timeframe_sr_engine(self, sample_daily_df):
        sr = pattern_service.calculate_multi_timeframe_sr(
            df_intraday=sample_daily_df,
            df_daily=sample_daily_df,
            df_weekly=sample_daily_df,
            symbol="^NSEI",
        )

        assert sr["symbol"] == "^NSEI"
        assert "current_price" in sr
        assert "nearest_resistance" in sr
        assert "nearest_support" in sr
        assert "r_near" in sr
        assert "r_mid" in sr
        assert "r_major" in sr
        assert "s_near" in sr
        assert "s_mid" in sr
        assert "s_major" in sr

        # S/R Hierarchy
        assert sr["r_major"] >= sr["r_mid"] >= sr["r_near"] > sr["current_price"]
        assert sr["current_price"] > sr["s_near"] >= sr["s_mid"] >= sr["s_major"]

        # Multi-day extremes
        extremes = sr["multi_day_extremes"]
        assert "pdh" in extremes and "pdl" in extremes
        assert "high_2d" in extremes and "low_2d" in extremes
        assert "high_5d" in extremes and "low_5d" in extremes
        assert "high_20d" in extremes and "low_20d" in extremes

        # CPR
        assert "daily_cpr" in sr
        assert "weekly_cpr" in sr
        assert sr["daily_cpr"]["pivot"] > 0
        assert sr["weekly_cpr"]["pivot"] > 0

        # Confluence & description
        assert "resistance_hierarchy" in sr
        assert "support_hierarchy" in sr
        assert "confluence_bias" in sr
        assert "description" in sr

    def test_strategy_engine_incorporates_multi_timeframe_sr(self):
        signal = strategy_engine.generate_options_call_put_signal(symbol="^NSEI")
        assert "multi_timeframe_sr" in signal
        sr = signal["multi_timeframe_sr"]
        assert "r_near" in sr
        assert "s_near" in sr
        assert "nearest_resistance" in sr
        assert "nearest_support" in sr

        # Verify reasons include S/R confluence
        reasons_text = " ".join(signal.get("confluence_reasons", []))
        assert "Historical S/R" in reasons_text or "CPR" in reasons_text or "PDH" in reasons_text or "Support" in reasons_text

        # Target and stop loss validity
        assert signal["spot_target_1"] > 0
        assert signal["spot_stop_loss"] > 0
        assert signal["option_target_1"] > 0
        assert signal["option_stop_loss"] > 0

    def test_rest_api_support_resistance_endpoints(self, client):
        # 1. /api/v1/market/support-resistance/{symbol}
        res1 = client.get("/api/v1/market/support-resistance/^NSEI")
        assert res1.status_code == 200
        data1 = res1.json()
        assert "r_near" in data1
        assert "s_near" in data1
        assert "multi_day_extremes" in data1
        assert "resistance_hierarchy" in data1
        assert "support_hierarchy" in data1

        # 2. /api/v1/market/camarilla/{symbol}
        res2 = client.get("/api/v1/market/camarilla/^NSEI")
        assert res2.status_code == 200
        data2 = res2.json()
        assert "h4" in data2
        assert "l4" in data2
        assert "status" in data2

        # 3. /api/v1/market/fibonacci/{symbol}
        res3 = client.get("/api/v1/market/fibonacci/^NSEI")
        assert res3.status_code == 200
        data3 = res3.json()
        assert "levels" in data3
        assert "golden_pocket" in data3 or "golden_pocket_support" in data3

        # 4. /api/v1/market/macro-fibonacci/{symbol}
        res3_macro = client.get("/api/v1/market/macro-fibonacci/^NSEI")
        assert res3_macro.status_code == 200
        data3_macro = res3_macro.json()
        assert "levels" in data3_macro
        assert "golden_pocket_support" in data3_macro

        # 5. /api/v1/market/daylevels/{symbol}
        res4 = client.get("/api/v1/market/daylevels/^NSEI")
        assert res4.status_code == 200
        data4 = res4.json()
        assert "classical_pivots" in data4 or "pivots" in data4
        assert "camarilla_pivots" in data4 or "camarilla" in data4
