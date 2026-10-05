import pytest
from app.services.volatility_service import volatility_service
from app.services.market_data import market_data_service


class TestInternationalIndicesAndGiftNifty:
    """
    Test suite for:
    1. Multi-Source GIFT NIFTY Engine (Direct Ticker + Econometric Synthesis)
    2. International Indices Cockpit (US, Asia, Europe, Macro Commodities, Offshore India ETFs)
    3. Sector-Specific Impact & FII Flow Predictions
    4. REST API Endpoints
    """

    def test_international_indices_cockpit_structure(self):
        cockpit = volatility_service.get_international_indices_cockpit()
        assert isinstance(cockpit, dict)
        assert "sentiment_score" in cockpit
        assert "sentiment_bias" in cockpit
        assert "sentiment_label" in cockpit
        assert "fii_flow_bias" in cockpit
        assert "fii_description" in cockpit
        assert "regional_averages" in cockpit
        assert "us_markets" in cockpit
        assert "asian_markets" in cockpit
        assert "european_markets" in cockpit
        assert "macro_commodities" in cockpit
        assert "offshore_india_etfs" in cockpit
        assert "sector_impacts" in cockpit

        # Validate US Markets tickers
        us_symbols = [m["symbol"] for m in cockpit["us_markets"]]
        assert "ES=F" in us_symbols or "^GSPC" in us_symbols
        assert "NQ=F" in us_symbols or "^IXIC" in us_symbols

        # Validate Asian Markets tickers
        asian_symbols = [m["symbol"] for m in cockpit["asian_markets"]]
        assert "^N225" in asian_symbols
        assert "^HSI" in asian_symbols

        # Validate European Markets tickers
        euro_symbols = [m["symbol"] for m in cockpit["european_markets"]]
        assert "^FTSE" in euro_symbols or "^GDAXI" in euro_symbols

        # Validate Macro commodities & currencies
        macro_symbols = [m["symbol"] for m in cockpit["macro_commodities"]]
        assert "BZ=F" in macro_symbols
        assert "USDINR=X" in macro_symbols

        # Validate Offshore India ETFs
        etf_symbols = [m["symbol"] for m in cockpit["offshore_india_etfs"]]
        assert "INDA" in etf_symbols or "EPI" in etf_symbols

    def test_sector_impacts(self):
        cockpit = volatility_service.get_international_indices_cockpit()
        sectors = cockpit["sector_impacts"]
        assert "it_sector" in sectors
        assert "banking_sector" in sectors
        assert "auto_and_energy" in sectors

        assert sectors["it_sector"]["bias"] in ["BULLISH", "BEARISH", "NEUTRAL"]
        assert "TCS" in sectors["it_sector"]["impacted_stocks"]

        assert sectors["banking_sector"]["bias"] in ["BULLISH", "BEARISH", "NEUTRAL"]
        assert "HDFCBANK" in sectors["banking_sector"]["impacted_stocks"]

        assert sectors["auto_and_energy"]["bias"] in ["BULLISH", "BEARISH", "NEUTRAL"]
        assert "MARUTI" in sectors["auto_and_energy"]["impacted_stocks"]

    def test_multi_source_gift_nifty(self):
        multi = volatility_service.get_gift_nifty_multi_source()
        assert "selected_price" in multi
        assert "selected_source" in multi
        assert "nifty_spot_price" in multi
        assert "projected_gap_pts" in multi
        assert "projected_gap_pct" in multi
        assert "opening_bias" in multi
        assert "direct_feed" in multi
        assert "synthetic_feed" in multi
        assert "international_cockpit" in multi

        assert multi["selected_price"] > 10000.0
        assert multi["nifty_spot_price"] > 10000.0

        synth = multi["synthetic_feed"]
        assert "basis_carry_pct" in synth
        assert "overnight_momentum_pct" in synth
        assert "drivers" in synth

    def test_gift_nifty_and_global_cues_interface(self):
        gift_info = volatility_service.get_gift_nifty_and_global_cues()
        assert "gift_nifty_price" in gift_info
        assert "nifty_spot_price" in gift_info
        assert "projected_gap_pts" in gift_info
        assert "projected_gap_pct" in gift_info
        assert "opening_bias" in gift_info
        assert "global_markets" in gift_info
        assert "international_cockpit" in gift_info
        assert "multi_source" in gift_info

    def test_api_endpoints(self, client):
        # 1. /api/v1/market/giftnifty
        res1 = client.get("/api/v1/market/giftnifty")
        assert res1.status_code == 200
        data1 = res1.json()
        assert "gift_nifty_price" in data1
        assert "international_cockpit" in data1

        # 2. /api/v1/market/giftnifty/multi-source
        res2 = client.get("/api/v1/market/giftnifty/multi-source")
        assert res2.status_code == 200
        data2 = res2.json()
        assert "direct_feed" in data2
        assert "synthetic_feed" in data2

        # 3. /api/v1/market/international-cockpit
        res3 = client.get("/api/v1/market/international-cockpit")
        assert res3.status_code == 200
        data3 = res3.json()
        assert "us_markets" in data3
        assert "asian_markets" in data3
        assert "sector_impacts" in data3

    def test_live_dashboard_snapshot_contains_enriched_gift_nifty(self, client):
        res = client.get("/api/v1/market/live-tick")
        assert res.status_code == 200
        snap = res.json()
        assert "gift_nifty" in snap
        assert "international_cockpit" in snap["gift_nifty"]
