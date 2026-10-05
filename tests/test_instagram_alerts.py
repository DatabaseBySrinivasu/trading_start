import pytest
from app.services.instagram_service import instagram_service
from app.services.strategy_engine import strategy_engine


def test_instagram_service_status(client):
    """Verify Instagram service status endpoint and default recipient 9100040008."""
    res = client.get("/api/v1/instagram/status")
    assert res.status_code == 200
    data = res.json()
    assert "target_recipient" in data
    assert data["target_recipient"] == "9100040008"
    assert "api_status" in data


def test_instagram_test_ping(client):
    """Verify Instagram test ping delivery to 9100040008."""
    res = client.post("/api/v1/instagram/test?recipient_id=9100040008")
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["recipient"] == "9100040008"
    assert "preview_text" in data or "preview" in data


def test_instagram_send_signal_alert(client):
    """Verify generation and dispatch of live PUT/CALL option strike signal to Instagram."""
    # Test with NIFTY
    res = client.post("/api/v1/instagram/send-signal/^NSEI?recipient_id=9100040008")
    assert res.status_code == 200
    data = res.json()
    assert "delivery_result" in data
    assert data["delivery_result"]["success"] is True
    assert "signal_sent" in data
    assert data["signal_sent"]["option_entry_price"] > 0
    assert data["signal_sent"]["suggested_strike"] is not None
    assert "CALL" in data["signal_sent"]["recommendation"] or "PUT" in data["signal_sent"]["recommendation"] or "NEUTRAL" in data["signal_sent"]["recommendation"]


def test_instagram_broadcast_all(client):
    """Verify multi-market broadcast of PUT/CALL option alerts to Instagram (9100040008)."""
    res = client.post("/api/v1/instagram/broadcast-all?recipient_id=9100040008&include_commodities=true")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "COMPLETED"
    assert data["recipient"] == "9100040008"
    assert data["total_scanned"] == 8  # 3 indices + 5 commodities
    assert "dispatched" in data


def test_instagram_custom_message(client):
    """Verify custom message endpoint."""
    res = client.post(
        "/api/v1/instagram/send-custom",
        json={"message": "URGENT: NIFTY 23400 CE BREAKOUT ALERT", "recipient_id": "9100040008"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["recipient"] == "9100040008"
