def test_list_trades(client):
    response = client.get("/api/v1/trades/")
    assert response.status_code == 200
    trades = response.json()
    assert isinstance(trades, list)
    assert len(trades) >= 2


def test_filter_trades_by_symbol(client):
    response = client.get("/api/v1/trades/?symbol=AAPL")
    assert response.status_code == 200
    trades = response.json()
    assert len(trades) >= 1
    assert all(t["symbol"] == "AAPL" for t in trades)


def test_get_trade_by_id(client):
    response = client.get("/api/v1/trades/1")
    assert response.status_code == 200
    trade = response.json()
    assert trade["id"] == 1
    assert trade["symbol"] == "AAPL"
    assert trade["side"] == "BUY"


def test_get_trade_not_found(client):
    response = client.get("/api/v1/trades/99999")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_create_trade(client):
    new_trade_payload = {
        "symbol": "TSLA",
        "side": "BUY",
        "order_type": "LIMIT",
        "quantity": 15.0,
        "price": 240.50,
        "notes": "Testing trade creation"
    }
    response = client.post("/api/v1/trades/", json=new_trade_payload)
    assert response.status_code == 201
    created = response.json()
    assert created["id"] > 2
    assert created["symbol"] == "TSLA"
    assert created["quantity"] == 15.0
    assert created["price"] == 240.50
    assert created["status"] == "PENDING"


def test_update_trade(client):
    update_payload = {
        "status": "FILLED",
        "notes": "Order successfully filled at market open"
    }
    response = client.put("/api/v1/trades/2", json=update_payload)
    assert response.status_code == 200
    updated = response.json()
    assert updated["id"] == 2
    assert updated["status"] == "FILLED"
    assert updated["notes"] == "Order successfully filled at market open"


def test_delete_trade(client):
    # First create a trade
    create_res = client.post(
        "/api/v1/trades/",
        json={"symbol": "MSFT", "side": "SELL", "quantity": 10.0, "notes": "Temporary order"}
    )
    assert create_res.status_code == 201
    trade_id = create_res.json()["id"]

    # Delete
    del_res = client.delete(f"/api/v1/trades/{trade_id}")
    assert del_res.status_code == 204

    # Verify not found
    get_res = client.get(f"/api/v1/trades/{trade_id}")
    assert get_res.status_code == 404
