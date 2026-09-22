# Trading Start 📈

A modern, production-ready REST API for trading execution, signal monitoring, and portfolio tracking built with **FastAPI**, **Pydantic v2**, and **Uvicorn**.

---

## 🌟 Features

- ⚡ **Fast & Asynchronous**: High-throughput order/trade processing powered by FastAPI & Uvicorn.
- 📊 **Trading Schema & Validation**: Pre-built models for orders (`BUY`/`SELL`), order types (`MARKET`, `LIMIT`, `STOP`), and statuses (`PENDING`, `FILLED`, `CANCELLED`).
- 🔍 **Filtering & Pagination**: Query trades by ticker symbol, order status, and page offset.
- 🧪 **Automated Test Suite**: Unit and integration testing with `pytest` and `httpx`.
- 🔄 **GitHub Actions CI/CD**: Automatic matrix testing across Python 3.10, 3.11, and 3.12.
- 🐳 **Containerized Deployment**: Ready-to-use `Dockerfile` and `.dockerignore`.
- 📖 **Interactive API Documentation**: Auto-generated Swagger UI (`/docs`) and ReDoc (`/redoc`).

---

## 📁 Project Structure

```
.
├── .github/
│   └── workflows/
│       └── ci.yml                  # GitHub Actions CI pipeline
├── app/
│   ├── __init__.py
│   ├── main.py                     # FastAPI app factory & routes
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py               # Pydantic BaseSettings config
│   ├── models/
│   │   ├── __init__.py
│   │   └── trade.py                # Pydantic models (Trade, Enums, Schemas)
│   └── api/
│       ├── __init__.py
│       └── v1/
│           ├── __init__.py
│           ├── api.py              # Router aggregator
│           └── endpoints/
│               ├── __init__.py
│               └── trades.py       # Trade CRUD endpoints
├── tests/
│   ├── __init__.py
│   ├── conftest.py                 # TestClient fixture
│   ├── test_main.py                # Health check & root endpoint tests
│   └── test_trades.py              # Trading API endpoints tests
├── .dockerignore
├── .env.example
├── .gitignore
├── Dockerfile
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.10+
- Git

### 2. Set Up Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements-dev.txt
```

### 4. Run the Development Server

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open your browser:
- **Interactive Swagger Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc Documentation**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
- **Health Check**: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

## 🧪 Running Tests

Run the test suite with pytest:

```bash
pytest -v
```

---

## 🛠️ API Endpoints Summary

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | API welcome & info |
| `GET` | `/health` | Application health check |
| `GET` | `/api/v1/trades/` | List trades (filter by `symbol`, `status`, `skip`, `limit`) |
| `POST` | `/api/v1/trades/` | Create a new trade or order |
| `GET` | `/api/v1/trades/{id}` | Get specific trade details |
| `PUT` | `/api/v1/trades/{id}` | Update trade status, price, or notes |
| `DELETE`| `/api/v1/trades/{id}` | Cancel/delete trade |

---

## 🐳 Docker

```bash
docker build -t trading_start .
docker run -d -p 8000:8000 --name trading-api trading_start
```

---

## 📄 License

This project is licensed under the MIT License.
