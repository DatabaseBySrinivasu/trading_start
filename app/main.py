import os
import sys
from pathlib import Path

# Ensure project root is in sys.path when running app/main.py directly
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from app.core.config import settings
from app.core.logging_config import setup_logging
from app.api.v1.api import api_router

# Initialize system file & console logging
setup_logging()

TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "templates", "index.html")


def create_application() -> FastAPI:
    app = FastAPI(
        title=settings.PROJECT_NAME,
        openapi_url=f"{settings.API_V1_STR}/openapi.json",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Set up CORS middleware
    if settings.BACKEND_CORS_ORIGINS:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Include API Routers
    app.include_router(api_router, prefix=settings.API_V1_STR)

    @app.get("/", response_class=HTMLResponse, tags=["dashboard"])
    def dashboard():
        """Serve interactive NSE & BSE Live Market Dashboard."""
        if os.path.exists(TEMPLATE_PATH):
            with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
                return f.read()
        return "<h1>Dashboard template not found</h1>"

    @app.get("/health", tags=["health"])
    def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "version": "0.1.0"
        }

    return app


app = create_application()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )

