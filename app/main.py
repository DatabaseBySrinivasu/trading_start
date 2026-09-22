from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.v1.api import api_router


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

    @app.get("/", tags=["root"])
    def root():
        """Root endpoint returning welcome message."""
        return {
            "message": f"Welcome to {settings.PROJECT_NAME}!",
            "docs": "/docs",
            "api_version": "v1"
        }

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
