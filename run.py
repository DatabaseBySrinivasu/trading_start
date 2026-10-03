import sys
import uvicorn
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.core.config import settings
from app.core.logging_config import setup_logging

if __name__ == "__main__":
    setup_logging()
    port = settings.PORT or 8000
    host = "127.0.0.1"
    print("\n" + "=" * 60)
    print("  Starting Trading Application...")
    print(f"  Live Dashboard : http://{host}:{port}")
    print(f"  API Docs (Swagger) : http://{host}:{port}/docs")
    print("=" * 60 + "\n")
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=settings.DEBUG,
    )
