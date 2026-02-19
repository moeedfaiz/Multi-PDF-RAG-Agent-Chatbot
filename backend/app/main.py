from fastapi import FastAPI
from .config import settings
from .services.mlflow_logger import setup_mlflow
from .api.routes.chat import router as chat_router
from .api.routes.chat_stream import router as chat_stream_router
from .api.routes.debug import router as debug_router

from .api.routes.admin import router as admin_router
from .api.routes.health import router as health_router
from .api.routes.upload import router as upload_router
from .api.routes.docs import router as docs_router
from .api.routes.ingest import router as ingest_router
import logging
logging.basicConfig(level=logging.INFO)


def create_app() -> FastAPI:
    app = FastAPI(title="PDF RAG API", version="0.3.0")

    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    settings.parsed_dir.mkdir(parents=True, exist_ok=True)

    setup_mlflow(settings.mlflow_tracking_uri)

    app.include_router(health_router, tags=["health"])
    app.include_router(upload_router, tags=["pdf"])
    app.include_router(docs_router, tags=["pdf"])
    app.include_router(ingest_router, tags=["rag"])
    app.include_router(chat_router, tags=["chat"])
    app.include_router(chat_stream_router, tags=["chat"])
    app.include_router(debug_router, tags=["debug"])
    app.include_router(admin_router, tags=["admin"])

    return app

app = create_app()
