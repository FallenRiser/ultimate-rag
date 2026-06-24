import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.routes import chat, documents, health, ingestion, query
from app.observability.logging import setup_logging
from app.observability.tracing import setup_tracing
from app.repositories.relational.database import get_engine
from app.repositories.relational.schema import create_tables
from app.repositories.vector.factory import create_vector_db
from app.utils.config import get_settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    setup_tracing()
    settings = get_settings()
    logger.info("Starting %s (%s)", settings.app.name, settings.app.env)
    await create_tables(get_engine())
    # Ensure the vector collection/table exists before any ingestion runs.
    await create_vector_db().initialize_collection(settings.vector_store.embedding_dim)
    yield
    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app.name,
        debug=settings.app.debug,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    prefix = "/api/v1"
    app.include_router(health.router, prefix=prefix)
    app.include_router(documents.router, prefix=prefix)
    app.include_router(ingestion.router, prefix=prefix)
    app.include_router(query.router, prefix=prefix)
    app.include_router(chat.router, prefix=prefix)

    return app


app = create_app()
