import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import chat
from app.routes import qdrant
from app.routes import training
from app.routes import collection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────
    logger.info("=== Service Preloading Started ===")

    logger.info("Loading embedding model...")
    from app.services.embedding_service import get_model
    get_model()
    logger.info("Embedding model loaded successfully.")

    logger.info("Connecting to Qdrant...")
    from app.services.qdrant_service import get_client
    get_client()
    logger.info("Qdrant client connected successfully.")

    logger.info("=== Service Preloading Complete — Ready to serve ===")

    yield
    # ── Shutdown ─────────────────────────────────────────────
    logger.info("Shutting down TalkToDB API.")


app = FastAPI(
    title="TalkToDB API",
    description="Backend API for TalkToDB",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(chat.router)
app.include_router(qdrant.router)
app.include_router(training.router)
app.include_router(collection.router)


@app.get("/")
async def root():
    return {"message": "Welcome to TalkToDB API"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
