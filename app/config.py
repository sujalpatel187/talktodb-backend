import logging
import sys
from pydantic_settings import BaseSettings
from pydantic import ValidationError
from functools import lru_cache

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    # App
    debug: bool = False

    # Qdrant — REQUIRED
    qdrant_url: str
    qdrant_collection_name: str
    qdrant_ddl_collection_name: str
    qdrant_docs_collection_name: str
    qdrant_api_key: str | None = None

    # Embeddings — REQUIRED
    embedding_model: str
    embedding_dimension: int

    # LLM — REQUIRED
    ollama_url: str
    model_name: str
    llm_api_key: str | None = None
    temperature: float
    max_tokens: int

    # PostgreSQL — REQUIRED for /execute
    pg_host: str
    pg_port: int = 5432
    pg_database: str
    pg_user: str
    pg_password: str

    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    try:
        return Settings()
    except ValidationError as e:
        missing = [err["loc"][0] for err in e.errors() if err["type"] == "missing"]
        for field in missing:
            logger.error("MISSING required env variable: %s", field.upper())
        logger.error("Application cannot start. Add the missing variables to .env and restart.")
        sys.exit(1)
