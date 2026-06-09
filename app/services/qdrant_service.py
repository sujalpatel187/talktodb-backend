import uuid
import logging
import warnings
from datetime import datetime, timezone
from qdrant_client import QdrantClient
from qdrant_client.models import (
    ScoredPoint, PointStruct, Record,
    VectorParams, Distance,
)
from typing import List, Optional
from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            _client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key or None,
            )
    return _client


def ensure_collection(collection_name: str, vector_size: int | None = None) -> bool:
    """Create *collection_name* if it does not already exist.

    Args:
        collection_name: Name of the Qdrant collection to verify / create.
        vector_size:     Embedding dimension.  Defaults to
                         ``settings.embedding_dimension``.

    Returns:
        ``True`` if the collection was created, ``False`` if it already existed.
    """
    client = get_client()
    dim = vector_size or settings.embedding_dimension

    existing = {c.name for c in client.get_collections().collections}
    if collection_name in existing:
        logger.info("Collection already exists: %s", collection_name)
        return False

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    logger.info("Collection created: %s  (dim=%d, metric=cosine)", collection_name, dim)
    return True


def search_similar(
    query_vector: List[float],
    top_k: int = 5,
    score_threshold: float = 0.0,
) -> List[ScoredPoint]:
    """
    Search for similar vectors in the configured Qdrant collection.

    Args:
        query_vector: Embedding vector to search with.
        top_k: Number of top results to return.
        score_threshold: Minimum similarity score (0.0 – 1.0).

    Returns:
        List of ScoredPoint results from Qdrant.
    """
    client = get_client()

    results = client.search(
        collection_name=settings.qdrant_collection_name,
        query_vector=query_vector,
        limit=top_k,
        score_threshold=score_threshold,
    )

    return results


def search_ddl(query_vector: List[float], top_k: int = 5) -> List[ScoredPoint]:
    """Search top-k similar DDL entries from the DDL collection."""
    client = get_client()
    return client.search(
        collection_name=settings.qdrant_ddl_collection_name,
        query_vector=query_vector,
        limit=top_k,
    )


def search_docs(query_vector: List[float], top_k: int = 5) -> List[ScoredPoint]:
    """Search top-k similar documentation entries from the docs collection."""
    client = get_client()
    return client.search(
        collection_name=settings.qdrant_docs_collection_name,
        query_vector=query_vector,
        limit=top_k,
    )


def search_glossary(
    query_vector: List[float],
    top_k: int = 5,
    score_threshold: float = 0.0,
) -> List[ScoredPoint]:
    """Search the business glossary collection for matching terms."""
    client = get_client()
    return client.search(
        collection_name=settings.qdrant_glossary_collection_name,
        query_vector=query_vector,
        limit=top_k,
        score_threshold=score_threshold,
    )


def push_record(question: str, sql: str, embedding: List[float]) -> str:
    """
    Push a question/sql record with its embedding to the Qdrant collection.

    Args:
        question: The natural language question.
        sql: The corresponding SQL query.
        embedding: Pre-computed embedding vector of the question.

    Returns:
        The UUID string of the inserted point.
    """
    client = get_client()
    point_id = str(uuid.uuid4())

    client.upsert(
        collection_name=settings.qdrant_collection_name,
        points=[
            PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "question": question,
                    "sql": sql,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
        ],
    )

    return point_id


def push_ddl_record(table_name: str, ddl: str, description: str, embedding: List[float]) -> str:
    """Push a DDL record to the DDL collection."""
    client = get_client()
    point_id = str(uuid.uuid4())
    client.upsert(
        collection_name=settings.qdrant_ddl_collection_name,
        points=[
            PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "table_name": table_name,
                    "ddl": ddl,
                    "description": description,
                    "type": "ddl",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
        ],
    )
    return point_id


def push_docs_record(content: str, category: str, embedding: List[float]) -> str:
    """Push a documentation record to the docs collection."""
    client = get_client()
    point_id = str(uuid.uuid4())
    client.upsert(
        collection_name=settings.qdrant_docs_collection_name,
        points=[
            PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "content": content,
                    "category": category,
                    "type": "documentation",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
        ],
    )
    return point_id


def bulk_upsert(collection_name: str, points: List[PointStruct]) -> int:
    """Bulk upsert a list of PointStructs into a collection. Returns count inserted."""
    client = get_client()
    client.upsert(collection_name=collection_name, points=points)
    return len(points)


def get_all_points(collection_name: str, limit: int = 100, offset: Optional[str] = None) -> List[Record]:
    """Scroll through all points in a collection."""
    client = get_client()
    records, _ = client.scroll(
        collection_name=collection_name,
        limit=limit,
        offset=offset,
        with_payload=True,
        with_vectors=False,
    )
    return records


def update_point_payload(collection_name: str, point_id: str, payload: dict) -> None:
    """Overwrite the payload of an existing point."""
    client = get_client()
    client.set_payload(
        collection_name=collection_name,
        payload=payload,
        points=[point_id],
    )


def delete_point(collection_name: str, point_id: str) -> None:
    """Delete a single point by ID from a collection."""
    client = get_client()
    client.delete(
        collection_name=collection_name,
        points_selector=[point_id],
    )
