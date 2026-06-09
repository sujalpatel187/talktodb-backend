import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from qdrant_client.models import PointStruct

from app.config import get_settings
from app.schemas.glossary_schema import (
    GlossaryRecord,
    BulkGlossaryRequest,
    GlossaryTrainResponse,
    BulkGlossaryTrainResponse,
)
from app.services.embedding_service import get_embedding, get_embeddings
from app.services.glossary_service import push_glossary_record
from app.services.qdrant_service import bulk_upsert

router = APIRouter(prefix="/train", tags=["training"])
logger = logging.getLogger(__name__)
settings = get_settings()


@router.post("/glossary", response_model=GlossaryTrainResponse)
async def train_glossary(body: GlossaryRecord):
    """Add a single business glossary entry to the glossary collection.

    The embedding is generated from ``"{term} {meaning}"`` so that
    both the acronym and its full expansion are semantically searchable.
    """
    try:
        embed_text = f"{body.term} {body.meaning}"
        embedding = get_embedding(embed_text)
        point_id = push_glossary_record(
            term=body.term,
            meaning=body.meaning,
            sql_hint=body.sql_hint,
            category=body.category,
            embedding=embedding,
        )
        logger.info("Glossary term ingested: %s", body.term)
        return GlossaryTrainResponse(
            id=point_id,
            term=body.term,
            message=f"Glossary term '{body.term}' inserted successfully.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/glossary/bulk", response_model=BulkGlossaryTrainResponse)
async def train_glossary_bulk(body: BulkGlossaryRequest):
    """Bulk insert business glossary entries into the glossary collection."""
    try:
        embed_texts = [f"{r.term} {r.meaning}" for r in body.records]
        embeddings = get_embeddings(embed_texts)
        now = datetime.now(timezone.utc).isoformat()

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=emb,
                payload={
                    "term": r.term,
                    "meaning": r.meaning,
                    "sql_hint": r.sql_hint,
                    "category": r.category,
                    "timestamp": now,
                },
            )
            for r, emb in zip(body.records, embeddings)
        ]

        count = bulk_upsert(settings.qdrant_glossary_collection_name, points)
        logger.info("Bulk glossary ingestion: %d terms inserted.", count)
        return BulkGlossaryTrainResponse(
            inserted=count,
            message=f"{count} glossary terms inserted successfully.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
