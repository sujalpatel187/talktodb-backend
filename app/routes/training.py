import logging
from fastapi import APIRouter, HTTPException
from qdrant_client.models import PointStruct
from app.schemas.training_schema import (
    QuestionSQLRecord, BulkQuestionSQLRequest,
    DDLRecord, BulkDDLRequest,
    DocsRecord, BulkDocsRequest,
    TrainResponse, BulkTrainResponse,
)
from app.services.embedding_service import get_embedding, get_embeddings
from app.services.qdrant_service import (
    push_record, push_ddl_record, push_docs_record, bulk_upsert,
)
from app.config import get_settings
from datetime import datetime, timezone
import uuid

router = APIRouter(prefix="/train", tags=["training"])
logger = logging.getLogger(__name__)
settings = get_settings()


# ── Question-SQL ─────────────────────────────────────────────────────────────

@router.post("/question-sql", response_model=TrainResponse)
async def train_question_sql(body: QuestionSQLRecord):
    """Add a single question-SQL pair to the Q&A collection."""
    try:
        embedding = get_embedding(body.question)
        point_id = push_record(question=body.question, sql=body.sql, embedding=embedding)
        logger.info("Trained Q&A: %s", body.question)
        return TrainResponse(id=point_id, message="Question-SQL record inserted.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/question-sql/bulk", response_model=BulkTrainResponse)
async def train_question_sql_bulk(body: BulkQuestionSQLRequest):
    """Bulk insert question-SQL pairs into the Q&A collection."""
    try:
        texts = [r.question for r in body.records]
        embeddings = get_embeddings(texts)
        now = datetime.now(timezone.utc).isoformat()
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=emb,
                payload={"question": r.question, "sql": r.sql, "timestamp": now},
            )
            for r, emb in zip(body.records, embeddings)
        ]
        count = bulk_upsert(settings.qdrant_collection_name, points)
        logger.info("Bulk trained %d Q&A records.", count)
        return BulkTrainResponse(inserted=count, message=f"{count} question-SQL records inserted.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── DDL ──────────────────────────────────────────────────────────────────────

@router.post("/ddl", response_model=TrainResponse)
async def train_ddl(body: DDLRecord):
    """Add a single DDL record to the DDL collection."""
    try:
        embed_text = f"{body.table_name} {body.description} {body.ddl}"
        embedding = get_embedding(embed_text)
        point_id = push_ddl_record(
            table_name=body.table_name,
            ddl=body.ddl,
            description=body.description,
            embedding=embedding,
        )
        logger.info("Trained DDL: %s", body.table_name)
        return TrainResponse(id=point_id, message="DDL record inserted.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ddl/bulk", response_model=BulkTrainResponse)
async def train_ddl_bulk(body: BulkDDLRequest):
    """Bulk insert DDL records into the DDL collection."""
    try:
        texts = [f"{r.table_name} {r.description} {r.ddl}" for r in body.records]
        embeddings = get_embeddings(texts)
        now = datetime.now(timezone.utc).isoformat()
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=emb,
                payload={
                    "table_name": r.table_name,
                    "ddl": r.ddl,
                    "description": r.description,
                    "type": "ddl",
                    "timestamp": now,
                },
            )
            for r, emb in zip(body.records, embeddings)
        ]
        count = bulk_upsert(settings.qdrant_ddl_collection_name, points)
        logger.info("Bulk trained %d DDL records.", count)
        return BulkTrainResponse(inserted=count, message=f"{count} DDL records inserted.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Docs ─────────────────────────────────────────────────────────────────────

@router.post("/docs", response_model=TrainResponse)
async def train_docs(body: DocsRecord):
    """Add a single documentation record to the docs collection."""
    try:
        embedding = get_embedding(body.content)
        point_id = push_docs_record(
            content=body.content,
            category=body.category,
            embedding=embedding,
        )
        logger.info("Trained doc: [%s]", body.category)
        return TrainResponse(id=point_id, message="Documentation record inserted.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/docs/bulk", response_model=BulkTrainResponse)
async def train_docs_bulk(body: BulkDocsRequest):
    """Bulk insert documentation records into the docs collection."""
    try:
        texts = [r.content for r in body.records]
        embeddings = get_embeddings(texts)
        now = datetime.now(timezone.utc).isoformat()
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=emb,
                payload={
                    "content": r.content,
                    "category": r.category,
                    "type": "documentation",
                    "timestamp": now,
                },
            )
            for r, emb in zip(body.records, embeddings)
        ]
        count = bulk_upsert(settings.qdrant_docs_collection_name, points)
        logger.info("Bulk trained %d docs records.", count)
        return BulkTrainResponse(inserted=count, message=f"{count} documentation records inserted.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
