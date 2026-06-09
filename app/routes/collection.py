import logging
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Query
from app.schemas.collection_schema import (
    GetAllResponse, PointPayload,
    DeleteResponse, UpdateResponse,
    UpdateQuestionSQL, UpdateDDL, UpdateDocs, UpdateGlossary,
)
from app.services.qdrant_service import get_all_points, update_point_payload, delete_point
from app.services.embedding_service import get_embedding
from app.config import get_settings

router = APIRouter(prefix="/collection", tags=["collection"])
logger = logging.getLogger(__name__)
settings = get_settings()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_response(records) -> GetAllResponse:
    points = [PointPayload(id=str(r.id), payload=r.payload or {}) for r in records]
    return GetAllResponse(total=len(points), points=points)


# ════════════════════════════════════════════════════════════════════════════
# Question-SQL collection
# ════════════════════════════════════════════════════════════════════════════

@router.get("/question-sql", response_model=GetAllResponse)
async def get_all_question_sql(limit: int = Query(100, le=1000)):
    try:
        records = get_all_points(settings.qdrant_collection_name, limit=limit)
        return _to_response(records)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/question-sql/{point_id}", response_model=UpdateResponse)
async def update_question_sql(point_id: str, body: UpdateQuestionSQL):
    try:
        embedding = get_embedding(body.question)
        payload = {
            "question": body.question,
            "sql": body.sql,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        update_point_payload(settings.qdrant_collection_name, point_id, payload)
        # re-upsert with new vector
        from app.services.qdrant_service import get_client
        from qdrant_client.models import PointStruct
        get_client().upsert(
            collection_name=settings.qdrant_collection_name,
            points=[PointStruct(id=point_id, vector=embedding, payload=payload)],
        )
        logger.info("Updated Q&A point: %s", point_id)
        return UpdateResponse(id=point_id, message="Question-SQL record updated.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/question-sql/{point_id}", response_model=DeleteResponse)
async def delete_question_sql(point_id: str):
    try:
        delete_point(settings.qdrant_collection_name, point_id)
        logger.info("Deleted Q&A point: %s", point_id)
        return DeleteResponse(id=point_id, message="Question-SQL record deleted.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ════════════════════════════════════════════════════════════════════════════
# DDL collection
# ════════════════════════════════════════════════════════════════════════════

@router.get("/ddl", response_model=GetAllResponse)
async def get_all_ddl(limit: int = Query(100, le=1000)):
    try:
        records = get_all_points(settings.qdrant_ddl_collection_name, limit=limit)
        return _to_response(records)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/ddl/{point_id}", response_model=UpdateResponse)
async def update_ddl(point_id: str, body: UpdateDDL):
    try:
        embed_text = f"{body.table_name} {body.description} {body.ddl}"
        embedding = get_embedding(embed_text)
        payload = {
            "table_name": body.table_name,
            "ddl": body.ddl,
            "description": body.description,
            "type": "ddl",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        from app.services.qdrant_service import get_client
        from qdrant_client.models import PointStruct
        get_client().upsert(
            collection_name=settings.qdrant_ddl_collection_name,
            points=[PointStruct(id=point_id, vector=embedding, payload=payload)],
        )
        logger.info("Updated DDL point: %s", point_id)
        return UpdateResponse(id=point_id, message="DDL record updated.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/ddl/{point_id}", response_model=DeleteResponse)
async def delete_ddl(point_id: str):
    try:
        delete_point(settings.qdrant_ddl_collection_name, point_id)
        logger.info("Deleted DDL point: %s", point_id)
        return DeleteResponse(id=point_id, message="DDL record deleted.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ════════════════════════════════════════════════════════════════════════════
# Docs collection
# ════════════════════════════════════════════════════════════════════════════

@router.get("/docs", response_model=GetAllResponse)
async def get_all_docs(limit: int = Query(100, le=1000)):
    try:
        records = get_all_points(settings.qdrant_docs_collection_name, limit=limit)
        return _to_response(records)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/docs/{point_id}", response_model=UpdateResponse)
async def update_docs(point_id: str, body: UpdateDocs):
    try:
        embedding = get_embedding(body.content)
        payload = {
            "content": body.content,
            "category": body.category,
            "type": "documentation",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        from app.services.qdrant_service import get_client
        from qdrant_client.models import PointStruct
        get_client().upsert(
            collection_name=settings.qdrant_docs_collection_name,
            points=[PointStruct(id=point_id, vector=embedding, payload=payload)],
        )
        logger.info("Updated docs point: %s", point_id)
        return UpdateResponse(id=point_id, message="Documentation record updated.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/docs/{point_id}", response_model=DeleteResponse)
async def delete_docs(point_id: str):
    try:
        delete_point(settings.qdrant_docs_collection_name, point_id)
        logger.info("Deleted docs point: %s", point_id)
        return DeleteResponse(id=point_id, message="Documentation record deleted.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════
# Glossary collection
# ═══════════════════════════════════════════════════════════════════════════

@router.get("/glossary", response_model=GetAllResponse)
async def get_all_glossary(limit: int = Query(500, le=2000)):
    try:
        records = get_all_points(settings.qdrant_glossary_collection_name, limit=limit)
        return _to_response(records)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/glossary/{point_id}", response_model=UpdateResponse)
async def update_glossary(point_id: str, body: UpdateGlossary):
    try:
        embed_text = f"{body.term} {body.meaning}"
        embedding = get_embedding(embed_text)
        payload = {
            "term": body.term,
            "meaning": body.meaning,
            "sql_hint": body.sql_hint,
            "category": body.category,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        from app.services.qdrant_service import get_client
        from qdrant_client.models import PointStruct
        get_client().upsert(
            collection_name=settings.qdrant_glossary_collection_name,
            points=[PointStruct(id=point_id, vector=embedding, payload=payload)],
        )
        logger.info("Updated glossary point: %s", point_id)
        return UpdateResponse(id=point_id, message="Glossary record updated.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/glossary/{point_id}", response_model=DeleteResponse)
async def delete_glossary(point_id: str):
    try:
        delete_point(settings.qdrant_glossary_collection_name, point_id)
        logger.info("Deleted glossary point: %s", point_id)
        return DeleteResponse(id=point_id, message="Glossary record deleted.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
