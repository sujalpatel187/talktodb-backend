import logging
from fastapi import APIRouter, HTTPException
from app.schemas.execute_schema import ExecuteRequest, ExecuteResponse
from app.services.pg_service import execute_query

router = APIRouter(prefix="/execute", tags=["execute"])
logger = logging.getLogger(__name__)


@router.post("/", response_model=ExecuteResponse)
async def execute_sql(body: ExecuteRequest):
    """Execute a read-only SQL query against the PostgreSQL database."""
    try:
        result = execute_query(body.sql)

        return ExecuteResponse(
            columns=result["columns"],
            rows=result["rows"],
            rowCount=result["row_count"],
            error=result["error"],
        )
    except Exception as e:
        logger.error("Execute endpoint error: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))
