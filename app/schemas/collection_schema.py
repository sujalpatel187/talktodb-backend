from pydantic import BaseModel
from typing import Any


class PointPayload(BaseModel):
    id: str
    payload: dict[str, Any]


class GetAllResponse(BaseModel):
    total: int
    points: list[PointPayload]


class DeleteResponse(BaseModel):
    id: str
    message: str


class UpdateResponse(BaseModel):
    id: str
    message: str


# ── Per-collection update request bodies ─────────────────────────────────────

class UpdateQuestionSQL(BaseModel):
    question: str
    sql: str


class UpdateDDL(BaseModel):
    table_name: str
    ddl: str
    description: str


class UpdateDocs(BaseModel):
    content: str
    category: str


class UpdateGlossary(BaseModel):
    term: str
    meaning: str
    sql_hint: str = ""
    category: str = "business_term"
