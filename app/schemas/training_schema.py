from pydantic import BaseModel
from typing import List


# ── Question-SQL ─────────────────────────────────────────────────────────────

class QuestionSQLRecord(BaseModel):
    question: str
    sql: str


class BulkQuestionSQLRequest(BaseModel):
    records: List[QuestionSQLRecord]


# ── DDL ──────────────────────────────────────────────────────────────────────

class DDLRecord(BaseModel):
    table_name: str
    ddl: str
    description: str


class BulkDDLRequest(BaseModel):
    records: List[DDLRecord]


# ── Docs ─────────────────────────────────────────────────────────────────────

class DocsRecord(BaseModel):
    content: str
    category: str


class BulkDocsRequest(BaseModel):
    records: List[DocsRecord]


# ── Shared responses ─────────────────────────────────────────────────────────

class TrainResponse(BaseModel):
    id: str
    message: str


class BulkTrainResponse(BaseModel):
    inserted: int
    message: str
