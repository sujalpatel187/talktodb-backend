from pydantic import BaseModel
from typing import Any


class ExecuteRequest(BaseModel):
    sql: str


class ExecuteResponse(BaseModel):
    columns: list[str] = []
    rows: list[list[Any]] = []
    rowCount: int = 0
    error: str | None = None
