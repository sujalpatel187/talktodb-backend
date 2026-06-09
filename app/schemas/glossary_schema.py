from pydantic import BaseModel, Field
from typing import List


# ── Single glossary entry ─────────────────────────────────────────────────────

class GlossaryRecord(BaseModel):
    term: str = Field(..., description="Business acronym or term, e.g. 'MDCS'")
    meaning: str = Field(..., description="Human-readable expansion of the term")
    sql_hint: str = Field(
        default="",
        description="Optional SQL fragment capturing the business rule, "
                    "e.g. \"society_type='DAIRY' AND registration_date>'2023-02-21'\"",
    )
    category: str = Field(
        default="business_term",
        description="Tag for grouping, e.g. 'business_term', 'acronym', 'kpi'",
    )


class BulkGlossaryRequest(BaseModel):
    records: List[GlossaryRecord]


# ── Responses ─────────────────────────────────────────────────────────────────

class GlossaryTrainResponse(BaseModel):
    id: str
    term: str
    message: str


class BulkGlossaryTrainResponse(BaseModel):
    inserted: int
    message: str
