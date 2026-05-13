from pydantic import BaseModel


class SimilarResult(BaseModel):
    score: float
    question: str
    sql: str


class SearchResponse(BaseModel):
    results: list[SimilarResult]
