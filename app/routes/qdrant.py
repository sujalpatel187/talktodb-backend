from fastapi import APIRouter, HTTPException
from app.schemas.qdrant_schema import SearchResponse, SimilarResult
from app.services.embedding_service import get_embedding
from app.services.qdrant_service import search_similar

router = APIRouter(prefix="/qdrant", tags=["qdrant"])


@router.get("/search", response_model=SearchResponse)
async def search_similar_questions(question: str):
    """
    Returns top 5 most similar questions from the Qdrant collection.
    """
    try:
        embedding = get_embedding(question)
        hits = search_similar(query_vector=embedding, top_k=5)

        results = [
            SimilarResult(
                score=round(hit.score, 4),
                question=hit.payload.get("question", ""),
                sql=hit.payload.get("sql", ""),
            )
            for hit in hits
        ]

        return SearchResponse(results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
