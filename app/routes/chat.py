from fastapi import APIRouter, HTTPException
from app.schemas.chat_schema import ChatRequest, ChatResponse
from app.services.chat_service import run_chat_pipeline

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/", response_model=ChatResponse)
async def chat(body: ChatRequest):
    try:
        return run_chat_pipeline(body.userMessage)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
