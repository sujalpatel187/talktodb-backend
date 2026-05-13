from pydantic import BaseModel


class ChatRequest(BaseModel):
    userMessage: str


class ChatResponse(BaseModel):
    userMessage: str
    generatedSQL: str | None = None
    error: str | None = None
