from typing import List, Optional
from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    question: str
    file_ids: Optional[List[str]] = None
    top_k: int = Field(default=8, ge=1, le=30)
    use_rerank: bool = True  # placeholder, not used yet
    max_tokens: int = Field(default=512, ge=64, le=2048)

class Citation(BaseModel):
    source: str
    page: int
    snippet: str
    score: float

class ChatResponse(BaseModel):
    answer: str
    refused: bool = False
    citations: List[Citation] = []
