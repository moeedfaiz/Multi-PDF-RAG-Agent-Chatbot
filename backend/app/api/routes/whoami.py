from fastapi import APIRouter
from ...config import settings

router = APIRouter()

@router.get("/whoami")
def whoami():
    return {
        "llm_provider": settings.llm_provider,
        "gemini_model": getattr(settings, "gemini_model", None),
        "ollama_model": getattr(settings, "ollama_model", None),
    }
