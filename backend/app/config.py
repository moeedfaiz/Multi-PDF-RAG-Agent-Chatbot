from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pathlib import Path
from typing import Optional


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # -------------------------
    # Infra
    # -------------------------
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    mlflow_tracking_uri: str = Field(default="http://localhost:5000", alias="MLFLOW_TRACKING_URI")
    app_data_dir: Path = Field(default=Path("../data"), alias="APP_DATA_DIR")
    collection_name: str = Field(default="pdf_chunks", alias="COLLECTION_NAME")

    # -------------------------
    # RAG Settings
    # -------------------------
    chunk_size: int = Field(default=900, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=150, alias="CHUNK_OVERLAP")
    rag_max_distance: float = Field(default=0.35, alias="RAG_MAX_DISTANCE")

    # -------------------------
    # API Keys
    # -------------------------
    api_keys_json: str = Field(default='{"dev-key":"demo"}', alias="API_KEYS_JSON")

    # -------------------------
    # LLM Provider (chat)
    # -------------------------
    llm_provider: str = Field(default="ollama", alias="LLM_PROVIDER")  # ollama | gemini

    # -------------------------
    # Embeddings Provider (vector DB)
    # IMPORTANT: This is what fixes your Railway Ollama timeout
    # -------------------------
    embeddings_provider: str = Field(default="ollama", alias="EMBEDDINGS_PROVIDER")  # ollama | gemini

    # -------------------------
    # Ollama (local)
    # -------------------------
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="phi3:mini", alias="OLLAMA_MODEL")
    ollama_embed_model: str = Field(default="nomic-embed-text", alias="OLLAMA_EMBED_MODEL")

    # -------------------------
    # Gemini (prod)
    # -------------------------
    gemini_api_key: Optional[str] = Field(default=None, alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-1.5-flash", alias="GEMINI_MODEL")

    # Gemini embeddings model (default is a good one)
    gemini_embed_model: str = Field(default="models/text-embedding-004", alias="GEMINI_EMBED_MODEL")

    @property
    def uploads_dir(self) -> Path:
        return self.app_data_dir / "uploads"

    @property
    def parsed_dir(self) -> Path:
        return self.app_data_dir / "parsed"


settings = Settings()