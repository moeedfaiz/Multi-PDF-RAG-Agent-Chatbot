from __future__ import annotations

from typing import List, Optional
import os

from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from langchain_community.embeddings import OllamaEmbeddings

from ..config import settings


# -----------------------------
# Gemini Embeddings (google-genai)
# -----------------------------
class GeminiEmbeddings(Embeddings):
    """
    Embeddings using the NEW google-genai SDK (recommended).
    This avoids legacy v1beta model-name issues in langchain_google_genai.
    """

    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise ValueError("GEMINI_API_KEY is missing.")
        self.api_key = api_key
        self.model = model

        from google import genai  # google-genai
        self._client = genai.Client(api_key=self.api_key)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        texts = [t if t is not None else "" for t in texts]
        out: List[List[float]] = []

        # NOTE: google-genai supports embed_content. We call per-text for simplicity.
        for t in texts:
            res = self._client.models.embed_content(
                model=self.model,
                contents=t,
            )
            # `res.embeddings` is a list; each item has `.values`
            vec = res.embeddings[0].values
            out.append(list(vec))

        return out

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


_VS: Optional[QdrantVectorStore] = None
_EMB: Optional[Embeddings] = None
_DIM: Optional[int] = None  # cache embedding dim to avoid repeated probing


def build_embeddings() -> Embeddings:
    """
    Controlled by env:
      EMBEDDINGS_PROVIDER = gemini | ollama
    Models:
      GEMINI_EMBED_MODEL default -> models/gemini-embedding-001
      OLLAMA_EMBED_MODEL default -> nomic-embed-text
    """
    global _EMB
    if _EMB is not None:
        return _EMB

    provider = (getattr(settings, "embeddings_provider", None) or os.getenv("EMBEDDINGS_PROVIDER", "ollama")).lower()

    if provider == "gemini":
        model = getattr(settings, "gemini_embed_model", None) or os.getenv("GEMINI_EMBED_MODEL", "models/gemini-embedding-001")
        _EMB = GeminiEmbeddings(
            api_key=settings.gemini_api_key or "",
            model=model,
        )
        return _EMB

    # Local-only fallback (requires reachable Ollama server)
    ollama_embed_model = getattr(settings, "ollama_embed_model", None) or os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    _EMB = OllamaEmbeddings(
        base_url=settings.ollama_base_url,
        model=ollama_embed_model,
    )
    return _EMB


def _ensure_collection_exists(client: QdrantClient, collection_name: str, dim: int):
    try:
        client.get_collection(collection_name)
        return
    except Exception:
        pass

    client.create_collection(
        collection_name=collection_name,
        vectors_config=rest.VectorParams(
            size=dim,
            distance=rest.Distance.COSINE,
        ),
    )


def get_vectorstore() -> QdrantVectorStore:
    """
    Builds vectorstore and auto-creates collection if missing.
    """
    global _VS, _DIM
    if _VS is not None:
        return _VS

    emb = build_embeddings()

    # Determine embedding dimension once
    if _DIM is None:
        try:
            test_vec = emb.embed_query("dimension probe")
            _DIM = len(test_vec)
        except Exception as e:
            raise RuntimeError(
                f"Embedding probe failed. Check embeddings provider + model. "
                f"EMBEDDINGS_PROVIDER={getattr(settings,'embeddings_provider',None)} "
                f"GEMINI_EMBED_MODEL={getattr(settings,'gemini_embed_model',None)} "
                f"Original error: {e}"
            ) from e

    client = QdrantClient(url=settings.qdrant_url)
    _ensure_collection_exists(client, settings.collection_name, _DIM)

    _VS = QdrantVectorStore(
        client=client,
        collection_name=settings.collection_name,
        embedding=emb,
    )
    return _VS


def _filter_for(tenant_id: str, file_id: str) -> rest.Filter:
    return rest.Filter(
        must=[
            rest.FieldCondition(
                key="metadata.tenant_id",
                match=rest.MatchValue(value=tenant_id),
            ),
            rest.FieldCondition(
                key="metadata.file_id",
                match=rest.MatchValue(value=file_id),
            ),
        ]
    )


def count_chunks(*, tenant_id: str, file_id: str) -> int:
    client = QdrantClient(url=settings.qdrant_url)
    res = client.count(
        collection_name=settings.collection_name,
        count_filter=_filter_for(tenant_id, file_id),
        exact=True,
    )
    return int(res.count or 0)


def delete_chunks(*, tenant_id: str, file_id: str) -> int:
    client = QdrantClient(url=settings.qdrant_url)
    n = count_chunks(tenant_id=tenant_id, file_id=file_id)

    client.delete(
        collection_name=settings.collection_name,
        points_selector=rest.FilterSelector(filter=_filter_for(tenant_id, file_id)),
        wait=True,
    )
    return n


def upsert_docs(docs: List[Document]) -> int:
    vs = get_vectorstore()
    ids = vs.add_documents(docs)
    return len(ids)


def similarity_search(
    query: str,
    k: int = 8,
    file_ids: Optional[List[str]] = None,
    tenant_id: Optional[str] = None,
):
    vs = get_vectorstore()

    must = []
    if tenant_id:
        must.append({"key": "metadata.tenant_id", "match": {"value": tenant_id}})
    if file_ids:
        must.append({"key": "metadata.file_id", "match": {"any": file_ids}})

    qdrant_filter = {"must": must} if must else None
    return vs.similarity_search(query=query, k=k, filter=qdrant_filter)