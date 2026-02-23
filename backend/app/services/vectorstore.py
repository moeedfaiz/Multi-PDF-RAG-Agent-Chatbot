# from typing import List, Optional

# from langchain_qdrant import QdrantVectorStore
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_core.documents import Document
# from qdrant_client import QdrantClient
# from qdrant_client.http import models as rest

# from ..config import settings

# _VS: Optional[QdrantVectorStore] = None
# _EMB: Optional[OllamaEmbeddings] = None


# def build_embeddings() -> OllamaEmbeddings:
#     global _EMB
#     if _EMB is None:
#         _EMB = OllamaEmbeddings(
#             base_url=settings.ollama_base_url,
#             model="nomic-embed-text",
#         )
#     return _EMB


# def get_vectorstore() -> QdrantVectorStore:
#     global _VS
#     if _VS is None:
#         embeddings = build_embeddings()
#         _VS = QdrantVectorStore.from_existing_collection(
#             url=settings.qdrant_url,
#             collection_name=settings.collection_name,
#             embedding=embeddings,
#         )
#     return _VS


# def _filter_for(tenant_id: str, file_id: str) -> rest.Filter:
#     return rest.Filter(
#         must=[
#             rest.FieldCondition(
#                 key="metadata.tenant_id",
#                 match=rest.MatchValue(value=tenant_id),
#             ),
#             rest.FieldCondition(
#                 key="metadata.file_id",
#                 match=rest.MatchValue(value=file_id),
#             ),
#         ]
#     )


# def count_chunks(*, tenant_id: str, file_id: str) -> int:
#     client = QdrantClient(url=settings.qdrant_url)
#     res = client.count(
#         collection_name=settings.collection_name,
#         count_filter=_filter_for(tenant_id, file_id),
#         exact=True,
#     )
#     return int(res.count or 0)


# def delete_chunks(*, tenant_id: str, file_id: str) -> int:
#     client = QdrantClient(url=settings.qdrant_url)

#     # count first (so we can return how many we removed)
#     n = count_chunks(tenant_id=tenant_id, file_id=file_id)

#     client.delete(
#         collection_name=settings.collection_name,
#         points_selector=rest.FilterSelector(filter=_filter_for(tenant_id, file_id)),
#         wait=True,
#     )
#     return n


# def upsert_docs(docs: List[Document]) -> int:
#     vs = get_vectorstore()
#     ids = vs.add_documents(docs)
#     return len(ids)


# def similarity_search(
#     query: str,
#     k: int = 8,
#     file_ids: Optional[List[str]] = None,
#     tenant_id: Optional[str] = None,
# ):
#     vs = get_vectorstore()

#     must = []
#     if tenant_id:
#         must.append({"key": "metadata.tenant_id", "match": {"value": tenant_id}})
#     if file_ids:
#         must.append({"key": "metadata.file_id", "match": {"any": file_ids}})

#     qdrant_filter = {"must": must} if must else None
#     return vs.similarity_search(query=query, k=k, filter=qdrant_filter)



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
# Embeddings (Gemini or Ollama)
# -----------------------------
class GeminiEmbeddings(Embeddings):
    """
    Lightweight Gemini embeddings wrapper without pulling in extra LangChain providers.
    Uses google-genai SDK (recommended) OR google-generativeai as fallback.
    """

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

        # Try google-genai first
        self._client = None
        try:
            from google import genai  # type: ignore
            self._client = genai.Client(api_key=self.api_key)
            self._mode = "genai"
        except Exception:
            self._mode = None

        # Fallback to google-generativeai
        if self._client is None:
            try:
                import google.generativeai as genai_old  # type: ignore
                genai_old.configure(api_key=self.api_key)
                self._client = genai_old
                self._mode = "generativeai"
            except Exception as e:
                raise RuntimeError(
                    "Gemini embeddings selected but no Gemini SDK is available. "
                    "Install `google-genai` (preferred) or `google-generativeai`."
                ) from e

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        texts = [t if t is not None else "" for t in texts]
        if self._mode == "genai":
            # google-genai
            out = []
            for t in texts:
                res = self._client.models.embed_content(
                    model=self.model,
                    contents=t,
                )
                # google-genai returns embeddings in a nested structure
                vec = res.embeddings[0].values
                out.append(list(vec))
            return out

        # google-generativeai
        out = []
        for t in texts:
            res = self._client.embed_content(
                model=self.model,
                content=t,
                task_type="retrieval_document",
            )
            out.append(list(res["embedding"]))
        return out

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


_VS: Optional[QdrantVectorStore] = None
_EMB: Optional[Embeddings] = None


def build_embeddings():
    global _EMB
    if _EMB is not None:
        return _EMB

    provider = (getattr(settings, "embeddings_provider", None) or "ollama").lower()

    if provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        _EMB = GoogleGenerativeAIEmbeddings(
            google_api_key=settings.gemini_api_key,
            model=getattr(settings, "gemini_embed_model", "models/text-embedding-004"),
        )
        return _EMB

    # local-only fallback
    _EMB = OllamaEmbeddings(
        base_url=settings.ollama_base_url,
        model="nomic-embed-text",
    )
    return _EMB


# -----------------------------
# Qdrant helpers
# -----------------------------
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
    global _VS
    if _VS is not None:
        return _VS

    emb = build_embeddings()

    # determine embedding dimension safely
    test_vec = emb.embed_query("dimension probe")
    dim = len(test_vec)

    client = QdrantClient(url=settings.qdrant_url)
    _ensure_collection_exists(client, settings.collection_name, dim)

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