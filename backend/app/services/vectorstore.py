from typing import List, Optional

from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from ..config import settings

_VS: Optional[QdrantVectorStore] = None
_EMB: Optional[OllamaEmbeddings] = None


def build_embeddings() -> OllamaEmbeddings:
    global _EMB
    if _EMB is None:
        _EMB = OllamaEmbeddings(
            base_url=settings.ollama_base_url,
            model="nomic-embed-text",
        )
    return _EMB


def get_vectorstore() -> QdrantVectorStore:
    global _VS
    if _VS is None:
        embeddings = build_embeddings()
        _VS = QdrantVectorStore.from_existing_collection(
            url=settings.qdrant_url,
            collection_name=settings.collection_name,
            embedding=embeddings,
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

    # count first (so we can return how many we removed)
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
