from typing import List, Optional, Tuple
from langchain_core.documents import Document

from ..config import settings
from .vectorstore import similarity_search


def retrieve(
    question: str,
    *,
    top_k: int,
    file_ids: Optional[List[str]],
    tenant_id: str,
) -> List[Tuple[Document, float]]:
    docs = similarity_search(
        query=question,
        k=top_k,
        file_ids=file_ids,
        tenant_id=tenant_id,
    )

    # Placeholder scores (LangChain similarity_search doesn't always return scores)
    return [(d, 0.0) for d in docs]


def build_context(docs: List[Document], *, max_chars: int = 3000) -> str:
    """
    Keep context small for phi3:mini speed.
    """
    parts = []
    total = 0

    for d in docs:
        txt = (d.page_content or "").strip()
        if not txt:
            continue

        meta = d.metadata or {}
        header = f"[source={meta.get('source','doc')} page={meta.get('page','?')}]"
        chunk = f"{header}\n{txt}\n"

        if total + len(chunk) > max_chars:
            break

        parts.append(chunk)
        total += len(chunk)

    return "\n".join(parts)


def make_citations(docs: List[Document], scores: List[float]):
    cits = []
    for d, s in zip(docs, scores):
        meta = d.metadata or {}
        snippet = (d.page_content or "").strip()
        if len(snippet) > 240:
            snippet = snippet[:240] + "..."

        cits.append(
            {
                "source": meta.get("source", "doc"),
                "page": int(meta.get("page", 0) or 0),
                "snippet": snippet,
                "score": float(s),
            }
        )
    return cits
