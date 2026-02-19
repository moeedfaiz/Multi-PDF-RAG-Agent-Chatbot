from typing import List, Dict, Any, Union, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


PageLike = Union[
    Dict[str, Any],          # {"page": 1, "text": "..."} or {"page_number": 1, "content": "..."}
    Tuple[int, str],         # (1, "...")
    Tuple[str, str],         # ("1", "...")
    str                      # "text..." (rare)
]


def _normalize_page(p: PageLike, i: int) -> tuple[int, str]:
    # tuple form: (page, text)
    if isinstance(p, tuple) and len(p) >= 2:
        page_num = int(p[0]) if str(p[0]).isdigit() else i
        text = str(p[1] or "")
        return page_num, text

    # dict form
    if isinstance(p, dict):
        page_num = p.get("page", p.get("page_number", i))
        try:
            page_num = int(page_num)
        except Exception:
            page_num = i
        text = p.get("text") or p.get("content") or ""
        return page_num, str(text or "")

    # string form
    if isinstance(p, str):
        return i, p

    return i, ""


def chunk_pages(
    pages: List[PageLike],
    chunk_size: int,
    chunk_overlap: int,
    source_name: str,
    file_id: str,
    tenant_id: str,
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    docs: List[Document] = []

    for i, p in enumerate(pages, start=1):
        page_num, text = _normalize_page(p, i)
        text = (text or "").strip()
        if not text:
            continue

        chunks = splitter.split_text(text)

        for idx, chunk in enumerate(chunks):
            chunk = (chunk or "").strip()
            if not chunk:
                continue

            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "tenant_id": tenant_id,
                        "file_id": file_id,
                        "source": source_name,
                        "page": page_num,
                        "chunk_index": idx,
                    },
                )
            )

    return docs
