from typing import List
from langchain_core.documents import Document

def should_refuse(docs: List[Document]) -> bool:
    if not docs:
        return True

    # If all chunks are tiny/empty, refuse
    non_empty = [d for d in docs if (d.page_content or "").strip()]
    if not non_empty:
        return True

    total_chars = sum(len(d.page_content.strip()) for d in non_empty[:5])
    return total_chars < 200
