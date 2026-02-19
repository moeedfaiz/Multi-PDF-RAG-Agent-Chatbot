from fastapi import APIRouter, Depends
from ...deps import get_tenant_id
from ...services.vectorstore import similarity_search

router = APIRouter()

@router.get("/debug/search")
def debug_search(
    q: str,
    k: int = 5,
    file_id: str | None = None,
    tenant_id: str = Depends(get_tenant_id),
):
    file_ids = [file_id] if file_id else None

    docs = similarity_search(
        query=q,
        k=k,
        tenant_id=tenant_id,
        file_ids=file_ids,
    )

    out = []
    for d in docs:
        txt = (d.page_content or "").strip()
        if len(txt) > 220:
            txt = txt[:220] + "…"
        out.append({"snippet": txt, "metadata": d.metadata or {}})

    return {"k": k, "docs": out}
