from fastapi import APIRouter, Depends, HTTPException
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from ...config import settings
from ...deps import get_tenant_id
from ...services.registry import load_records, rewrite_records

router = APIRouter()


def qdrant_count_for_file(client: QdrantClient, *, tenant_id: str, file_id: str) -> int:
    filt = rest.Filter(
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
    res = client.count(
        collection_name=settings.collection_name,
        count_filter=filt,
        exact=True,
    )
    return int(res.count or 0)


def qdrant_delete_for_file(client: QdrantClient, *, tenant_id: str, file_id: str) -> None:
    filt = rest.Filter(
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
    client.delete(
        collection_name=settings.collection_name,
        points_selector=rest.FilterSelector(filter=filt),
        wait=True,
    )


@router.get("/documents")
def documents(tenant_id: str = Depends(get_tenant_id)):
    # Load from registry.jsonl
    recs = [r for r in load_records(settings.app_data_dir) if r.get("tenant_id") == tenant_id]

    client = QdrantClient(url=settings.qdrant_url)

    enriched = []
    for r in recs:
        fid = r.get("file_id")
        if not fid:
            continue  # skip corrupted record

        n = qdrant_count_for_file(client, tenant_id=tenant_id, file_id=fid)
        enriched.append(
            {
                "tenant_id": tenant_id,
                "file_id": fid,
                "filename": r.get("filename"),
                "stored_name": r.get("stored_name", f"{fid}.pdf"),
                "created_at": r.get("created_at"),
                "num_chunks": n,
                "ingested": n > 0,
            }
        )

    # sort newest first if created_at exists
    enriched.sort(key=lambda x: (x["created_at"] or ""), reverse=True)

    return {"tenant_id": tenant_id, "docs": enriched}


@router.delete("/documents/{file_id}")
def delete_document(file_id: str, tenant_id: str = Depends(get_tenant_id)):
    # 1) remove from registry
    all_recs = load_records(settings.app_data_dir)
    tenant_recs = [r for r in all_recs if r.get("tenant_id") == tenant_id]

    exists = any(r.get("file_id") == file_id for r in tenant_recs)
    if not exists:
        raise HTTPException(status_code=404, detail="File not found in registry.")

    new_all = [r for r in all_recs if not (r.get("tenant_id") == tenant_id and r.get("file_id") == file_id)]
    rewrite_records(settings.app_data_dir, new_all)

    # 2) delete pdf from disk
    pdf_path = settings.uploads_dir / f"{file_id}.pdf"
    if pdf_path.exists():
        pdf_path.unlink()

    # (optional) delete parsed cache if you store it with file_id
    parsed_path = settings.parsed_dir / f"{file_id}.json"
    if parsed_path.exists():
        parsed_path.unlink()

    # 3) delete vectors from qdrant
    client = QdrantClient(url=settings.qdrant_url)
    qdrant_delete_for_file(client, tenant_id=tenant_id, file_id=file_id)

    return {"tenant_id": tenant_id, "file_id": file_id, "deleted": True}
