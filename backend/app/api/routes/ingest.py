from fastapi import APIRouter, HTTPException, Depends, Query
from ...config import settings
from ...deps import get_tenant_id
from ...schemas.ingest import IngestResponse

from ...services.pdf_loader import extract_pdf_text_by_page
from ...services.chunker import chunk_pages
from ...services.vectorstore import upsert_docs, count_chunks, delete_chunks
from ...services.mlflow_logger import Timer, log_ingest

router = APIRouter()

@router.post("/ingest/{file_id}", response_model=IngestResponse)
def ingest(
    file_id: str,
    tenant_id: str = Depends(get_tenant_id),
    force: bool = Query(False, description="If true, delete existing chunks and re-ingest"),
):
    pdf_path = settings.uploads_dir / f"{file_id}.pdf"
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF not found. Upload first.")

    existing = count_chunks(tenant_id=tenant_id, file_id=file_id)
    if existing > 0 and not force:
        return IngestResponse(
            file_id=file_id,
            num_pages=None,
            num_chunks=existing,
            collection=settings.collection_name,
            ingested=True,
            already_ingested=True,
        )

    if force and existing > 0:
        delete_chunks(tenant_id=tenant_id, file_id=file_id)

    pages = extract_pdf_text_by_page(pdf_path)
    docs = chunk_pages(
        pages,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        source_name=pdf_path.name,
        file_id=file_id,
        tenant_id=tenant_id,
    )

    with Timer() as t:
        num_added = upsert_docs(docs)

    log_ingest(
        file_id=file_id,
        filename=pdf_path.name,
        num_pages=len(pages),
        num_chunks=num_added,
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
        collection=settings.collection_name,
        elapsed=t.dt,
    )

    return IngestResponse(
        file_id=file_id,
        num_pages=len(pages),
        num_chunks=num_added,
        collection=settings.collection_name,
        ingested=True,
        already_ingested=False,
    )
