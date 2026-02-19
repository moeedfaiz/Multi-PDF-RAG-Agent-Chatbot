from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Query
import uuid
from datetime import datetime

from ...config import settings
from ...deps import get_tenant_id
from ...schemas.upload import UploadResponse
from ...services.registry import append_record
from ...services.pdf_loader import extract_pdf_text_by_page
from ...services.chunker import chunk_pages
from ...services.vectorstore import upsert_docs
from ...services.mlflow_logger import Timer, log_ingest

router = APIRouter()


@router.post("/upload", response_model=UploadResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    tenant_id: str = Depends(get_tenant_id),
    ingest: bool = Query(False),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    settings.uploads_dir.mkdir(parents=True, exist_ok=True)

    file_id = str(uuid.uuid4())
    out_path = settings.uploads_dir / f"{file_id}.pdf"
    out_path.write_bytes(await file.read())

    append_record(
        settings.app_data_dir,
        {
            "tenant_id": tenant_id,
            "file_id": file_id,
            "filename": file.filename,
            "stored_name": out_path.name,
            "created_at": datetime.utcnow().isoformat() + "Z",
        },
    )

    # default response (upload-only)
    resp = UploadResponse(file_id=file_id, filename=file.filename)

    # optional ingest
    if ingest:
        pages = extract_pdf_text_by_page(out_path)

        docs = chunk_pages(
            pages,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            source_name=out_path.name,
            file_id=file_id,
            tenant_id=tenant_id,
        )

        with Timer() as t:
            num_added = upsert_docs(docs)

        log_ingest(
            file_id=file_id,
            filename=out_path.name,
            num_pages=len(pages),
            num_chunks=num_added,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
            collection=settings.collection_name,
            elapsed=t.dt,
        )

        resp.ingested = True
        resp.num_pages = len(pages)
        resp.num_chunks = num_added

    return resp
