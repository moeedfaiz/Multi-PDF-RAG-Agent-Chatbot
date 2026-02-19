from fastapi import APIRouter, Depends
from ...config import settings
from ...deps import get_tenant_id
from ...services.registry import load_records, rewrite_records

router = APIRouter()

@router.post("/admin/registry/cleanup")
def cleanup_registry(tenant_id: str = Depends(get_tenant_id)):
    records = load_records(settings.app_data_dir)

    fixed = []
    changed = 0

    for r in records:
        # keep other tenants untouched
        if r.get("tenant_id") != tenant_id:
            fixed.append(r)
            continue

        fr = dict(r)

        fid = fr.get("file_id")
        if isinstance(fid, str) and fid.endswith(".pdf"):
            fr["file_id"] = fid[:-4]
            changed += 1

        if not fr.get("filename"):
            fr["filename"] = fr.get("stored_name")
            changed += 1

        fixed.append(fr)

    rewrite_records(settings.app_data_dir, fixed)
    return {"tenant_id": tenant_id, "changed": changed}
