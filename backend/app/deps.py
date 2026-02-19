import json
from fastapi import Header, HTTPException
from .config import settings

def get_tenant_id(x_api_key: str = Header(default="", alias="X-API-Key")) -> str:
    try:
        mapping = json.loads(settings.api_keys_json)
    except Exception:
        mapping = {"dev-key": "demo"}

    if not x_api_key or x_api_key not in mapping:
        raise HTTPException(status_code=401, detail="Missing/invalid X-API-Key")

    return str(mapping[x_api_key])
