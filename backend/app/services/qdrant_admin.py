from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from ..config import settings


def qdrant_client() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url)


def delete_points_for_file(*, tenant_id: str, file_id: str) -> int:
    """
    Deletes all points where metadata.tenant_id == tenant_id AND metadata.file_id == file_id
    Returns number deleted (best-effort: Qdrant returns operation result, not always exact count).
    """
    client = qdrant_client()

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

    # delete by filter
    client.delete(
        collection_name=settings.collection_name,
        points_selector=rest.FilterSelector(filter=filt),
        wait=True,
    )

    # optional: count after delete to verify
    res = client.count(
        collection_name=settings.collection_name,
        count_filter=filt,
        exact=True,
    )
    remaining = int(res.count or 0)

    # We can't always know exactly what was deleted, but we can say "remaining == 0"
    return 0 if remaining == 0 else remaining
