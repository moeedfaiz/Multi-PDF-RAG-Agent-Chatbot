from typing import Optional
from pydantic import BaseModel

class IngestResponse(BaseModel):
    file_id: str
    num_pages: Optional[int] = None
    num_chunks: int
    collection: str
    ingested: bool = True
    already_ingested: bool = False
