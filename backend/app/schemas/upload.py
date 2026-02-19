from typing import Optional
from pydantic import BaseModel

class UploadResponse(BaseModel):
    file_id: str
    filename: str
    ingested: bool = False
    num_pages: Optional[int] = None
    num_chunks: Optional[int] = None
