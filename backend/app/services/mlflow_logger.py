import time
import mlflow

def setup_mlflow(tracking_uri: str):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("pdf_rag")

class Timer:
    def __enter__(self):
        self.t0 = time.time()
        return self
    def __exit__(self, exc_type, exc, tb):
        self.dt = time.time() - self.t0

def log_ingest(file_id: str, filename: str, num_pages: int, num_chunks: int, chunk_size: int, overlap: int, collection: str, elapsed: float):
    with mlflow.start_run(run_name=f"ingest:{file_id}"):
        mlflow.log_params({
            "file_id": file_id,
            "filename": filename,
            "chunk_size": chunk_size,
            "chunk_overlap": overlap,
            "collection": collection,
        })
        mlflow.log_metrics({
            "num_pages": num_pages,
            "num_chunks": num_chunks,
            "ingest_seconds": elapsed,
        })
