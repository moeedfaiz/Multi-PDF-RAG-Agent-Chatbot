````md
# Multi-PDF RAG Agent Chatbot (FastAPI + Qdrant + MLflow + Streamlit)

A production-ready **multi-PDF RAG chatbot**:
- Upload PDFs → chunk → embed → store in **Qdrant**
- Ask questions (streaming supported) with **citations**
- Track ingests/metrics in **MLflow**
- Clean UI in **Streamlit**

---

## Architecture

- **Backend:** FastAPI (RAG API, upload/ingest/chat/stream)
- **Vector DB:** Qdrant
- **Experiment Tracking:** MLflow
- **Frontend:** Streamlit (upload + library + chat + citations)
- **LLM:** Gemini or Ollama (configurable)
- **Embeddings:** Gemini or Ollama (configurable)

---

## Local (Docker Compose) — Recommended

### 1) Prerequisites
- Docker + Docker Compose

### 2) Setup env
Copy the example env file and update values:

```bash
cp .env.example .env
````

If you want **Gemini** (recommended):

* Set `LLM_PROVIDER=gemini`
* Set `EMBEDDINGS_PROVIDER=gemini`
* Set `GEMINI_API_KEY=...`
* Set `GEMINI_MODEL=models/gemini-flash-latest`
* Set `GEMINI_EMBED_MODEL=models/gemini-embedding-1.0`

If you want **Ollama** (local only):

* Set `LLM_PROVIDER=ollama`
* Set `EMBEDDINGS_PROVIDER=ollama`
* Make sure Ollama is running and `OLLAMA_BASE_URL` is correct

### 3) Run

```bash
docker compose up --build
```

### 4) Open

* **Frontend (Streamlit):** [http://localhost:8501](http://localhost:8501)
* **Backend (FastAPI docs):** [http://localhost:8000/docs](http://localhost:8000/docs)
* **Qdrant Dashboard:** [http://localhost:6333/dashboard](http://localhost:6333/dashboard)
* **MLflow UI:** [http://localhost:5000](http://localhost:5000)

---

## Local (Ollama option)

If using Ollama locally:

```bash
ollama pull phi3:mini
ollama pull nomic-embed-text
ollama serve
```

Then set in `.env`:

```env
LLM_PROVIDER=ollama
EMBEDDINGS_PROVIDER=ollama
OLLAMA_BASE_URL=http://host.docker.internal:11434
OLLAMA_MODEL=phi3:mini
OLLAMA_EMBED_MODEL=nomic-embed-text
```

---

## Production (Railway)

This repo can be deployed on Railway as **4 services**:

1. **Qdrant** (Docker Image)

* Image: `qdrant/qdrant:latest`
* Add a **Volume** mounted to: `/qdrant/storage`
* Internal URL used by backend:

  * `QDRANT_URL=http://qdrant.railway.internal:6333`

2. **MLflow** (Docker Image)

* Image: `ghcr.io/mlflow/mlflow:v2.14.3`
* Start command:

  ```bash
  mlflow server --host 0.0.0.0 --port $PORT \
    --backend-store-uri sqlite:////mlflow/mlflow.db \
    --default-artifact-root /mlflow/artifacts
  ```
* Add a **Volume** mounted to: `/mlflow`

3. **Backend (FastAPI)** (GitHub repo)

* Root directory: `/backend`
* Expose port: `8080` (or whatever Railway assigns via `$PORT`)
* Required variables:

  ```env
  API_KEYS_JSON={"dev-key":"demo"}
  APP_DATA_DIR=/app/data

  QDRANT_URL=http://qdrant.railway.internal:6333
  MLFLOW_TRACKING_URI=http://mlflow:8080
  COLLECTION_NAME=pdf_chunks_gemini

  CHUNK_SIZE=900
  CHUNK_OVERLAP=150
  RAG_MAX_DISTANCE=0.35

  LLM_PROVIDER=gemini
  EMBEDDINGS_PROVIDER=gemini
  GEMINI_API_KEY=YOUR_KEY
  GEMINI_MODEL=models/gemini-flash-latest
  GEMINI_EMBED_MODEL=models/gemini-embedding-1.0
  GEMINI_EMBED_BATCH_SIZE=32
  GEMINI_EMBED_MAX_RETRIES=10
  ```

4. **Frontend (Streamlit)** (GitHub repo)

* Root directory: `/frontend/streamlit`
* https://patient-analysis-production.up.railway.app/
* Expose port: `8501` (or Railway `$PORT`)
* Required variables:

  ```env
  BACKEND_URL=[https://<YOUR-BACKEND-DOMAIN>](https://multi-pdf-rag-agent-chatbot-production.up.railway.ap)
  DEFAULT_API_KEY=dev-key
  ```

---

## Usage

### Upload a PDF

Use the Streamlit UI (recommended) or FastAPI:

* `POST /upload?ingest=true` (upload + ingest into Qdrant)
* `POST /upload?ingest=false` (upload only)
* `POST /ingest/{file_id}` (ingest later)

### Ask a question

* `POST /chat` (non-stream)
* `POST /chat/stream` (SSE streaming)

### Documents

* `GET /documents`

---

## Notes / Troubleshooting

### Qdrant collection not found

If you see “Collection doesn't exist”, the backend will auto-create it on first use.
Make sure `QDRANT_URL` is correct and Qdrant service is running.

### Gemini quota (429 RESOURCE_EXHAUSTED)

Large PDFs can hit free-tier embedding rate limits.
Options:

* Retry after the suggested delay
* Reduce chunks (increase `CHUNK_SIZE`, reduce `CHUNK_OVERLAP`)
* Use billing / higher quota
* Upload with `ingest=false` and ingest later

---

## License

MIT (or update as needed)

```
::contentReference[oaicite:0]{index=0}
```
