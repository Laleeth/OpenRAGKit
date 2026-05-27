# RAG Starter Kit

Production-ready starter kit for building retrieval-augmented generation apps over PDFs, Markdown, text, DOCX, HTML, and web pages.

## What is included

- FastAPI backend with clean modular architecture
- Persistent Chroma vector store
- SentenceTransformers embeddings
- PyMuPDF / python-docx / BeautifulSoup document ingestion
- Chunking, indexing, search, and chat endpoints
- OpenAI-compatible LLM provider support
- Built-in extractive fallback when no API key is set
- Gradio UI for quick demos
- SQLite metadata store
- Docker + docker-compose
- Seed script and tests

## Architecture

- `app/api` — HTTP routes
- `app/services` — ingestion, chunking, embeddings, retrieval, RAG, evaluation
- `app/db` — SQLite models and session handling
- `ui/gradio_app.py` — local demo UI
- `scripts/seed_demo.py` — sample data ingestion
- `tests/` — unit tests

## Quick start

### 1) Create an environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Configure env

```bash
cp .env.example .env
```

Edit `.env` if needed.

### 4) Run the API

```bash
uvicorn ragstarter.main:app --reload
```

### 5) Run the UI

```bash
python -m ragstarter.ui.gradio_app
```

## Docker

```bash
docker compose up --build
```

## Typical workflow

1. Upload documents through `POST /documents/upload`
2. Search with `POST /search`
3. Chat with `POST /chat`
4. Inspect metrics with `POST /evaluate`

## API examples

### Upload a document

```bash
curl -X POST "http://localhost:8000/documents/upload"   -F "file=@./examples/sample_policy.md"
```

### Search

```bash
curl -X POST "http://localhost:8000/search"   -H "Content-Type: application/json"   -d '{"query":"What is the refund policy?","top_k":5}'
```

### Chat

```bash
curl -X POST "http://localhost:8000/chat"   -H "Content-Type: application/json"   -d '{"question":"What is the refund policy?"}'
```

## LLM provider setup

This project works out of the box with an extractive fallback, so it runs without any API key.

For stronger answers, set an OpenAI-compatible endpoint:

```env
LLM_PROVIDER=openai_compatible
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=...
LLM_MODEL=gpt-4o-mini
```

It also supports local Ollama:

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=llama3.1
```

## Why this starter kit is useful

Most RAG demos stop at “upload PDF and ask questions.” This code adds the pieces that matter in practice:

- persistent storage
- document tracking
- chunk inspection
- metadata-aware retrieval
- citations
- evaluation hooks
- production-friendly structure

## Notes

- Embeddings are generated with SentenceTransformers.
- PDFs are parsed with PyMuPDF.
- Chroma stores vectors persistently on disk.
- SQLite stores document metadata and ingestion status.
