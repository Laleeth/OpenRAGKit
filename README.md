# OpenRAGKit

Production-ready open-source toolkit for building and debugging RAG systems.
<img width="814" height="605" alt="image" src="https://github.com/user-attachments/assets/40721f97-2b29-4194-b36a-d614ee81bcdd" />

## Features

- PDF/TXT/Markdown ingestion
- Semantic chunking pipeline
- SentenceTransformers embeddings
- ChromaDB vector storage
- FastAPI backend
- Gradio UI
- Persistent vector storage
- Docker support
- Modular RAG architecture

---

## Why RAGLab?

Most RAG starter kits stop at:

> Upload PDF → Chat

RAGLab is designed for developers who want:
- reusable RAG components,
- production-style architecture,
- observability,
- retrieval inspection,
- and future extensibility.

---

## Architecture

```text
Documents
   ↓
Extraction
   ↓
Chunking
   ↓
Embeddings
   ↓
ChromaDB
   ↓
Retriever
   ↓
LLM
   ↓
Grounded Response
```

---

## Quickstart

### Clone repository

```bash
git clone https://github.com/YOUR_USERNAME/raglab.git
cd raglab
```

### Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run backend

```bash
uvicorn app.main:app --reload
```

### Run UI

```bash
python ui/app.py
```

---

## Docker

```bash
docker compose up --build
```

---

## API Endpoints

### Health

```http
GET /health
```

### Ingest document

```http
POST /ingest
```

### Query documents

```http
POST /query
```

---

## Example Workflow

1. Upload PDF
2. Document gets chunked
3. Embeddings are generated
4. Chunks are indexed
5. Query retrieves relevant context
6. LLM generates grounded response

---

## Project Structure

```text
raglab/
│
├── app/
├── ui/
├── tests/
├── docs/
├── examples/
├── assets/
│
├── README.md
├── CONTRIBUTING.md
├── LICENSE
├── docker-compose.yml
└── requirements.txt
```

---

## Roadmap

- [x] Basic RAG pipeline
- [x] ChromaDB integration
- [x] FastAPI backend
- [ ] Hybrid search
- [ ] Retrieval inspector
- [ ] Reranking
- [ ] Hallucination detection
- [ ] Streaming responses
- [ ] Ollama integration
- [ ] Multi-user support

---

## Future Vision

RAGLab aims to evolve into:

- RAG observability platform
- evaluation toolkit
- retrieval debugger
- enterprise RAG framework

---

## Contributing

Contributions are welcome.

Please open issues and pull requests for:
- bug fixes,
- improvements,
- retrieval strategies,
- evaluation tooling.

See `CONTRIBUTING.md`.

---

## License

MIT License
