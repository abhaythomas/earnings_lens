# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Run the app (v2 — active version)
```bash
streamlit run app_v2.py
```

### Run the app (v1 — ChromaDB/local)
```bash
streamlit run app.py
```

### Ingest documents into Pinecone (v2)
```bash
python ingest_v2.py
```

### Ingest documents into ChromaDB (v1)
```bash
python ingest.py
```

### Docker (v2)
```bash
docker compose up --build
# Or without compose:
docker build -t earningslens-v2 .
docker run -p 8501:8501 --env-file .env earningslens-v2
```

## Required environment variables

| Key | Used by |
|-----|---------|
| `GROQ_API_KEY` | Both v1 and v2 — LLM inference |
| `PINECONE_API_KEY` | v2 only — vector store |
| `LANGCHAIN_API_KEY` | Optional — LangSmith tracing |
| `LANGCHAIN_TRACING_V2` | Optional — set to `"true"` to enable tracing |
| `LANGCHAIN_PROJECT` | Optional — LangSmith project name |

Copy `.env.example` → `.env` and fill in keys. On Streamlit Cloud, put keys under App Settings → Secrets.

## Architecture

The system is an **Adaptive RAG** pipeline built with LangGraph. There are two parallel versions:

| | v1 | v2 (active) |
|--|----|----|
| Vector store | ChromaDB (local `chroma_db/`) | Pinecone (cloud) |
| Entry point | `app.py` | `app_v2.py` |
| Graph | `graph.py` | `graph_v2.py` |
| Nodes | `nodes.py` | `nodes_v2.py` |
| Ingestion | `ingest.py` | `ingest_v2.py` |

**v2 is the production version.** v1 files are kept for reference. The logic in both is identical — the only difference is the vector store backend.

### LangGraph workflow (defined in `graph_v2.py`)

Three routes from `route_question`:

1. **`direct`** → `direct_answer` → END
   General knowledge questions (e.g. "what does EPS mean?")

2. **`compare`** → `compare_companies` → END
   Multi-company comparison queries. This node handles its own retrieval internally (one Pinecone search per company). **No hallucination check or usefulness check runs on this path** — the only guard is a string-match filter on retrieved chunks.

3. **`retrieve`** → `grade_documents` → `generate` → `check_hallucination` → `check_usefulness` → END
   Standard single-company / topic queries. Self-corrects via `rewrite_query` → `retrieve` loop (max 2 rewrites / `generation_count >= 2` guard).

### Node functions (`nodes_v2.py`)

Each node is a plain function `(state: dict) -> dict` that returns an updated copy of state. Key nodes:

- **`route_question`**: LLM classifier — returns `"compare"`, `"retrieve"`, or `"direct"`
- **`retrieve`**: `similarity` search, `k=5`, no metadata filtering
- **`grade_documents`**: single LLM call grades all 5 chunks at once; keeps only relevant ones
- **`generate`**: RAG generation with last 3 turns of chat history injected
- **`check_hallucination`**: LLM binary yes/no — is the answer grounded in the source docs?
- **`check_usefulness`**: LLM binary yes/no — does the answer address the question?
- **`rewrite_query`**: rewrites the query to use earnings-call vocabulary; resolves pronouns using chat history
- **`compare_companies`**: Step 1 — LLM extracts company names. Step 2 — per-company similarity search with company-name string filter fallback. Step 3 — structured side-by-side generation.

### Ingestion pipeline (`ingest_v2.py`)

- Loads `.txt` (transcripts) and `.pdf` (SEC filings) from `data/`
- PDF pages < 100 chars or matching 2+ boilerplate phrases (EDGAR headers, TOC) are skipped
- PDF tables extracted by `pdfplumber` and appended as `[TABLE]` blocks
- Chunks: `RecursiveCharacterTextSplitter`, size=1000 chars, overlap=200, separators `["\n\n", "\n", ". ", " ", ""]`
- Embedding model: `all-MiniLM-L6-v2` (HuggingFace, 384 dims, CPU-only)
- Pinecone index: cosine similarity metric, serverless on AWS `us-east-1`
- Chunk IDs: `sha256(source)[:12]-chunk-{index}` — idempotent upserts, re-running won't duplicate

### Observability (`app_v2.py`)

- Every query appended to `logs/query_log.jsonl` (timestamp, latency_ms, docs_retrieved, route, is_grounded)
- Sidebar shows live aggregate metrics (total queries, avg latency, grounded %)
- Each assistant message stores a `graph_path` list showing which nodes ran (visible in "Reasoning path" expander)
- LangSmith tracing enabled when `LANGCHAIN_TRACING_V2=true`

### Data

Drop `.txt` transcript files or `.pdf` SEC filings into `data/`. Re-run `ingest_v2.py` to index them. The app also supports real-time PDF upload via the sidebar (calls `process_and_ingest_pdf` which runs the same pipeline in-memory).

## Key constants (change these to tune behavior)

| Location | Constant | Value | Effect |
|----------|----------|-------|--------|
| `ingest_v2.py` | `CHUNK_SIZE` | 1000 | Characters per chunk |
| `ingest_v2.py` | `CHUNK_OVERLAP` | 200 | Overlap between chunks |
| `nodes_v2.py` | `GROQ_MODEL` | `llama-3.3-70b-versatile` | LLM for all nodes |
| `nodes_v2.py` | `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Embedding model |
| `nodes_v2.py` | `retrieve` | `k=5` | Chunks fetched per query |
| `nodes_v2.py` | `compare_companies` | `k=5` per company | Chunks per company in compare |
| `graph_v2.py` | `generation_count >= 2` | 2 | Max rewrite attempts before giving up |
| `app_v2.py` | `recent = history_messages[-6:]` | 6 messages | Chat history window (last 3 turns) |
