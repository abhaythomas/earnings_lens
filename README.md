# 📊 EarningsLens — Adaptive RAG for Earnings Call Analysis


🔗 **[Live Demo](https://earningslens-b4fjpwzptyqempqmzuhuff.streamlit.app/)**

![EarningsLens Demo](screenshots/demo.png)

An intelligent Q&A system that lets you ask questions about earnings call transcripts and SEC filings and get grounded, cited answers. Built with **LangGraph** for agentic self-correcting retrieval.

Unlike basic RAG (retrieve → generate), EarningsLens uses an **Adaptive RAG** architecture that verifies its own answers and self-corrects when retrieval or generation fails.

## 🏗️ Architecture

```
User Question
     │
     ▼
┌──────────┐    general question    ┌───────────────┐
│  Router   │ ─────────────────────▶│ Direct Answer  │──▶ END
└─────┬─────┘                       └───────────────┘
      │ needs retrieval
      ▼
┌──────────┐
│ Retrieve  │ ◄─────────────────────────────────────────┐
└─────┬─────┘                                           │
      ▼                                                 │
┌──────────┐   no relevant docs   ┌──────────────┐     │
│  Grade    │ ───────────────────▶│ Rewrite Query │─────┘
│ Documents │                     └──────────────┘
└─────┬─────┘
      │ relevant docs found
      ▼
┌──────────┐
│ Generate  │
│  Answer   │
└─────┬─────┘
      ▼
┌──────────┐   hallucination detected   ┌──────────────┐
│  Check    │ ─────────────────────────▶│ Rewrite Query │──▶ Retrieve (loop)
│ Grounded  │                           └──────────────┘
└─────┬─────┘
      │ grounded ✅
      ▼
┌──────────┐   not useful   ┌──────────────┐
│  Check    │ ─────────────▶│ Rewrite Query │──▶ Retrieve (loop)
│ Useful    │               └──────────────┘
└─────┬─────┘
      │ useful ✅
      ▼
  Final Answer (with source citations)
```

**Why this matters:** In financial analysis, hallucinated numbers or misattributed quotes are dangerous. The self-correcting loop ensures answers are both grounded in source documents and actually useful.

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- A [Groq API key](https://console.groq.com) (free)

### Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/earningslens.git
cd earningslens

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure API key
cp .env.example .env
# Edit .env and add your Groq API key
```

### Add Data

Drop documents into the `data/` folder — the ingestion pipeline handles both formats automatically.

| Format | Source | Example |
|--------|--------|---------|
| `.txt` | Earnings call transcripts | Motley Fool, Seeking Alpha |
| `.pdf` | SEC filings (10-Q, 10-K) | Company IR pages, SEC EDGAR |

Free sources:
- [The Motley Fool - Earnings Call Transcripts](https://www.fool.com/earnings-call-transcripts/)
- [Seeking Alpha](https://seekingalpha.com/earnings/earnings-call-transcripts)
- [SEC EDGAR Full-Text Search](https://efts.sec.gov/LATEST/search-index?forms=10-Q) — search for 10-Q or 10-K filings

### Run

```bash
# Step 1: Ingest documents (run once, or when you add new files)
python ingest.py

# Step 2: Launch the app
streamlit run app.py
```

## 🛠️ Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Orchestration | **LangGraph** | Graph-based agentic workflows with conditional routing |
| LLM | **Groq (Llama 3.3 70B)** | Free, fast inference — no local GPU needed |
| Embeddings | **HuggingFace (all-MiniLM-L6-v2)** | Free, local, ~90MB model |
| Vector Store | **ChromaDB** | Local, no cloud dependency, easy setup |
| Frontend | **Streamlit** | Rapid prototyping, chat interface |
| Documents | **LangChain + pdfplumber** | Document loading, chunking, retrieval, PDF table extraction |

## 📁 Project Structure

```
earningslens/
├── README.md              # You are here
├── requirements.txt       # Python dependencies
├── .env.example           # API key template
├── .gitignore             # Protects API keys and vector DB
├── ingest.py              # Document ingestion pipeline (txt + pdf)
├── nodes.py               # LangGraph node functions
├── graph.py               # LangGraph workflow definition
├── app.py                 # Streamlit frontend
└── data/                  # Drop .txt transcripts and .pdf filings here
    └── README.md          # Instructions for adding documents
```

## 💡 Key Design Decisions

**Adaptive RAG over basic RAG:** A naive retrieve-then-generate pipeline has no way to know when it fails. Our graph-based approach adds three verification layers (document relevance, hallucination, usefulness) and self-corrects by rewriting the query and retrying — up to 3 attempts.

**Groq over local models:** While this project could run fully locally with Ollama, Groq provides access to Llama 3.3 70B for free — a much more capable model than what most consumer hardware can run. The architecture is LLM-agnostic; swap `ChatGroq` for `ChatOllama` to run locally.

**ChromaDB for simplicity:** No cloud vector database needed. ChromaDB persists to disk and loads on startup. For a production system, you'd migrate to Pinecone, Weaviate, or pgvector.

**Separate grading and hallucination checks:** These are distinct failure modes. A document can be retrieved but irrelevant (grading catches this). An answer can use relevant documents but still hallucinate details (hallucination check catches this). Separating them gives more precise self-correction.

**pdfplumber for PDF ingestion:** Unlike basic PDF parsers, pdfplumber has table-aware extraction — it reconstructs financial tables (revenue breakdowns, balance sheets) as readable row/column strings rather than garbled text. Boilerplate pages (cover, table of contents) are automatically skipped. Each page is stored with its page number in metadata so citations are precise: `Apple_10Q.pdf (p. 4)`.

## 🔮 Potential Extensions

- [x] ~~Add PDF ingestion (SEC filings, annual reports)~~ ✅
- [x] ~~Multi-company comparison queries~~ ✅
- [ ] Time-series analysis across quarterly calls
- [ ] Financial entity extraction (revenue, EPS, guidance numbers)
- [ ] Deployment to Hugging Face Spaces or Streamlit Cloud

## 📜 License

MIT
