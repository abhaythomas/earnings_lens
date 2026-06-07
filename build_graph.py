"""
build_graph.py — Offline Knowledge Graph Builder for EarningsLens GraphRAG

Reads local documents from data/, extracts entities and relationships via the
Groq LLM, and saves a NetworkX DiGraph to graph.json.

Run this once after ingestion (or after adding new documents):

    python build_graph.py              # Build / update graph from data/
    python build_graph.py --rebuild    # Force full rebuild even if graph.json exists

graph.json is read at query time by retrieve_graph() in nodes_v2.py.
"""

import os
import re
import sys
import json
import time
import argparse
from datetime import datetime, timezone

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import networkx as nx
import pdfplumber
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ── Constants ────────────────────────────────────────────────────────────────

GRAPH_PATH = "graph.json"
PROGRESS_PATH = "graph_build_progress.json"
DATA_DIR = "data"

GROQ_MODEL = "llama-3.3-70b-versatile"
CHUNKS_PER_FILE = 5      # Chunks sampled per document (1 LLM call per file)
BATCH_SIZE = 5           # Chunks per LLM call
RATE_LIMIT_SLEEP = 2.5   # Seconds between LLM calls (Groq free tier: ~30 req/min)

# Fixed theme vocabulary — LLM must pick from these exactly
THEME_VOCAB = [
    "AI adoption",
    "cloud growth",
    "supply chain",
    "consumer demand",
    "macro headwinds",
    "regulatory risk",
    "margin expansion",
    "margin compression",
    "international expansion",
    "cost cutting",
    "share buyback",
    "guidance raised",
    "guidance lowered",
    "M&A activity",
    "capex increase",
]

# Company name → ticker (for normalizing competitor mentions)
COMPANY_TICKER_MAP = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "amazon": "AMZN",
    "nvidia": "NVDA",
    "tesla": "TSLA",
    "alphabet": "GOOGL",
    "google": "GOOGL",
    "meta": "META",
    "netflix": "NFLX",
    "salesforce": "CRM",
    "adobe": "ADBE",
    "intel": "INTC",
    "amd": "AMD",
    "qualcomm": "QCOM",
    "broadcom": "AVGO",
    "oracle": "ORCL",
    "ibm": "IBM",
    "cisco": "CSCO",
}

# Chunker — same params as ingest_v2.py for consistency
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""],
)


# ── Filename parsing ─────────────────────────────────────────────────────────

def parse_filename(filename: str) -> dict:
    """
    Extract ticker, company, year, quarter from a filename.
    Handles formats like:
      - "Apple (AAPL) Q1 2025 Earnings Call Transcript.txt"
      - "10Q-Q2-2025-Apple.pdf"
    """
    base = os.path.splitext(filename)[0]

    ticker_match = re.search(r'\(([A-Z]{2,5})\)', base)
    ticker = ticker_match.group(1) if ticker_match else None

    year_match = re.search(r'20[2-9]\d', base)
    year = year_match.group(0) if year_match else None

    quarter_match = re.search(r'Q[1-4]', base, re.IGNORECASE)
    quarter = quarter_match.group(0).upper() if quarter_match else None

    # Company name: text before the ticker/quarter/year marker
    company = re.split(r'\s*[\(\-]', base)[0].strip()

    # PDF format: "10Q-Q2-2025-Apple"
    if not ticker and "Apple" in base:
        ticker = "AAPL"
    if not company or company.startswith("10"):
        company_match = re.search(r'-([\w\s]+)$', base)
        if company_match:
            company = company_match.group(1).strip()

    return {
        "ticker": ticker or "UNKNOWN",
        "company": company or filename,
        "year": year or "unknown",
        "quarter": quarter or "unknown",
    }


# ── File loading ─────────────────────────────────────────────────────────────

def load_file_text(filepath: str) -> str:
    """Load a .txt or .pdf file and return its raw text."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".txt":
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    elif ext == ".pdf":
        pages = []
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                if len(text) >= 100:
                    pages.append(text)
        return "\n\n".join(pages)
    return ""


# ── Entity extraction ────────────────────────────────────────────────────────

def _parse_json_response(raw: str) -> dict | None:
    """Parse JSON from an LLM response, stripping markdown fences if present."""
    text = raw.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s*```\s*$', '', text, flags=re.MULTILINE)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return None


_llm = None

def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGroq(model=GROQ_MODEL, temperature=0)
    return _llm


def extract_entities(chunks: list[str], ticker: str, period: str) -> dict | None:
    """
    Run one LLM call to extract entities from a batch of text chunks.
    Builds messages directly (no ChatPromptTemplate) to avoid curly-brace
    conflicts between the JSON schema example and LangChain's template engine.
    Returns parsed dict or None on failure.
    """
    from langchain_core.messages import SystemMessage, HumanMessage

    themes_str = ", ".join(THEME_VOCAB)
    chunks_text = "\n---\n".join(
        f"[Chunk {i+1}]\n{chunk[:600]}" for i, chunk in enumerate(chunks)
    )

    system_text = (
        "You are a financial knowledge graph extractor. "
        "Extract entities from earnings call transcript chunks.\n\n"
        "Return ONLY valid JSON matching this schema exactly:\n"
        '{\n'
        '  "executives": [{"name": "string", "role": "CEO|CFO|COO|CTO|other"}],\n'
        '  "metrics": [{"name": "revenue|gross_margin|eps|operating_income|net_income|guidance|capex|other",'
        ' "value": "string with unit e.g. 94.9B or 46.2%", "period": "string e.g. Q1 2025"}],\n'
        '  "products": [{"name": "string"}],\n'
        f'  "themes": ["must be one of: {themes_str}"],\n'
        '  "competitor_mentions": [{"ticker": "stock ticker string", "context": "one sentence"}]\n'
        '}\n\n'
        "Rules:\n"
        "- Only extract what is EXPLICITLY stated, never infer\n"
        "- For themes, choose from the list only; omit if none apply\n"
        "- For competitor_mentions, only include named companies with stock tickers\n"
        "- Return empty arrays [] if nothing found for a category\n"
        "- Keep metric values concise: prefer '94.9B' over '$94.9 billion'"
    )
    human_text = f"Company: {ticker} | Period: {period}\n\nChunks:\n{chunks_text}"

    try:
        response = _get_llm().invoke([
            SystemMessage(content=system_text),
            HumanMessage(content=human_text),
        ])
        return _parse_json_response(response.content)
    except Exception as e:
        print(f"   [WARN]  LLM extraction failed: {e}")
        return None


# ── Graph population ─────────────────────────────────────────────────────────

def _normalize_ticker_mention(raw: str) -> str | None:
    """Normalize a competitor mention to a known ticker."""
    clean = raw.strip().upper()
    if re.match(r'^[A-Z]{1,5}$', clean):
        return clean
    lower = raw.strip().lower()
    return COMPANY_TICKER_MAP.get(lower)


def add_entities_to_graph(G: nx.DiGraph, ticker: str, doc_id: str, entities: dict):
    """Add extracted entities from one document to the graph."""
    # Executives
    for exec_info in entities.get("executives", []) or []:
        name = (exec_info.get("name") or "").strip()
        role = (exec_info.get("role") or "").strip()
        if not name:
            continue
        exec_id = f"exec|{name.lower()}|{ticker}"
        if not G.has_node(exec_id):
            G.add_node(exec_id, type="executive", name=name, role=role, company_ticker=ticker)
        G.add_edge(doc_id, exec_id, type="speaker", role=role)

    # Metrics
    for metric in entities.get("metrics", []) or []:
        name = (metric.get("name") or "").strip()
        value = (metric.get("value") or "").strip()
        period = (metric.get("period") or "").strip()
        if not name or not value:
            continue
        metric_id = f"metric|{name.lower()}|{ticker}|{period}"
        if not G.has_node(metric_id):
            G.add_node(metric_id, type="metric", name=name, value=value, period=period, ticker=ticker)
        G.add_edge(doc_id, metric_id, type="reports", value=value)

    # Products
    for product in entities.get("products", []) or []:
        name = (product.get("name") or "").strip()
        if not name or len(name) > 60:
            continue
        product_id = f"product|{name.lower()}|{ticker}"
        if not G.has_node(product_id):
            G.add_node(product_id, type="product", name=name, company_ticker=ticker)
        G.add_edge(doc_id, product_id, type="mentions")

    # Themes
    for theme_raw in entities.get("themes", []) or []:
        theme = (theme_raw or "").strip()
        if theme not in THEME_VOCAB:
            continue
        theme_id = f"theme|{theme.lower().replace(' ', '_')}"
        if not G.has_node(theme_id):
            G.add_node(theme_id, type="theme", name=theme)
        G.add_edge(doc_id, theme_id, type="discusses")
        # Company-level theme edge
        if not G.has_edge(ticker, theme_id):
            G.add_edge(ticker, theme_id, type="theme_of", frequency=0)
        G[ticker][theme_id]["frequency"] = G[ticker][theme_id].get("frequency", 0) + 1

    # Competitor mentions
    for comp in entities.get("competitor_mentions", []) or []:
        raw_ticker = (comp.get("ticker") or "").strip()
        context = (comp.get("context") or "").strip()
        comp_ticker = _normalize_ticker_mention(raw_ticker)
        if not comp_ticker or comp_ticker == ticker:
            continue
        if not G.has_node(comp_ticker):
            G.add_node(comp_ticker, type="company", ticker=comp_ticker, name=comp_ticker, aliases=[comp_ticker])
        if not G.has_edge(ticker, comp_ticker):
            G.add_edge(ticker, comp_ticker, type="competes_with", evidence=context[:120])


# ── Main build logic ─────────────────────────────────────────────────────────

def _sample_chunks(chunks: list, n: int) -> list:
    """Pick n evenly-spaced chunks from the list."""
    if len(chunks) <= n:
        return chunks
    step = len(chunks) / n
    return [chunks[int(i * step)] for i in range(n)]


def _load_progress() -> set:
    """Load set of already-processed doc IDs (for resuming)."""
    if os.path.exists(PROGRESS_PATH):
        try:
            with open(PROGRESS_PATH, "r") as f:
                return set(json.load(f).get("processed", []))
        except Exception:
            pass
    return set()


def _save_progress(processed: set):
    with open(PROGRESS_PATH, "w") as f:
        json.dump({"processed": list(processed)}, f)


def build_graph(data_dir: str = DATA_DIR, rebuild: bool = False) -> nx.DiGraph:
    """
    Build a knowledge graph from all .txt and .pdf files in data_dir.
    Saves result to graph.json and returns the graph.
    """
    G = nx.DiGraph()

    # Load existing graph if not rebuilding
    processed = set()
    if not rebuild and os.path.exists(GRAPH_PATH):
        print(f"📂 Loading existing graph from {GRAPH_PATH}...")
        with open(GRAPH_PATH, "r", encoding="utf-8") as f:
            G = nx.node_link_graph(json.load(f))
        processed = _load_progress()
        print(f"   {G.number_of_nodes()} nodes, {G.number_of_edges()} edges already in graph")
        print(f"   {len(processed)} files already processed")

    # Discover local files
    supported_exts = {".txt", ".pdf"}
    files = [
        f for f in os.listdir(data_dir)
        if os.path.splitext(f)[1].lower() in supported_exts
    ]
    print(f"\n[FILE] Found {len(files)} files in {data_dir}/")

    new_count = 0
    for filename in sorted(files):
        if filename in processed:
            print(f"   [SKIP]  Skipping (already processed): {filename}")
            continue

        filepath = os.path.join(data_dir, filename)
        meta = parse_filename(filename)
        ticker = meta["ticker"]
        period = f"{meta['quarter']} {meta['year']}"
        doc_id = f"doc|{ticker}|{meta['year']}|{meta['quarter']}|{os.path.splitext(filename)[0][:30]}"

        print(f"\n[BUILD] Processing: {filename}")
        print(f"   Ticker={ticker}, Period={period}")

        # Load and chunk the file
        raw_text = load_file_text(filepath)
        if not raw_text.strip():
            print(f"   [WARN]  Empty file, skipping")
            processed.add(filename)
            _save_progress(processed)
            continue

        chunks = _splitter.split_text(raw_text)
        sampled = _sample_chunks(chunks, CHUNKS_PER_FILE)
        print(f"   {len(chunks)} total chunks → sampling {len(sampled)}")

        # Ensure company node exists
        if not G.has_node(ticker):
            G.add_node(ticker, type="company", ticker=ticker, name=meta["company"], aliases=[meta["company"], ticker])
        else:
            # Merge aliases
            existing_aliases = G.nodes[ticker].get("aliases", [])
            if meta["company"] not in existing_aliases:
                existing_aliases.append(meta["company"])
            G.nodes[ticker]["aliases"] = existing_aliases

        # Ensure document node exists
        if not G.has_node(doc_id):
            G.add_node(doc_id, type="document", ticker=ticker,
                       year=meta["year"], quarter=meta["quarter"],
                       doc_type="pdf" if filename.endswith(".pdf") else "transcript",
                       source=filename, chunk_count=len(chunks))
        G.add_edge(ticker, doc_id, type="filed", year=meta["year"], quarter=meta["quarter"])

        # LLM entity extraction
        entities = extract_entities(sampled, ticker, period)
        if entities:
            add_entities_to_graph(G, ticker, doc_id, entities)
            execs = len(entities.get("executives") or [])
            metrics = len(entities.get("metrics") or [])
            themes = len(entities.get("themes") or [])
            comps = len(entities.get("competitor_mentions") or [])
            print(f"   [OK] Extracted: {execs} execs, {metrics} metrics, {themes} themes, {comps} competitors")
        else:
            print(f"   [WARN]  Entity extraction returned nothing — structural nodes only")

        processed.add(filename)
        _save_progress(processed)
        new_count += 1

        # Rate limit guard
        time.sleep(RATE_LIMIT_SLEEP)

    # Build temporal chains (next_quarter edges within each company)
    _add_temporal_edges(G)

    # Save graph
    data = nx.node_link_data(G)
    data["built_at"] = datetime.now(timezone.utc).isoformat()
    data["doc_count"] = len([n for n, d in G.nodes(data=True) if d.get("type") == "document"])

    with open(GRAPH_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"\n[OK] Graph saved to {GRAPH_PATH}")
    print(f"   {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"   {new_count} new files processed")

    # Clean up progress file on full success
    if os.path.exists(PROGRESS_PATH):
        os.remove(PROGRESS_PATH)

    return G


def _add_temporal_edges(G: nx.DiGraph):
    """
    Add next_quarter edges between consecutive document nodes for each company.
    E.g.: AAPL Q4 2024 → AAPL Q1 2025
    """
    QUARTER_ORDER = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}

    # Group document nodes by ticker
    company_docs: dict[str, list] = {}
    for node_id, data in G.nodes(data=True):
        if data.get("type") != "document":
            continue
        ticker = data.get("ticker")
        year = data.get("year")
        quarter = data.get("quarter")
        if ticker and year and quarter and quarter in QUARTER_ORDER:
            company_docs.setdefault(ticker, []).append((node_id, int(year), QUARTER_ORDER[quarter]))

    for ticker, docs in company_docs.items():
        sorted_docs = sorted(docs, key=lambda x: (x[1], x[2]))
        for i in range(len(sorted_docs) - 1):
            curr_id = sorted_docs[i][0]
            next_id = sorted_docs[i + 1][0]
            if not G.has_edge(curr_id, next_id):
                G.add_edge(curr_id, next_id, type="next_quarter")


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build EarningsLens knowledge graph")
    parser.add_argument("--rebuild", action="store_true", help="Force full rebuild")
    parser.add_argument("--data-dir", default=DATA_DIR, help="Directory with data files")
    args = parser.parse_args()

    G = build_graph(data_dir=args.data_dir, rebuild=args.rebuild)

    # Print summary
    node_types = {}
    for _, d in G.nodes(data=True):
        t = d.get("type", "unknown")
        node_types[t] = node_types.get(t, 0) + 1
    print("\nNode type breakdown:")
    for t, count in sorted(node_types.items()):
        print(f"  {t}: {count}")
