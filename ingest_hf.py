"""
ingest_hf.py — HuggingFace Dataset Ingestion for EarningsLens v2

Pulls pre-curated earnings call transcripts from HuggingFace Hub and ingests
them into the same Pinecone index (and manifest) used by ingest_v2.py.

Recommended datasets:
  Bose345/sp500_earnings_transcripts  — 33k transcripts, MIT, 2005–2025
  glopardo/sp500-earnings-transcripts — 20k transcripts, richer financial metadata

Usage:
  # Specific tickers, specific years
  python ingest_hf.py --dataset Bose345/sp500_earnings_transcripts \\
                      --tickers AAPL MSFT NVDA AMZN \\
                      --years 2023 2024 2025

  # One ticker, all years
  python ingest_hf.py --dataset Bose345/sp500_earnings_transcripts --tickers TSLA

  # Ingest everything (33k rows — takes a while)
  python ingest_hf.py --dataset Bose345/sp500_earnings_transcripts

  # Use the richer metadata dataset
  python ingest_hf.py --dataset glopardo/sp500-earnings-transcripts \\
                      --tickers NVDA AAPL --years 2024 2025

Manifest tracking:
  Each (dataset, ticker, year, quarter) tuple is written to
  data/ingested_manifest.json after a successful upsert. Re-running is
  incremental — only missing tuples are ingested. Shares the manifest with
  the file-based ingest_v2.py so both sources are tracked in one place.
"""

import os
import sys
import hashlib
import json
import argparse
from datetime import datetime, timezone

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

# ── Config (must match ingest_v2.py) ────────────────────────────────
MANIFEST_FILE = "data/ingested_manifest.json"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
PINECONE_INDEX_NAME = "earningslens-v2"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"

# ── Dataset field mappings ───────────────────────────────────────────
#
# Each entry describes how to map a HuggingFace dataset's columns to the
# metadata fields EarningsLens expects. Add new datasets here — no other
# code changes needed.
#
DATASET_CONFIGS: dict[str, dict] = {
    "Bose345/sp500_earnings_transcripts": {
        # Column that holds the full transcript text
        "text_field": "content",
        # Metadata columns (None → field doesn't exist in this dataset)
        "ticker_field": "symbol",
        "company_field": "company_name",
        "year_field": "year",
        "quarter_field": "quarter",   # integer 1–4
        "date_field": "date",
        # Extra columns to carry through as metadata (optional)
        "extra_fields": [],
        "doc_type": "transcript",
    },
    "glopardo/sp500-earnings-transcripts": {
        "text_field": "transcript",
        "ticker_field": "ticker",
        "company_field": "company",
        "year_field": "year",
        "quarter_field": "quarter",   # integer 1–4
        "date_field": "earnings_date",
        # This dataset also carries sector/industry — useful for filtering
        "extra_fields": ["sector", "industry"],
        "doc_type": "transcript",
    },
}


# ── Manifest (shared with ingest_v2.py) ─────────────────────────────

def load_manifest() -> dict:
    if os.path.exists(MANIFEST_FILE):
        with open(MANIFEST_FILE, "r") as f:
            return json.load(f)
    return {}


def save_manifest(manifest: dict):
    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2)


def hf_manifest_key(dataset_name: str, ticker: str, year: int, quarter: int) -> str:
    """
    Unique key per (dataset, ticker, year, quarter) tuple.
    Granularity: one quarter of one company from one dataset.
    This means adding Q1 2026 data is incremental — prior quarters aren't re-ingested.
    """
    return f"hf::{dataset_name}::{ticker}::{year}::Q{quarter}"


# ── Chunk IDs ────────────────────────────────────────────────────────

def make_hf_chunk_id(dataset_name: str, ticker: str, year: int, quarter: int, i: int) -> str:
    """
    Stable chunk ID for HF-sourced chunks.
    Format: sha256(dataset::ticker::year::quarter)[:12]-chunk-{i:04d}
    Idempotent: re-ingesting the same (ticker, year, quarter) overwrites existing vectors.
    """
    key = f"{dataset_name}::{ticker}::{year}::Q{quarter}"
    h = hashlib.sha256(key.encode()).hexdigest()[:12]
    return f"{h}-chunk-{i:04d}"


# ── Pinecone setup ───────────────────────────────────────────────────

def get_pinecone_index():
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY not set. Add it to .env.")

    pc = Pinecone(api_key=api_key)
    existing = [idx.name for idx in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing:
        print(f"🆕 Creating Pinecone index '{PINECONE_INDEX_NAME}'...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
    return pc.Index(PINECONE_INDEX_NAME)


# ── Row → Documents ──────────────────────────────────────────────────

def row_to_documents(row: dict, cfg: dict) -> tuple[str, int, int, list[Document]]:
    """
    Convert one dataset row to a list of Document objects (one per field chunk).

    Returns (ticker, year, quarter, [Document, ...]) so the caller can build
    the manifest key. Returns empty list if the row has no usable text.
    """
    text = row.get(cfg["text_field"], "") or ""
    text = text.strip()
    if len(text) < 200:
        return "", 0, 0, []   # skip near-empty rows

    ticker  = str(row.get(cfg["ticker_field"], "UNKNOWN")).upper().strip()
    company = str(row.get(cfg["company_field"], "")) or ticker
    year    = int(row.get(cfg["year_field"], 0) or 0)
    quarter = int(row.get(cfg["quarter_field"], 0) or 0)
    date    = str(row.get(cfg["date_field"], "") or "")

    # Build source label (mirrors the file-based pipeline's "source" field)
    source = f"{ticker}_Q{quarter}_{year}_hf_transcript.txt"

    meta = {
        "source":   source,
        "doc_type": cfg["doc_type"],
        "ticker":   ticker,
        "company":  company,
        "year":     str(year),
        "quarter":  f"Q{quarter}",
        "date":     date,
        "hf_dataset": cfg.get("_dataset_name", ""),
    }
    for field in cfg.get("extra_fields", []):
        val = row.get(field)
        if val is not None:
            meta[field] = str(val)

    doc = Document(page_content=text, metadata=meta)
    return ticker, year, quarter, [doc]


# ── Per-batch ingest ─────────────────────────────────────────────────

def ingest_batch(
    docs: list[Document],
    ticker: str,
    year: int,
    quarter: int,
    dataset_name: str,
    vectorstore: PineconeVectorStore,
    splitter: RecursiveCharacterTextSplitter,
    manifest: dict,
) -> int:
    """
    Chunk and upsert one (ticker, year, quarter) batch.
    Returns chunk count. Updates manifest on success.
    """
    chunks = splitter.split_documents(docs)
    if not chunks:
        return 0

    ids = [make_hf_chunk_id(dataset_name, ticker, year, quarter, i) for i in range(len(chunks))]
    vectorstore.add_documents(chunks, ids=ids)

    key = hf_manifest_key(dataset_name, ticker, year, quarter)
    manifest[key] = {
        "chunks": len(chunks),
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "ticker": ticker,
        "year": year,
        "quarter": f"Q{quarter}",
        "source": "huggingface",
        "dataset": dataset_name,
    }
    save_manifest(manifest)
    return len(chunks)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Ingest HuggingFace earnings datasets into EarningsLens Pinecone index."
    )
    parser.add_argument(
        "--dataset",
        default="Bose345/sp500_earnings_transcripts",
        help="HuggingFace dataset name (must be in DATASET_CONFIGS)",
    )
    parser.add_argument(
        "--tickers",
        nargs="*",
        help="Stock tickers to ingest (e.g. AAPL MSFT NVDA). Omit for all.",
    )
    parser.add_argument(
        "--years",
        nargs="*",
        type=int,
        help="Years to ingest (e.g. 2023 2024 2025). Omit for all.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after N rows (useful for testing). Omit for all rows.",
    )
    args = parser.parse_args()

    dataset_name = args.dataset
    tickers_filter = {t.upper() for t in args.tickers} if args.tickers else None
    years_filter   = set(args.years) if args.years else None

    print("=" * 60)
    print("EarningsLens — HuggingFace Dataset Ingestion")
    print("=" * 60)
    print(f"  Dataset : {dataset_name}")
    print(f"  Tickers : {sorted(tickers_filter) if tickers_filter else 'ALL'}")
    print(f"  Years   : {sorted(years_filter) if years_filter else 'ALL'}")
    if args.limit:
        print(f"  Limit   : {args.limit} rows (test mode)")
    print()

    if dataset_name not in DATASET_CONFIGS:
        print(f"❌ '{dataset_name}' is not in DATASET_CONFIGS.")
        print("   Supported datasets:")
        for name in DATASET_CONFIGS:
            print(f"     {name}")
        print("\n   To add a new dataset, add an entry to DATASET_CONFIGS in ingest_hf.py")
        return

    cfg = {**DATASET_CONFIGS[dataset_name], "_dataset_name": dataset_name}

    # ── Connect to Pinecone ──────────────────────────────────────────
    print("🔗 Connecting to Pinecone...")
    get_pinecone_index()

    manifest = load_manifest()
    print(f"📋 Manifest: {len(manifest)} entry/entries previously ingested\n")

    # ── Load models ──────────────────────────────────────────────────
    print(f"🔄 Loading embedding model: {EMBEDDING_MODEL}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )
    vectorstore = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
    )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    print()

    # ── Stream dataset from HuggingFace ─────────────────────────────
    # streaming=True means HF downloads and processes records on demand
    # rather than loading the entire dataset into RAM — safe for large datasets.
    print(f"📡 Streaming dataset from HuggingFace Hub...")
    print("   (first access may download a few MB of metadata)\n")

    dataset = load_dataset(dataset_name, split="train", streaming=True)

    # ── Group by (ticker, year, quarter) and ingest batch by batch ───
    # We accumulate docs for one (ticker, year, quarter) tuple, then chunk
    # and upsert as a batch. This keeps memory O(one call) rather than O(dataset).

    current_key = None
    current_docs: list[Document] = []
    rows_seen = 0
    rows_skipped = 0
    batches_ingested = 0
    batches_skipped = 0
    total_chunks = 0

    def flush_batch():
        nonlocal total_chunks, batches_ingested, current_docs, current_key
        if not current_docs or current_key is None:
            return
        ticker, year, quarter = current_key
        n = ingest_batch(
            current_docs, ticker, year, quarter,
            dataset_name, vectorstore, splitter, manifest,
        )
        total_chunks += n
        batches_ingested += 1
        print(f"   ✅ {ticker} Q{quarter} {year}: {n} chunks")
        current_docs = []
        current_key = None

    for row in dataset:
        rows_seen += 1

        ticker, year, quarter, docs = row_to_documents(row, cfg)

        if not docs:
            rows_skipped += 1
            continue

        # Apply filters
        if tickers_filter and ticker not in tickers_filter:
            rows_skipped += 1
            continue
        if years_filter and year not in years_filter:
            rows_skipped += 1
            continue

        # --limit counts matching rows, not all rows
        if args.limit and (batches_ingested + batches_skipped) >= args.limit:
            break

        # Skip already-ingested (ticker, year, quarter) tuples
        key = hf_manifest_key(dataset_name, ticker, year, quarter)
        if key in manifest:
            rows_skipped += 1
            batches_skipped += 1
            continue

        # Group by (ticker, year, quarter) — flush when the key changes
        batch_key = (ticker, year, quarter)
        if batch_key != current_key:
            flush_batch()
            current_key = batch_key

        current_docs.extend(docs)

    flush_batch()  # flush the last batch

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"✅ Ingested {batches_ingested} new (ticker, year, quarter) batch(es)")
    print(f"   {total_chunks} chunks added to Pinecone")
    print(f"   {batches_skipped} batch(es) skipped (already in manifest)")
    print(f"   {rows_skipped} row(s) filtered out (wrong ticker/year or empty)")
    print(f"\n🎉 Run the app: streamlit run app_v2.py")


if __name__ == "__main__":
    main()
