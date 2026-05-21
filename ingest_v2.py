"""
ingest_v2.py — Document Ingestion Pipeline for EarningsLens v2

Scaling changes (v2.1):
- Manifest-based skip: tracks content hash per file — re-running only processes
  new or changed files. File already in Pinecone + unchanged → skipped entirely
  (no embedding, no upsert, no wasted time).
- File-by-file processing: load → chunk → upsert one file at a time, so memory
  stays bounded regardless of how many PDFs are in data/.
- Structured metadata: extracts company, ticker, year, quarter from filenames
  and stores them on every chunk. Enables proper Pinecone metadata filtering
  instead of the brittle source-filename string-match used at small scale.
- Resumable: manifest is saved after each file, so a crash mid-run loses at
  most one file's work.

Original behavior unchanged:
- Same HuggingFace embedding model (all-MiniLM-L6-v2, 384 dims)
- Same chunking strategy (RecursiveCharacterTextSplitter)
- Same PDF extraction logic (pdfplumber + table extraction)
- Same Pinecone index, same chunk ID scheme (idempotent upserts)

Run once: python ingest_v2.py
Re-run when you add new documents — only new/changed files are processed.
"""

import os
import sys
import glob
import hashlib
import json
import re
from datetime import datetime

# Fix Windows console encoding for emoji output
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pdfplumber
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

# ── Config ──────────────────────────────────────────────────────────
DATA_DIR = "data"
MANIFEST_FILE = "data/ingested_manifest.json"  # Tracks what's been ingested
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384          # Must match the model — all-MiniLM-L6-v2 = 384
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
PINECONE_INDEX_NAME = "earningslens-v2"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"

BOILERPLATE_PHRASES = [
    "edgar online",
    "united states securities and exchange commission",
    "form 10-q",
    "form 10-k",
    "table of contents",
]


# ── Pinecone setup ───────────────────────────────────────────────────

def get_pinecone_index():
    """Connect to Pinecone and ensure the index exists."""
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise ValueError(
            "PINECONE_API_KEY not found. "
            "Add it to your .env file or Streamlit secrets."
        )

    pc = Pinecone(api_key=api_key)
    existing_indexes = [idx.name for idx in pc.list_indexes()]

    if PINECONE_INDEX_NAME not in existing_indexes:
        print(f"🆕 Creating Pinecone index '{PINECONE_INDEX_NAME}' "
              f"({EMBEDDING_DIM} dims, cosine metric)...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
        print("   ✅ Index created.")
    else:
        print(f"✅ Pinecone index '{PINECONE_INDEX_NAME}' already exists.")

    return pc.Index(PINECONE_INDEX_NAME)


def make_chunk_id(source: str, chunk_index: int) -> str:
    """
    Stable, unique ID per chunk — sha256(source)[:12]-chunk-{index}.
    Pinecone upserts are idempotent by ID, so re-running is safe.
    """
    source_hash = hashlib.sha256(source.encode()).hexdigest()[:12]
    return f"{source_hash}-chunk-{chunk_index:04d}"


# ── Manifest ─────────────────────────────────────────────────────────

def load_manifest() -> dict:
    """
    Load the ingestion manifest from disk.

    The manifest maps filename → {hash, chunks, ingested_at, company, ticker, ...}.
    It serves two purposes:
    1. Skip logic: if a file's content hash matches the manifest, skip it.
    2. Company listing: nodes_v2.get_loaded_companies_v2() reads this instead
       of probing Pinecone with k=100 (which misses things at scale).
    """
    if os.path.exists(MANIFEST_FILE):
        with open(MANIFEST_FILE, "r") as f:
            return json.load(f)
    return {}


def save_manifest(manifest: dict):
    """Write manifest to disk. Called after each file so crashes are resumable."""
    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2)


def file_content_hash(filepath: str) -> str:
    """SHA-256 of file bytes. Changes when the file changes."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


# ── Filename metadata extraction ──────────────────────────────────────

def parse_filename_metadata(filename: str) -> dict:
    """
    Extract structured metadata from a filename.

    Handles naming conventions like:
      "Apple (AAPL) Q1 2025 Earnings Call Transcript.txt"
      "Nvidia (NVDA) Q4 2025 Earnings Call Transcript.txt"
      "10Q-Q2-2025-Apple.pdf"

    Returns: {company, ticker, year, quarter} — any field may be None.

    These are stored on every chunk so Pinecone metadata filtering can target
    specific companies/quarters without relying on filename string-matching.
    """
    name = os.path.splitext(filename)[0]

    # Ticker: (AAPL), (MSFT), (NVDA) etc.
    ticker_match = re.search(r'\(([A-Z]{1,5})\)', name)
    ticker = ticker_match.group(1) if ticker_match else None

    # Quarter: Q1–Q4
    quarter_match = re.search(r'\bQ([1-4])\b', name, re.IGNORECASE)
    quarter = f"Q{quarter_match.group(1)}" if quarter_match else None

    # Year: 2020–2029
    year_match = re.search(r'\b(20[2-9]\d)\b', name)
    year = year_match.group(1) if year_match else None

    # Company: text before the first '(' or first quarter marker
    company = None
    company_match = re.match(
        r'^([A-Za-z][A-Za-z\s\.]+?)(?:\s*\(|\s+Q[1-4]\b|\s+10[KQ]\b)',
        name,
    )
    if company_match:
        company = company_match.group(1).strip()
    elif ticker_match:
        company = name[:ticker_match.start()].strip()

    return {"company": company, "ticker": ticker, "year": year, "quarter": quarter}


# ── Document loading — per file ──────────────────────────────────────

def _format_table(table: list[list]) -> str:
    """Convert pdfplumber table rows to a readable string."""
    rows = []
    for row in table:
        cleaned = [str(cell).strip() if cell else "-" for cell in row]
        rows.append(" | ".join(cleaned))
    return "\n".join(rows)


def _is_boilerplate(text: str) -> bool:
    """Return True if the page is mostly boilerplate (EDGAR headers, TOC, etc.)."""
    lowered = text.lower()
    hits = sum(1 for phrase in BOILERPLATE_PHRASES if phrase in lowered)
    return hits >= 2


def load_single_txt(filepath: str) -> list[Document]:
    """Load one .txt file and attach structured metadata."""
    filename = os.path.basename(filepath)
    extra_meta = parse_filename_metadata(filename)

    loader = TextLoader(filepath, encoding="utf-8")
    docs = loader.load()
    for doc in docs:
        doc.metadata["source"] = filename
        doc.metadata["doc_type"] = "transcript"
        doc.metadata.update({k: v for k, v in extra_meta.items() if v is not None})
    return docs


def load_single_pdf(filepath: str) -> list[Document]:
    """Load one .pdf with pdfplumber and attach structured metadata."""
    filename = os.path.basename(filepath)
    extra_meta = parse_filename_metadata(filename)
    documents = []
    skipped = 0

    with pdfplumber.open(filepath) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            raw_text = page.extract_text() or ""
            tables = page.extract_tables()
            table_text = ""
            if tables:
                formatted = [_format_table(t) for t in tables if t]
                table_text = "\n\n[TABLE]\n" + "\n\n[TABLE]\n".join(formatted)

            full_text = raw_text + table_text

            if len(full_text.strip()) < 100:
                skipped += 1
                continue
            if _is_boilerplate(full_text):
                skipped += 1
                continue

            meta = {"source": filename, "page_number": page_num, "doc_type": "pdf"}
            meta.update({k: v for k, v in extra_meta.items() if v is not None})
            documents.append(Document(page_content=full_text, metadata=meta))

    if skipped:
        print(f"      ({skipped} page(s) skipped — too short or boilerplate)")
    return documents


# ── Per-file ingest ───────────────────────────────────────────────────

def process_file(
    filepath: str,
    vectorstore: PineconeVectorStore,
    manifest: dict,
) -> int:
    """
    Load, chunk, and upsert one file into Pinecone.

    Returns the number of chunks upserted (0 if skipped or errored).
    Saves the manifest immediately after a successful upsert so the run
    is resumable — a crash loses at most one file's work.
    """
    filename = os.path.basename(filepath)
    content_hash = file_content_hash(filepath)

    # Skip if content hash matches what's already in Pinecone
    if manifest.get(filename, {}).get("hash") == content_hash:
        print(f"   ⏭️  {filename}  (unchanged, skipping)")
        return 0

    ext = os.path.splitext(filepath)[1].lower()
    try:
        if ext == ".txt":
            docs = load_single_txt(filepath)
        elif ext == ".pdf":
            docs = load_single_pdf(filepath)
        else:
            return 0
    except Exception as e:
        print(f"   ⚠️  {filename}: load error — {e}")
        return 0

    if not docs:
        print(f"   ⚠️  {filename}: no content extracted")
        return 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    # Stable IDs — idempotent upserts if the file is reprocessed
    ids = [make_chunk_id(filename, i) for i in range(len(chunks))]

    vectorstore.add_documents(chunks, ids=ids)

    # Record in manifest and save immediately (crash-safe)
    extra_meta = parse_filename_metadata(filename)
    manifest[filename] = {
        "hash": content_hash,
        "chunks": len(chunks),
        "ingested_at": datetime.utcnow().isoformat() + "Z",
        **{k: v for k, v in extra_meta.items() if v is not None},
    }
    save_manifest(manifest)

    print(f"   ✅ {filename}: {len(chunks)} chunks → Pinecone")
    return len(chunks)


# ── Entry point ───────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("EarningsLens v2 — Document Ingestion Pipeline (Pinecone)")
    print("=" * 60 + "\n")

    # Step 0: Connect to Pinecone
    print("🔗 Connecting to Pinecone...")
    index = get_pinecone_index()

    # Step 1: Load manifest (what's already been ingested)
    manifest = load_manifest()
    print(f"📋 Manifest: {len(manifest)} file(s) previously ingested\n")

    # Step 2: Load embedding model once — reused for all files
    print(f"🔄 Loading embedding model: {EMBEDDING_MODEL}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )
    vectorstore = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
    )
    print()

    # Step 3: Find all files
    all_files = sorted(
        glob.glob(os.path.join(DATA_DIR, "**", "*.txt"), recursive=True) +
        glob.glob(os.path.join(DATA_DIR, "**", "*.pdf"), recursive=True)
    )

    if not all_files:
        print("⚠️  No .txt or .pdf files found in data/")
        print("   Drop files into data/ and re-run.")
        return

    new_files = [
        f for f in all_files
        if manifest.get(os.path.basename(f), {}).get("hash") != file_content_hash(f)
    ]

    print(f"📂 {len(all_files)} file(s) found — "
          f"{len(new_files)} new/changed, {len(all_files) - len(new_files)} unchanged\n")

    if not new_files:
        print("✅ Nothing to do — all files are up to date.")
        stats = index.describe_index_stats()
        print(f"📊 Pinecone index: {stats.get('total_vector_count', '?')} vectors")
        return

    # Step 4: Process new/changed files one at a time (bounded memory)
    total_chunks = 0
    errors = 0
    for filepath in new_files:
        n = process_file(filepath, vectorstore, manifest)
        if n > 0:
            total_chunks += n
        elif manifest.get(os.path.basename(filepath), {}).get("hash") != file_content_hash(filepath):
            errors += 1

    # Summary
    stats = index.describe_index_stats()
    total_vectors = stats.get("total_vector_count", "?")
    print(f"\n{'=' * 60}")
    print(f"✅ {total_chunks} new chunks added across {len(new_files) - errors} file(s)")
    if errors:
        print(f"⚠️  {errors} file(s) had errors — check output above")
    print(f"📊 Pinecone index total: {total_vectors} vectors")
    print(f"\n🎉 Run the app: streamlit run app_v2.py")


if __name__ == "__main__":
    main()
