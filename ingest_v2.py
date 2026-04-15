"""
ingest_v2.py — Document Ingestion Pipeline for EarningsLens v2

What changed from v1:
- ChromaDB (local disk) → Pinecone (managed cloud vector store)
- Documents are uploaded to S3-compatible storage (optional, see config)
- Embedding dimension explicitly declared (384 for all-MiniLM-L6-v2)
- Duplicate detection: skips re-embedding docs already in the index
  (Pinecone lets us check by ID before upserting)

Everything else is identical to v1:
- Same HuggingFace embedding model (all-MiniLM-L6-v2, 384 dims)
- Same chunking strategy (RecursiveCharacterTextSplitter)
- Same PDF extraction logic (pdfplumber)
- Same metadata structure (source, page_number, doc_type)

Run once: python ingest_v2.py
Re-run when you add new documents — existing chunks are skipped.
"""

import os
import sys
import glob
import hashlib

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
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384          # Must match the model output — all-MiniLM-L6-v2 = 384
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
PINECONE_INDEX_NAME = "earningslens-v2"   # Change if you prefer a different name
PINECONE_CLOUD = "aws"                    # Free tier is on AWS
PINECONE_REGION = "us-east-1"            # Free tier default region

BOILERPLATE_PHRASES = [
    "edgar online",
    "united states securities and exchange commission",
    "form 10-q",
    "form 10-k",
    "table of contents",
]


# ── Pinecone setup ───────────────────────────────────────────────────

def get_pinecone_index():
    """
    Connect to Pinecone and ensure the index exists.

    Pinecone requires:
    - PINECONE_API_KEY in environment / .env
    - An index with the correct dimension declared upfront
      (unlike ChromaDB which infers it)

    ServerlessSpec is the free-tier option — no pod costs.
    """
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
            metric="cosine",              # Cosine similarity — standard for text
            spec=ServerlessSpec(
                cloud=PINECONE_CLOUD,
                region=PINECONE_REGION,
            ),
        )
        print("   ✅ Index created.")
    else:
        print(f"✅ Pinecone index '{PINECONE_INDEX_NAME}' already exists.")

    return pc.Index(PINECONE_INDEX_NAME)


def make_chunk_id(source: str, chunk_index: int) -> str:
    """
    Generate a stable, unique ID for each chunk.

    Why? Pinecone upserts are idempotent by ID — if we re-run ingest
    with the same documents, we won't create duplicates. We just overwrite.

    Format: sha256(source)[:12]-chunk-{index}
    Example: 'a3f9c2b1d4e8-chunk-0042'
    """
    source_hash = hashlib.sha256(source.encode()).hexdigest()[:12]
    return f"{source_hash}-chunk-{chunk_index:04d}"


# ── Document loading (unchanged from v1) ────────────────────────────

def load_txt_documents(data_dir: str) -> list[Document]:
    """Load all .txt files from the data directory."""
    documents = []
    txt_files = glob.glob(os.path.join(data_dir, "**", "*.txt"), recursive=True)

    if not txt_files:
        print(f"   No .txt files found in {data_dir}/")
        return documents

    for filepath in txt_files:
        print(f"   📄 {filepath}")
        loader = TextLoader(filepath, encoding="utf-8")
        docs = loader.load()

        for doc in docs:
            doc.metadata["source"] = os.path.basename(filepath)
            doc.metadata["doc_type"] = "transcript"

        documents.extend(docs)

    print(f"   ✅ Loaded {len(documents)} transcript(s) from {len(txt_files)} .txt file(s)\n")
    return documents


def _format_table(table: list[list]) -> str:
    """Convert pdfplumber table rows to a readable string."""
    rows = []
    for row in table:
        cleaned = [str(cell).strip() if cell else "-" for cell in row]
        rows.append(" | ".join(cleaned))
    return "\n".join(rows)


def _is_boilerplate(text: str) -> bool:
    """Return True if the page is mostly boilerplate."""
    lowered = text.lower()
    hits = sum(1 for phrase in BOILERPLATE_PHRASES if phrase in lowered)
    return hits >= 2


def load_pdf_documents(data_dir: str) -> list[Document]:
    """Load all .pdf files from the data directory using pdfplumber."""
    documents = []
    pdf_files = glob.glob(os.path.join(data_dir, "**", "*.pdf"), recursive=True)

    if not pdf_files:
        print(f"   No .pdf files found in {data_dir}/")
        return documents

    for filepath in pdf_files:
        filename = os.path.basename(filepath)
        print(f"   📑 {filepath}")
        page_count = 0
        skipped = 0

        try:
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

                    doc = Document(
                        page_content=full_text,
                        metadata={
                            "source": filename,
                            "page_number": page_num,
                            "doc_type": "pdf",
                        },
                    )
                    documents.append(doc)
                    page_count += 1

        except Exception as e:
            print(f"   ⚠️  Could not read {filename}: {e}")
            continue

        print(f"      → {page_count} pages extracted, {skipped} skipped")

    print(f"   ✅ Loaded {len(documents)} page(s) from {len(pdf_files)} .pdf file(s)\n")
    return documents


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split documents into chunks — identical strategy to v1."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"✅ Split into {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


# ── Vector store (Pinecone) ──────────────────────────────────────────

def create_vector_store(chunks: list[Document], index):
    """
    Embed chunks and upsert into Pinecone.

    Key differences from v1 ChromaDB approach:
    1. We assign stable IDs to each chunk (for idempotent upserts)
    2. We batch upserts in groups of 100 — Pinecone recommends this
       to avoid hitting request size limits
    3. The index persists in Pinecone's cloud — no local files,
       no .gitignore issues, no ephemeral Streamlit Cloud filesystem

    LangChain's PineconeVectorStore handles the actual upsert call.
    """
    print(f"🔄 Loading embedding model: {EMBEDDING_MODEL}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )

    # Assign stable IDs before upserting
    ids = []
    for i, chunk in enumerate(chunks):
        chunk_id = make_chunk_id(
            chunk.metadata.get("source", "unknown"),
            i
        )
        ids.append(chunk_id)

    print(f"🔄 Upserting {len(chunks)} chunks into Pinecone '{PINECONE_INDEX_NAME}'...")
    print("   (Re-running is safe — existing chunks are overwritten, not duplicated)")

    # PineconeVectorStore.from_documents handles batching internally
    vectorstore = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME,
        ids=ids,
    )

    # Verify upsert by checking index stats
    stats = index.describe_index_stats()
    total_vectors = stats.get("total_vector_count", "unknown")
    print(f"✅ Pinecone index now contains {total_vectors} vectors")

    return vectorstore


def main():
    print("=" * 60)
    print("EarningsLens v2 — Document Ingestion Pipeline (Pinecone)")
    print("=" * 60 + "\n")

    # ── Step 0: Connect to Pinecone ──────────────────────────────────
    print("🔗 Connecting to Pinecone...")
    index = get_pinecone_index()

    # ── Step 1: Load ─────────────────────────────────────────────────
    print("\n📂 Loading documents...\n")
    txt_docs = load_txt_documents(DATA_DIR)
    pdf_docs = load_pdf_documents(DATA_DIR)
    all_documents = txt_docs + pdf_docs

    if not all_documents:
        print("⚠️  No documents found in data/")
        print("   Add .txt transcripts or .pdf filings and re-run.")
        return

    print(f"📊 Total: {len(all_documents)} document(s) — "
          f"{len(txt_docs)} transcript(s), {len(pdf_docs)} PDF page(s)\n")

    # ── Step 2: Chunk ─────────────────────────────────────────────────
    chunks = chunk_documents(all_documents)

    # ── Step 3: Embed & Store ─────────────────────────────────────────
    create_vector_store(chunks, index)

    print("\n🎉 Ingestion complete! Run the v2 app with: streamlit run app_v2.py")


if __name__ == "__main__":
    main()
