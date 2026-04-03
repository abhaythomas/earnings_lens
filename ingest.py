"""
ingest.py — Document Ingestion Pipeline for EarningsLens

What this does:
1. Loads all .txt earnings call transcripts from the data/ folder
2. Loads all .pdf files (SEC filings, earnings press releases) from the data/ folder
   - Extracts plain text page by page
   - Extracts tables separately and formats them as readable strings
3. Splits each document into overlapping chunks (~1000 chars each)
   - Why overlap? So that if an important sentence falls on a chunk boundary,
     it still appears in at least one chunk fully intact.
4. Embeds each chunk using a local HuggingFace model (no API key needed)
5. Stores everything in a local ChromaDB vector database

Run this once: python ingest.py
Re-run whenever you add new transcripts to data/
"""

import os
import glob
import pdfplumber
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# ── Config ──────────────────────────────────────────────────────────
DATA_DIR = "data"
CHROMA_DIR = "chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # ~90MB, runs locally, free
CHUNK_SIZE = 1000     # Characters per chunk
CHUNK_OVERLAP = 200   # Overlap between consecutive chunks

# Headers/footers common in SEC filings — stripped before chunking
BOILERPLATE_PHRASES = [
    "edgar online",
    "united states securities and exchange commission",
    "form 10-q",
    "form 10-k",
    "table of contents",
]


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
    """
    Convert a pdfplumber table (list of rows, each row a list of cells)
    into a readable string.

    Example output:
        Revenue | $12.3B | +8% YoY
        Net Income | $3.1B | +12% YoY

    Empty cells are replaced with '-' so the structure stays clear.
    """
    rows = []
    for row in table:
        # Replace None cells with '-'
        cleaned = [str(cell).strip() if cell else "-" for cell in row]
        rows.append(" | ".join(cleaned))
    return "\n".join(rows)


def _is_boilerplate(text: str) -> bool:
    """Return True if the page is mostly boilerplate (cover pages, ToC, etc.)."""
    lowered = text.lower()
    # If more than 2 boilerplate phrases appear, skip the page
    hits = sum(1 for phrase in BOILERPLATE_PHRASES if phrase in lowered)
    return hits >= 2


def load_pdf_documents(data_dir: str) -> list[Document]:
    """
    Load all .pdf files from the data directory using pdfplumber.

    For each page:
    - Extract plain text
    - Extract any tables and format them as readable strings
    - Skip boilerplate-heavy pages (cover, ToC)
    - Attach metadata: source filename, page number, doc_type

    Returns a list of LangChain Document objects — one per page.
    Why one per page? It makes citations precise ("Apple 10-Q, page 4")
    and keeps chunks naturally sized since PDF pages are already ~500-1500 chars.
    """
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

                    # ── Extract plain text ────────────────────────
                    raw_text = page.extract_text() or ""

                    # ── Extract tables ────────────────────────────
                    # pdfplumber detects table boundaries and returns
                    # structured rows — much better than raw text for financials
                    tables = page.extract_tables()
                    table_text = ""
                    if tables:
                        formatted = [_format_table(t) for t in tables if t]
                        table_text = "\n\n[TABLE]\n" + "\n\n[TABLE]\n".join(formatted)

                    full_text = raw_text + table_text

                    # ── Skip near-empty or boilerplate pages ──────
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
    """
    Split documents into smaller chunks for embedding.

    Why RecursiveCharacterTextSplitter?
    - It tries to split on natural boundaries (paragraphs → sentences → words)
    - Falls back to character-level only if needed
    - This preserves meaning better than naive fixed-size splits

    Note: metadata (source, page_number, doc_type) is preserved on every chunk
    automatically by LangChain's splitter.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],  # Priority order for splits
    )

    chunks = splitter.split_documents(documents)
    print(f"✅ Split into {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


def create_vector_store(chunks: list[Document]):
    """
    Embed chunks and store in ChromaDB.

    HuggingFaceEmbeddings runs locally — no API key needed.
    all-MiniLM-L6-v2 is a ~90MB model that produces 384-dimensional vectors.
    It downloads automatically on first run.
    """
    print(f"🔄 Loading embedding model: {EMBEDDING_MODEL} (first run downloads ~90MB)...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},  # Use "cuda" if you have a GPU
    )

    print("🔄 Embedding chunks and storing in ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )

    print(f"✅ Vector store created at {CHROMA_DIR}/ with {len(chunks)} vectors")
    return vectorstore


def main():
    print("=" * 60)
    print("EarningsLens — Document Ingestion Pipeline")
    print("=" * 60 + "\n")

    # ── Step 1: Load ─────────────────────────────────────────────────
    print("📂 Loading documents...\n")

    txt_docs = load_txt_documents(DATA_DIR)
    pdf_docs = load_pdf_documents(DATA_DIR)

    all_documents = txt_docs + pdf_docs

    if not all_documents:
        print("⚠️  No documents found in data/")
        print("   Add .txt transcripts or .pdf filings and re-run.")
        return

    txt_count = len(txt_docs)
    pdf_count = len(pdf_docs)
    print(f"📊 Total: {len(all_documents)} document(s) — {txt_count} transcript(s), {pdf_count} PDF page(s)\n")

    # ── Step 2: Chunk ─────────────────────────────────────────────────
    chunks = chunk_documents(all_documents)

    # ── Step 3: Embed & Store ─────────────────────────────────────────
    create_vector_store(chunks)

    print("\n🎉 Ingestion complete! You can now run the app with: streamlit run app.py")


if __name__ == "__main__":
    main()
