"""
ingest.py — Document Ingestion Pipeline for EarningsLens

What this does:
1. Loads all .txt earnings call transcripts from the data/ folder
2. Splits each document into overlapping chunks (~1000 chars each)
   - Why overlap? So that if an important sentence falls on a chunk boundary,
     it still appears in at least one chunk fully intact.
3. Embeds each chunk using a local HuggingFace model (no API key needed)
4. Stores everything in a local ChromaDB vector database

Run this once: python ingest.py
Re-run whenever you add new transcripts to data/
"""

import os
import glob
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ── Config ──────────────────────────────────────────────────────────
DATA_DIR = "data"
CHROMA_DIR = "chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # ~90MB, runs locally, free
CHUNK_SIZE = 1000     # Characters per chunk
CHUNK_OVERLAP = 200   # Overlap between consecutive chunks


def load_documents(data_dir: str):
    """Load all .txt files from the data directory."""
    documents = []
    txt_files = glob.glob(os.path.join(data_dir, "**", "*.txt"), recursive=True)

    if not txt_files:
        print(f"⚠️  No .txt files found in {data_dir}/")
        print("   Add earnings call transcripts as .txt files and re-run.")
        return documents

    for filepath in txt_files:
        print(f"📄 Loading: {filepath}")
        loader = TextLoader(filepath, encoding="utf-8")
        docs = loader.load()

        # Add the filename as metadata so we can cite sources later
        for doc in docs:
            doc.metadata["source"] = os.path.basename(filepath)

        documents.extend(docs)

    print(f"\n✅ Loaded {len(documents)} document(s) from {len(txt_files)} file(s)")
    return documents


def chunk_documents(documents):
    """
    Split documents into smaller chunks for embedding.

    Why RecursiveCharacterTextSplitter?
    - It tries to split on natural boundaries (paragraphs → sentences → words)
    - Falls back to character-level only if needed
    - This preserves meaning better than naive fixed-size splits
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],  # Priority order for splits
    )

    chunks = splitter.split_documents(documents)
    print(f"✅ Split into {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


def create_vector_store(chunks):
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

    # Step 1: Load
    documents = load_documents(DATA_DIR)
    if not documents:
        return

    # Step 2: Chunk
    chunks = chunk_documents(documents)

    # Step 3: Embed & Store
    create_vector_store(chunks)

    print("\n🎉 Ingestion complete! You can now run the app with: streamlit run app.py")


if __name__ == "__main__":
    main()
