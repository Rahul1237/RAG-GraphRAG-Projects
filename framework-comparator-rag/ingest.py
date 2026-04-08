"""
################################################################################
### ingest.py — Framework Comparator RAG: Chunk + Embed + Store
### Loads raw_docs JSON → chunks → embeds → upserts into ChromaDB
################################################################################
### Author : Rahul Varma Cherukuri
### Project : Framework Comparator RAG (#8)
### Stack   : LangChain, ChromaDB, OpenAI text-embedding-3-small
################################################################################
"""

import os
import json
import argparse
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RAW_DIR    = "raw_docs"
CHROMA_DIR = "chroma_db"
COLLECTION = "framework_comparator"

CHUNK_SIZE    = 800     # tokens-ish — good balance for technical docs
CHUNK_OVERLAP = 100

EMBED_MODEL = "text-embedding-3-small"   # cheap + fast, plenty for this


# ---------------------------------------------------------------------------
# Load raw docs
# ---------------------------------------------------------------------------

def load_raw_docs(raw_dir: str, framework: str | None = None) -> list[dict]:
    """Load JSON files from raw_docs/. Optionally filter by framework."""
    docs = []
    raw_path = Path(raw_dir)

    if framework:
        files = [raw_path / f"{framework}.json"]
    else:
        files = list(raw_path.glob("*.json"))
        files = [f for f in files if f.stem != "all_frameworks"]

    for fp in files:
        if not fp.exists():
            print(f"  [WARN] File not found: {fp}")
            continue
        with open(fp, encoding="utf-8") as f:
            data = json.load(f)
        print(f"  Loaded {len(data):>4} docs from {fp.name}")
        docs.extend(data)

    return docs


# ---------------------------------------------------------------------------
# Chunk
# ---------------------------------------------------------------------------

def chunk_docs(raw_docs: list[dict]) -> list[Document]:
    """Split raw docs into LangChain Document chunks with metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    lc_docs = []
    for raw in raw_docs:
        chunks = splitter.split_text(raw["text"])
        for i, chunk in enumerate(chunks):
            lc_docs.append(Document(
                page_content=chunk,
                metadata={
                    "framework":  raw["framework"],
                    "title":      raw["title"],
                    "url":        raw["url"],
                    "doc_id":     raw["id"],
                    "chunk_idx":  i,
                    "chunk_total": len(chunks),
                }
            ))

    print(f"\n  Total chunks: {len(lc_docs)}")
    return lc_docs


# ---------------------------------------------------------------------------
# Embed + Store
# ---------------------------------------------------------------------------

def ingest_to_chroma(chunks: list[Document],
                     chroma_dir: str,
                     collection: str) -> Chroma:
    """Embed chunks and upsert into ChromaDB."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY not set. Export it before running ingest."
        )

    embeddings = OpenAIEmbeddings(
        model=EMBED_MODEL,
        openai_api_key=api_key,
    )

    print(f"\n  Embedding {len(chunks)} chunks with {EMBED_MODEL}...")
    print("  This may take a minute depending on volume.\n")

    # Chroma handles batching internally
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=chroma_dir,
        collection_name=collection,
    )

    print(f"  Stored in ChromaDB → {chroma_dir}/{collection}")
    return vectorstore


# ---------------------------------------------------------------------------
# Stats helper
# ---------------------------------------------------------------------------

def print_stats(chunks: list[Document]):
    from collections import Counter
    fw_counts = Counter(c.metadata["framework"] for c in chunks)
    print("\n  Chunk distribution by framework:")
    for fw, count in sorted(fw_counts.items()):
        bar = "█" * (count // 10)
        print(f"    {fw:<12} {count:>5} chunks  {bar}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Ingest framework docs into ChromaDB"
    )
    parser.add_argument(
        "--framework", "-f",
        default=None,
        choices=["langchain", "langgraph", "crewai", "autogen"],
        help="Ingest only one framework (default: all)",
    )
    parser.add_argument(
        "--raw-dir",  default=RAW_DIR,
        help=f"Directory with scraped JSON files (default: {RAW_DIR})",
    )
    parser.add_argument(
        "--chroma-dir", default=CHROMA_DIR,
        help=f"ChromaDB persist directory (default: {CHROMA_DIR})",
    )
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("  Framework Comparator RAG — Ingestion Pipeline")
    print(f"{'='*60}\n")

    # 1. Load
    print("  [1/3] Loading raw docs...")
    raw_docs = load_raw_docs(args.raw_dir, args.framework)
    if not raw_docs:
        print("  No docs found. Run scraper.py first.")
        return

    # 2. Chunk
    print(f"\n  [2/3] Chunking {len(raw_docs)} docs...")
    chunks = chunk_docs(raw_docs)
    print_stats(chunks)

    # 3. Embed + Store
    print("  [3/3] Embedding and storing...")
    ingest_to_chroma(chunks, args.chroma_dir, COLLECTION)

    print(f"\n{'='*60}")
    print("  Ingestion complete. Ready to query.")
    print(f"  ChromaDB path : {args.chroma_dir}")
    print(f"  Collection    : {COLLECTION}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
