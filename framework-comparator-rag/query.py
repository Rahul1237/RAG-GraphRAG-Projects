"""
################################################################################
### query.py — Framework Comparator RAG: Query Interface
### CLI + reusable query() function for the framework comparator
################################################################################
### Author : Rahul Varma Cherukuri
### Project : Framework Comparator RAG (#8)
### Stack   : LangChain, ChromaDB, Anthropic Claude
################################################################################
"""

import os
import argparse
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CHROMA_DIR  = "chroma_db"
COLLECTION  = "framework_comparator"
EMBED_MODEL = "text-embedding-3-small"
TOP_K       = 20     # retrieve top-k chunks across all frameworks

SYSTEM_PROMPT = """You are an expert AI framework architect with deep knowledge of
LangChain, LangGraph, CrewAI, and AutoGen. Your job is to compare and contrast
these frameworks based strictly on their official documentation.

Rules:
- Always cite which framework(s) each point applies to
- When frameworks differ, explicitly call out the difference
- Be concrete — reference actual class names, methods, or config patterns
- If the retrieved context doesn't cover the question, say so
- Format comparisons as clear tables or labeled sections when helpful"""


# ---------------------------------------------------------------------------
# Load vectorstore
# ---------------------------------------------------------------------------

def load_vectorstore(chroma_dir: str, collection: str) -> Chroma:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set.")

    embeddings = OpenAIEmbeddings(
        model=EMBED_MODEL,
        openai_api_key=api_key,
    )
    return Chroma(
        persist_directory=chroma_dir,
        collection_name=collection,
        embedding_function=embeddings,
    )


# ---------------------------------------------------------------------------
# Retrieve
# ---------------------------------------------------------------------------

def retrieve(vectorstore: Chroma,
             query: str,
             top_k: int = TOP_K,
             framework_filter: str | None = None) -> list[dict]:
    """Retrieve top-k chunks. Optionally filter by framework."""

    where = {"framework": framework_filter} if framework_filter else None

    results = vectorstore.similarity_search_with_score(
        query,
        k=top_k,
        filter=where,
    )

    chunks = []
    for doc, score in results:
        chunks.append({
            "framework": doc.metadata["framework"],
            "title":     doc.metadata["title"],
            "url":       doc.metadata["url"],
            "text":      doc.page_content,
            "score":     round(score, 4),
        })

    return chunks


# ---------------------------------------------------------------------------
# Format context for Claude
# ---------------------------------------------------------------------------

def build_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(
            f"[{i}] [{c['framework'].upper()}] {c['title']}\n"
            f"URL: {c['url']}\n"
            f"Relevance score: {c['score']}\n"
            f"---\n{c['text']}\n"
        )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Generate answer via Claude
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Generate answer via OpenAI
# ---------------------------------------------------------------------------

def generate_answer(query: str, context: str) -> str:
    # Initialize the OpenAI Chat model (gpt-4o or gpt-3.5-turbo)
    llm = ChatOpenAI(
        model="gpt-4o", 
        temperature=0,  # Keep it at 0 for factual RAG responses
        max_tokens=2048
    )

    user_msg = (
        f"RETRIEVED DOCUMENTATION CONTEXT:\n\n{context}\n\n"
        f"{'='*60}\n\n"
        f"QUESTION: {query}"
    )

    # LangChain allows passing a list of tuples for (role, content)
    messages = [
        ("system", SYSTEM_PROMPT),
        ("user", user_msg)
    ]

    # Invoke the model and return the string content
    response = llm.invoke(messages)
    return response.content


# ---------------------------------------------------------------------------
# Full RAG pipeline
# ---------------------------------------------------------------------------

def query(question: str,
          framework_filter: str | None = None,
          verbose: bool = False) -> str:
    """End-to-end RAG query. Returns Claude's answer."""
    vs      = load_vectorstore(CHROMA_DIR, COLLECTION)
    chunks  = retrieve(vs, question, framework_filter=framework_filter)
    context = build_context(chunks)

    if verbose:
        print(f"\n  Retrieved {len(chunks)} chunks:")
        for c in chunks:
            print(f"    [{c['framework']}] {c['title'][:50]}  score={c['score']}")
        print()

    return generate_answer(question, context)


# ---------------------------------------------------------------------------
# Example queries to try
# ---------------------------------------------------------------------------

EXAMPLE_QUERIES = [
    "How do LangGraph and AutoGen differ in handling multi-agent communication?",
    "Compare how each framework implements human-in-the-loop workflows.",
    "Which framework is best for building a long-running stateful agent and why?",
    "How does CrewAI's role-based agent design compare to LangGraph's node-based approach?",
    "What are the memory management options in each framework?",
    "Compare error handling and retry logic across all four frameworks.",
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Query the Framework Comparator RAG"
    )
    parser.add_argument("question", nargs="?", help="Question to ask")
    parser.add_argument(
        "--framework", "-f",
        default=None,
        choices=["langchain", "langgraph", "crewai", "autogen"],
        help="Filter retrieval to one framework",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--examples", action="store_true",
                        help="Print example queries and exit")
    args = parser.parse_args()

    if args.examples:
        print("\nExample queries to try:\n")
        for i, q in enumerate(EXAMPLE_QUERIES, 1):
            print(f"  {i}. {q}")
        print()
        return

    question = args.question
    if not question:
        question = input("Enter your question: ").strip()

    print(f"\n  Query: {question}\n")
    answer = query(question,
                   framework_filter=args.framework,
                   verbose=args.verbose)

    print("=" * 60)
    print(answer)
    print("=" * 60)


if __name__ == "__main__":
    main()
