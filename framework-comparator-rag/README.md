# Framework Comparator RAG 🤖📊

An autonomous Retrieval-Augmented Generation (RAG) pipeline designed to compare Agentic AI frameworks (**LangGraph, CrewAI, AutoGen, and LangChain**) based strictly on their official documentation. 

This tool solves the "framework fatigue" problem by scraping the latest technical docs, vectorizing them, and providing a query interface for architectural comparisons.

## 🏗 Architecture & Tech Stack

* **Scraper:** Custom BFS crawler using `requests` and `BeautifulSoup4` to target deep technical documentation while ignoring "marketing fluff."
* **Orchestration:** `LangChain` & `langchain-text-splitters` for modular RAG logic.
* **Vector Database:** `ChromaDB` (local storage).
* **Embeddings:** OpenAI `text-embedding-3-small`.
* **LLM (Generation):** OpenAI `gpt-4o` (customized via system prompts for architect-level analysis).

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.10+ installed.

```bash
# Clone the master repository if you haven't already
git clone [https://github.com/Rahul1237/RAG-GraphRAG-Projects.git](https://github.com/Rahul1237/RAG-GraphRAG-Projects.git)
cd RAG-GraphRAG-Projects/framework-comparator-rag

# Install dependencies
pip install -r requirements.txt
