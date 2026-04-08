"""
################################################################################
### scraper.py — Framework Comparator RAG: Doc Scraper
### Scrapes LangChain, LangGraph, CrewAI, AutoGen docs into raw JSON chunks
################################################################################
### Author : Rahul Varma Cherukuri
### Project : Framework Comparator RAG (#8)
### Stack   : requests, BeautifulSoup, JSON
################################################################################
"""

import os
import json
import time
import hashlib
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from datetime import datetime

# ---------------------------------------------------------------------------
# Config: one entry per framework
# ---------------------------------------------------------------------------
FRAMEWORKS = {
    "langchain": {
        "base_url": "https://python.langchain.com",
        "start_paths": [
            "/docs/concepts/#agents",
            "/docs/how_to/#agents",
        ],
        "url_must_contain": "/docs/",
    },
    "langgraph": {
        "base_url": "https://langchain-ai.github.io",
        "start_paths": [
            "/langgraph/concepts/multi_agent/",
            "/langgraph/how-tos/#multi-agent",
        ],
        "url_must_contain": "/langgraph/",
    },
    "crewai": {
        "base_url": "https://docs.crewai.com",
        "start_paths": [
            "/concepts/agents",
            "/concepts/tasks",
            "/how-to/hierarchical",
            "/how-to/sequential",
        ],
        "url_must_contain": "/",
    },
    "autogen": {
        "base_url": "https://microsoft.github.io",
        "start_paths": [
            "/autogen/stable/user-guide/core-user-guide/framework/conversable-agent",
            "/autogen/stable/user-guide/core-user-guide/design-patterns/intro",
        ],
        "url_must_contain": "/autogen/",
    },
}

OUTPUT_DIR  = "raw_docs"
MAX_PAGES   = 150         # per framework — bump up if you want more coverage
DELAY       = 0.5         # seconds between requests — be polite

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; RAGBot/1.0; +research-project)"
    )
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_id(url: str) -> str:
    """Stable doc ID from URL hash."""
    return hashlib.md5(url.encode()).hexdigest()[:12]


def clean_text(soup: BeautifulSoup) -> str:
    """Strip nav/footer/scripts and return main content text."""
    for tag in soup(["nav", "footer", "script", "style", "header",
                     "aside", ".sidebar", ".toc"]):
        tag.decompose()

    # Try to grab main content area
    main = (
        soup.find("main")
        or soup.find("article")
        or soup.find(class_=["content", "md-content", "doc-content"])
        or soup.find("body")
    )
    if not main:
        return ""

    text = main.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines)


def extract_links(soup: BeautifulSoup, base_url: str,
                  must_contain: str) -> list[str]:
    """Return internal links that match must_contain filter."""
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        full = urljoin(base_url, href)
        parsed = urlparse(full)
        # Same domain + must_contain filter + no fragments-only
        if (parsed.netloc == urlparse(base_url).netloc
                and must_contain in parsed.path
                and not href.startswith("#")):
            clean = full.split("#")[0]  # drop anchors
            links.append(clean)
    return list(set(links))


def scrape_framework(name: str, cfg: dict) -> list[dict]:
    """BFS crawl of a framework's docs. Returns list of doc dicts."""
    base_url    = cfg["base_url"]
    must_contain = cfg["url_must_contain"]

    visited: set[str] = set()
    queue: list[str]  = [
        urljoin(base_url, p) for p in cfg["start_paths"]
    ]
    docs: list[dict]  = []

    print(f"\n{'='*60}")
    print(f"  Scraping: {name.upper()}  |  base={base_url}")
    print(f"{'='*60}")

    while queue and len(docs) < MAX_PAGES:
        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)

        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            if resp.status_code != 200:
                print(f"  [SKIP {resp.status_code}] {url}")
                continue

            soup = BeautifulSoup(resp.text, "html.parser")
            title = soup.title.string.strip() if soup.title else url
            text  = clean_text(soup)

            if len(text) < 100:        # skip near-empty pages
                continue

            doc = {
                "id":          make_id(url),
                "framework":   name,
                "url":         url,
                "title":       title,
                "text":        text,
                "scraped_at":  datetime.utcnow().isoformat(),
                "char_count":  len(text),
            }
            docs.append(doc)
            print(f"  [OK {len(docs):>3}] {title[:60]}")

            # Discover more links
            new_links = extract_links(soup, base_url, must_contain)
            for lnk in new_links:
                if lnk not in visited:
                    queue.append(lnk)

            time.sleep(DELAY)

        except Exception as exc:
            print(f"  [ERR] {url} — {exc}")

    print(f"  Done: {len(docs)} pages scraped for {name}")
    return docs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_docs = []

    for name, cfg in FRAMEWORKS.items():
        docs = scrape_framework(name, cfg)
        all_docs.extend(docs)

        # Save per-framework file immediately
        fw_path = os.path.join(OUTPUT_DIR, f"{name}.json")
        with open(fw_path, "w", encoding="utf-8") as f:
            json.dump(docs, f, indent=2, ensure_ascii=False)
        print(f"  Saved → {fw_path}")

    # Save combined
    combined_path = os.path.join(OUTPUT_DIR, "all_frameworks.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_docs, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  TOTAL DOCS SCRAPED : {len(all_docs)}")
    print(f"  Combined file      : {combined_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
