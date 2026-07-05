"""Fetch abstracts from arXiv and save them to data/papers.json."""
from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import quote

import feedparser


ARXIV_API = "http://export.arxiv.org/api/query"


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def fetch_arxiv(query: str, max_results: int = 8) -> list[dict]:
    url = f"{ARXIV_API}?search_query={quote(query)}&max_results={max_results}"
    feed = feedparser.parse(url)
    papers = []
    for entry in feed.entries:
        arxiv_id = entry.id.rsplit("/", 1)[-1]
        papers.append({
            "paper_id": arxiv_id,
            "title": _clean(entry.title),
            "abstract": _clean(entry.summary),
            "url": entry.id,
            "authors": [a.name for a in entry.authors],
            "published": entry.published,
        })
    return papers


def main():
    # Pick a topic you can actually judge — if you don't know the field,
    # you can't tell when a claim overreaches.
    queries = [
        "cat:cs.CV AND (retrieval OR grounding)",  # vision + retrieval
        "cat:cs.CL AND hallucination",             # NLP + hallucination
    ]
    all_papers = []
    for q in queries:
        print(f"Fetching: {q}")
        papers = fetch_arxiv(q, max_results=4)
        all_papers.extend(papers)
        print(f"  got {len(papers)}")

    out = Path(__file__).parent.parent / "data" / "papers.json"
    out.write_text(json.dumps(all_papers, indent=2, ensure_ascii=False))
    print(f"\nSaved {len(all_papers)} papers to {out}")
    print("\nNext: read the abstracts, pick 6–8 you understand, and start writing claims in data/claims.csv")


if __name__ == "__main__":
    main()
