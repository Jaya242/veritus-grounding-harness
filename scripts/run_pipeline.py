"""End-to-end runner: load claims + papers, score every claim, save results, evaluate.

Usage:
  python scripts/run_pipeline.py                    # uses data/papers.json + data/claims.csv
  python scripts/run_pipeline.py --papers seed      # uses data/seed_papers.json (for smoke tests)
  python scripts/run_pipeline.py --mock             # skip real models — use a fake harness for plumbing test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Make src importable when running from repo root
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.harness import CombinedHarness, GroundingResult, Verdict  # noqa: E402


def load_papers(path: Path) -> dict[str, str]:
    """Return {paper_id: abstract}."""
    papers = json.loads(path.read_text())
    return {p["paper_id"]: p["abstract"] for p in papers}


class MockHarness:
    """No-model stand-in for plumbing tests. Deterministic, obviously fake."""
    def evaluate(self, claim_id: str, claim: str, passage: str) -> GroundingResult:
        # crude: if any 5-char substring of claim appears in passage, call it GROUNDED
        overlap = any(claim[i:i + 8].lower() in passage.lower() for i in range(0, max(1, len(claim) - 8), 4))
        verdict = Verdict.GROUNDED if overlap else Verdict.UNGROUNDED
        return GroundingResult(
            claim_id=claim_id,
            similarity=0.7 if overlap else 0.2,
            nli_label="entailment" if overlap else "neutral",
            nli_confidence=0.8,
            verdict=verdict,
            reasoning=f"[mock] substring_overlap={overlap}",
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--papers", choices=("full", "seed"), default="seed",
                        help="'seed' uses hand-authored seed_papers.json; 'full' uses arXiv-fetched papers.json")
    parser.add_argument("--mock", action="store_true", help="Skip real models (fast plumbing check)")
    parser.add_argument("--claims", type=str, default="data/claims.csv")
    parser.add_argument("--no-sentence-retrieval", action="store_true",
                        help="Feed the whole passage to NLI (baseline v1, before the sentence-level fix)")
    args = parser.parse_args()

    papers_file = ROOT / ("data/seed_papers.json" if args.papers == "seed" else "data/papers.json")
    claims_file = ROOT / args.claims

    if not papers_file.exists():
        print(f"ERROR: {papers_file} not found. Run src/collect_papers.py first, or use --papers seed.")
        sys.exit(1)

    papers = load_papers(papers_file)
    # Force paper_id as string — arXiv IDs like "2010.11929" get auto-parsed as float otherwise.
    claims = pd.read_csv(claims_file, dtype={"source_paper_id": str})
    # Strip whitespace on all string columns — Numbers/Excel love to add stray spaces.
    for col in claims.select_dtypes(include="object").columns:
        claims[col] = claims[col].astype(str).str.strip()

    missing = set(claims["source_paper_id"]) - set(papers)
    if missing:
        print(f"WARNING: {len(missing)} claim(s) reference papers not in {papers_file.name}: {missing}")

    print(f"Loaded {len(papers)} papers, {len(claims)} claims")
    print(f"Harness: {'MOCK (no models)' if args.mock else 'real (sentence-transformers + NLI)'}")

    harness = MockHarness() if args.mock else CombinedHarness(
        use_sentence_retrieval=not args.no_sentence_retrieval,
    )

    rows = []
    for _, row in claims.iterrows():
        passage = papers.get(row["source_paper_id"])
        if not passage:
            continue
        result = harness.evaluate(row["claim_id"], row["claim_text"], passage)
        rows.append(result.to_dict())
        print(f"  {row['claim_id']}: {result.verdict.value:12s} sim={result.similarity:.2f} "
              f"nli={result.nli_label}({result.nli_confidence:.2f}) — {result.reasoning}")

    out = ROOT / "results" / "harness_results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved {len(rows)} results to {out}")

    # Chain into eval automatically
    from src.eval import evaluate
    evaluate(out, claims_file, ROOT / "results")


if __name__ == "__main__":
    main()
