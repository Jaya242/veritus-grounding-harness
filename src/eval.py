"""Evaluate the harness against the labeled claim set.

Reports:
  - Precision / Recall / F1 for FABRICATED-detection (the useful thing to catch)
  - Confusion matrix
  - Per-fabrication-type breakdown (numeric_distortion, scope_overreach, ...)
  - The full per-claim breakdown saved to results/eval_details.csv for error analysis
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


# WEAK verdicts are ambiguous. We evaluate two policies so we can report both:
#   - conservative: WEAK counts as "flagged as ungrounded" (favours recall on fabrications)
#   - permissive:   WEAK counts as "grounded" (favours precision)
def _binarize(verdict: str, policy: str) -> str:
    if verdict == "UNGROUNDED":
        return "FABRICATED"
    if verdict == "GROUNDED":
        return "GROUNDED"
    # WEAK
    return "FABRICATED" if policy == "conservative" else "GROUNDED"


def evaluate(results_csv: Path, claims_csv: Path, out_dir: Path) -> dict:
    results = pd.read_csv(results_csv)
    claims = pd.read_csv(claims_csv, dtype={"source_paper_id": str})
    df = results.merge(claims, on="claim_id", suffixes=("", "_true"))

    # Ground truth: FABRICATED vs GROUNDED
    df["truth"] = df["label"].str.upper()

    summary = {}
    for policy in ("conservative", "permissive"):
        df[f"pred_{policy}"] = df["verdict"].apply(lambda v: _binarize(v, policy))
        y_true = df["truth"]
        y_pred = df[f"pred_{policy}"]

        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=["FABRICATED"], average="binary", pos_label="FABRICATED"
        )
        cm = confusion_matrix(y_true, y_pred, labels=["FABRICATED", "GROUNDED"])
        summary[policy] = {
            "precision": round(float(p), 3),
            "recall": round(float(r), 3),
            "f1": round(float(f1), 3),
            "confusion_matrix": cm.tolist(),
        }

    # Per-fabrication-type recall — which failure modes does the harness catch?
    fab = df[df["truth"] == "FABRICATED"].copy()
    if not fab.empty and "fabrication_type" in fab.columns:
        fab["caught_conservative"] = fab["pred_conservative"] == "FABRICATED"
        by_type = (
            fab.groupby("fabrication_type")["caught_conservative"]
            .agg(["sum", "count"])
            .rename(columns={"sum": "caught", "count": "total"})
        )
        by_type["recall"] = (by_type["caught"] / by_type["total"]).round(3)
        summary["per_fabrication_type_recall"] = by_type.reset_index().to_dict(orient="records")

    # Print human-readable report
    print("\n=== HARNESS EVALUATION ===\n")
    for policy in ("conservative", "permissive"):
        s = summary[policy]
        print(f"[{policy}] precision={s['precision']}  recall={s['recall']}  f1={s['f1']}")
        print(f"  confusion matrix (rows=truth, cols=pred, order=[FABRICATED, GROUNDED]):")
        for row in s["confusion_matrix"]:
            print(f"    {row}")
        print()

    print("Full classification report (conservative policy):")
    print(classification_report(df["truth"], df["pred_conservative"], zero_division=0))

    if "per_fabrication_type_recall" in summary:
        print("Recall by fabrication type (conservative):")
        for row in summary["per_fabrication_type_recall"]:
            print(f"  {row['fabrication_type']:25s} {int(row['caught'])}/{int(row['total'])}  recall={row['recall']}")

    # Save per-claim details for error analysis
    out_dir.mkdir(parents=True, exist_ok=True)
    detail_cols = [
        "claim_id", "claim_text", "source_paper_id", "truth", "fabrication_type",
        "verdict", "pred_conservative", "pred_permissive",
        "similarity", "nli_label", "nli_confidence", "reasoning",
    ]
    detail_cols = [c for c in detail_cols if c in df.columns]
    df[detail_cols].to_csv(out_dir / "eval_details.csv", index=False)
    print(f"\nPer-claim breakdown saved to {out_dir / 'eval_details.csv'}")

    return summary


if __name__ == "__main__":
    root = Path(__file__).parent.parent
    evaluate(
        results_csv=root / "results" / "harness_results.csv",
        claims_csv=root / "data" / "claims.csv",
        out_dir=root / "results",
    )
