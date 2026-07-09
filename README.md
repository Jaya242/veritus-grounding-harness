# Grounding Harness for Research-Paper Claims

A small tool that checks whether a factual claim about a research paper is actually supported by that paper's abstract. Built as a proof-of-concept for the "hallucination-free, source-verified" property that Veritus.ai advertises.

## The problem

LLM assistants that answer questions over research papers routinely produce statements that *sound* like the source but aren't actually in it — wrong numbers, invented comparisons, subtle scope overreach, wrong attribution. These are hard to catch by eye and easy to miss even with a citation link, because the citation looks correct.

This project builds a small measurement tool: given a (claim, source passage) pair, output one of `GROUNDED / UNGROUNDED / WEAK`, plus the underlying signals. Then it evaluates the tool itself against a hand-labeled test set, so the reported precision/recall are honest instead of asserted.

## Method

Two independent signals per claim–passage pair:

1. **Semantic similarity** — sentence-transformers (`all-MiniLM-L6-v2`) cosine similarity between claim and passage embeddings. Answers: are they topically related?
2. **Textual entailment** — a cross-encoder NLI model (`cross-encoder/nli-deberta-v3-base`) that classifies the pair as `entailment / neutral / contradiction`. Answers: does the passage support the claim?

Similarity alone is insufficient: a claim can be topically similar to a passage and still contradict it (this is the whole reason for adding NLI). NLI alone can be noisy at low confidence, so similarity is used as a tie-breaker.

**Decision logic** (in `src/harness.py`):

| NLI label      | Confidence         | Similarity         | Verdict     |
|----------------|--------------------|--------------------|-------------|
| contradiction  | ≥ 0.55             | any                | UNGROUNDED  |
| entailment     | ≥ 0.55             | ≥ 0.55             | GROUNDED    |
| neutral        | any                | < 0.30             | UNGROUNDED  |
| anything else  | —                  | —                  | WEAK        |

`WEAK` means "the two signals disagree or neither is confident" → flag for human review, don't auto-decide.

## Dataset

A hand-labeled set of claims across a handful of real arXiv abstracts. Every fabricated claim carries a `fabrication_type` from a fixed taxonomy so we can measure which failure modes the harness catches vs. misses:

- `numeric_distortion` — right claim, wrong number
- `scope_overreach` — a true statement stretched further than the paper supports
- `causal_overreach` — correlation restated as causation
- `invented_comparison` — comparison to a baseline the paper never makes
- `wrong_attribution` — real finding, wrong paper/method/dataset

Current dataset: **25 claims across 5 abstracts** (`data/claims.csv`) — 15 fabricated, 10 grounded. Papers span computer vision (ViT, Fisher Vectors) and hallucination detection (RAG, PSRD, ECLIPSE) — thematically close to what a research-paper RAG system like Veritus needs to grade against.

## How to run

```bash
# 1. Install deps (once)
pip install -r requirements.txt

# 2. Smoke test (no models — verifies file plumbing works)
python scripts/run_pipeline.py --mock --papers seed

# 3. Real run against seed papers (downloads ~500MB of models the first time)
python scripts/run_pipeline.py --papers seed

# 4. Fetch fresh arXiv abstracts and score against those instead
python src/collect_papers.py                # writes data/papers.json
python scripts/run_pipeline.py --papers full
```

Outputs land in `results/`:
- `harness_results.csv` — per-claim scores + verdict
- `eval_details.csv` — merged with ground truth for error analysis

## Results (N = 25)

Two decision policies are reported:
- **Conservative** — a `WEAK` verdict counts as flagged-as-ungrounded (favours recall on fabrications; matches a production "flag for human review" pattern).
- **Permissive** — a `WEAK` verdict counts as grounded (favours precision; only strict `UNGROUNDED` verdicts flag).

| Variant | Policy | Precision | Recall | F1 |
|---------|--------|-----------|--------|-----|
| v1 — paragraph-level NLI | conservative | 0.652 | 1.000 | 0.789 |
| v1 — paragraph-level NLI | permissive   | 1.000 | 0.267 | 0.421 |
| **v2 — sentence-level retrieval + NLI** | **conservative** | **0.682** | **1.000** | **0.811** |
| v2 — sentence-level retrieval + NLI | permissive   | 1.000 | 0.333 | 0.500 |

Sentence-level retrieval (v2) picks the single sentence in the abstract most semantically similar to the claim before running NLI, instead of feeding the full abstract. This addresses a documented weakness of MNLI-trained cross-encoders on long premises. On this dataset, it improves conservative F1 by +2.2 points and permissive F1 by +7.9 points without ever missing a fabrication.

### Recall by fabrication type (v2, conservative)

| Fabrication type      | Caught | Total | Recall |
|-----------------------|--------|-------|--------|
| numeric_distortion    | 5      | 5     | 1.000  |
| scope_overreach       | 3      | 3     | 1.000  |
| wrong_attribution     | 2      | 2     | 1.000  |
| invented_comparison   | 3      | 3     | 1.000  |
| causal_overreach      | 2      | 2     | 1.000  |
| **Total**             | **15** | **15**| **1.000** |

Every fabricated claim is caught under conservative policy. Precision is limited by the harness's caution on grounded paraphrases: 7 of 10 grounded claims land in `WEAK` because the NLI model returns `neutral` with high confidence. In a production Veritus deployment, these `WEAK` verdicts route to human review — the correct behaviour when the harness is unsure.

## Where this breaks

- **Small dataset.** 25 claims is enough for a proof-of-concept, not enough to draw strong conclusions. Reported numbers should be read with wide error bars.
- **NLI is trigger-happy on paraphrases.** MNLI-trained cross-encoders often return `neutral` on genuine paraphrases when phrasing diverges from the source. This is the dominant cause of `WEAK` verdicts on grounded claims here — even after sentence-level retrieval, 7 of 10 grounded claims still land in `WEAK`. A production version would fine-tune the NLI head on scientific paraphrase pairs.
- **Abstracts, not full papers.** Real hallucinations often happen in claims about methods or results buried deep in a paper. Abstract-only grounding is a lower bound on the real difficulty.
- **Single NLI model.** `cross-encoder/nli-deberta-v3-base` is decent but not state of the art. A production version would ensemble multiple NLI models.
- **No adversarial claim generation.** Every fabricated claim was human-written. A stronger evaluation would include LLM-generated distractors calibrated to be plausible.
- **Single evaluator.** Claim labels were written by one person (the author). Two-annotator agreement on the boundary cases (subtle scope_overreach vs. paraphrase) would tighten the ground truth.

 ### Selected failure cases

**False alarm — grounded paraphrase flagged as WEAK (C007)**
- Claim: "RAG sets a new state of the art on three open-domain question answering tasks."
- Verdict: WEAK (sim=0.46, NLI=neutral 0.97)
- What went wrong: The abstract says "set the state-of-the-art on three open domain QA tasks" — nearly identical wording. NLI still returned neutral with high confidence. This is the dominant failure mode: MNLI-trained cross-encoders are conservative even on lexically overlapping paraphrases.

  **False alarm — threshold-tuning miss (C021)**
- Claim: "On a controlled financial QA dataset with GPT-3.5-turbo,ECLIPSEachieves ROC AUC of 0.89 and average precision of 0.90"
- Verdict: WEAK (sim=0.46, NLI=entailment 1.00)
- What went wrong: NLI correctly returned entailment with 1.00 confidence, but similarity (0.46) fell below the sim_high=0.55 threshold. A pure threshold-tuning issue — lowering sim_high to 0.40 would flip this to GROUNDED. Suggests a defensible v3 improvement.

**True positive — subtle numeric distortion caught (C023)**
- Claim: "On the financial QA benchmark, ECLIPSE achieves ROC AUC of 0.94 and average precision of 0.95"
- Verdict: UNGROUNDED (sim=0.38, NLI=contradiction 0.59)
- Why it worked: NLI detected the swapped numbers (0.89 → 0.94) as a contradiction, though at only 0.59 confidence — the subtlest fabrication in the dataset. Borderline but correct.
  

## Repo layout

```
veritus-grounding-harness/
├── src/
│   ├── collect_papers.py    # arXiv fetcher → data/papers.json
│   ├── harness.py           # embedding + NLI + verdict logic
│   └── eval.py              # P/R/F1 + confusion matrix + per-type breakdown
├── scripts/
│   └── run_pipeline.py      # end-to-end runner
├── data/
│   ├── seed_papers.json     # 2 hand-picked abstracts for smoke testing
│   ├── papers.json          # (generated) full fetched set
│   └── claims.csv           # labeled claim set
├── results/                 # (generated) harness_results.csv, eval_details.csv
├── requirements.txt
└── README.md
```
 ## Contact
  Jaya Arora — jayaarora2402@gmail.com — [LinkedIn](https://www.linkedin.com/in/jaya-arora-6892a93a0/)
