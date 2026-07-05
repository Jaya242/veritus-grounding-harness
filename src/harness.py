"""Grounding harness: does the passage actually support the claim?

Two independent signals per (claim, passage) pair:
  1. Semantic similarity via sentence-transformers (are they topically related?)
  2. Entailment via a cross-encoder NLI model (does the passage support the claim?)

Combined into one verdict: GROUNDED / UNGROUNDED / WEAK.
WEAK = the two signals disagree or neither is confident → flag for human review.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional


def _split_sentences(text: str) -> list[str]:
    """Simple sentence splitter — good enough for well-punctuated abstracts.

    Drops fragments shorter than 15 chars (headings, artifacts) so we don't
    hand NLI a passage like 'et al.'.
    """
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sents if len(s) >= 15]


class Verdict(str, Enum):
    GROUNDED = "GROUNDED"
    UNGROUNDED = "UNGROUNDED"
    WEAK = "WEAK"


@dataclass
class GroundingResult:
    claim_id: str
    similarity: float
    nli_label: str          # entailment / neutral / contradiction
    nli_confidence: float
    verdict: Verdict
    reasoning: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d["verdict"] = self.verdict.value
        return d


class EmbeddingScorer:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def score(self, claim: str, passage: str) -> float:
        from sentence_transformers import util
        emb = self.model.encode([claim, passage], convert_to_tensor=True)
        return float(util.cos_sim(emb[0], emb[1]).item())


class EntailmentScorer:
    """Wraps a cross-encoder NLI model.

    cross-encoder/nli-deberta-v3-base outputs 3 scores per pair:
    [contradiction, entailment, neutral]. We take the argmax as the label
    and its softmax probability as the confidence.
    """

    LABELS = ["contradiction", "entailment", "neutral"]

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-base"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)

    def score(self, claim: str, passage: str) -> tuple[str, float]:
        import numpy as np
        # NLI convention: input is (premise, hypothesis). Passage = premise, claim = hypothesis.
        raw = self.model.predict([(passage, claim)])[0]
        # raw is a length-3 array of logits — softmax → probabilities
        exp = np.exp(raw - np.max(raw))
        probs = exp / exp.sum()
        idx = int(np.argmax(probs))
        return self.LABELS[idx], float(probs[idx])


class CombinedHarness:
    """Combines embedding similarity + NLI into a single verdict.

    Decision logic (kept simple + explicit so it's easy to defend in a write-up):
      - NLI contradiction, confident            → UNGROUNDED
      - NLI entailment + similarity above HIGH  → GROUNDED
      - NLI neutral + similarity below LOW      → UNGROUNDED
      - anything else                           → WEAK (flag for review)

    Thresholds are hyperparameters — tune them on your labeled dev set,
    then report the final numbers.
    """

    def __init__(
        self,
        sim_high: float = 0.55,
        sim_low: float = 0.30,
        nli_conf_min: float = 0.55,
        use_sentence_retrieval: bool = True,
        embedding_scorer: Optional[EmbeddingScorer] = None,
        entailment_scorer: Optional[EntailmentScorer] = None,
    ):
        self.sim_high = sim_high
        self.sim_low = sim_low
        self.nli_conf_min = nli_conf_min
        self.use_sentence_retrieval = use_sentence_retrieval
        self.embed = embedding_scorer or EmbeddingScorer()
        self.nli = entailment_scorer or EntailmentScorer()

    def _pick_best_sentence(self, claim: str, passage: str) -> str:
        """SNLI/MNLI-trained NLI models dilute on long premises. Give them the
        single sentence most likely to entail the claim, not the whole abstract."""
        from sentence_transformers import util
        sentences = _split_sentences(passage)
        if len(sentences) <= 1:
            return passage
        embs = self.embed.model.encode([claim] + sentences, convert_to_tensor=True)
        sims = util.cos_sim(embs[0:1], embs[1:])[0]
        best_idx = int(sims.argmax().item())
        return sentences[best_idx]

    def evaluate(self, claim_id: str, claim: str, passage: str) -> GroundingResult:
        # Similarity uses the full passage (topical relevance is a whole-abstract signal)
        # NLI uses just the best-matching sentence (strict entailment is sentence-scoped)
        sim = self.embed.score(claim, passage)
        nli_passage = self._pick_best_sentence(claim, passage) if self.use_sentence_retrieval else passage
        label, conf = self.nli.score(claim, nli_passage)

        if label == "contradiction" and conf >= self.nli_conf_min:
            verdict = Verdict.UNGROUNDED
            reasoning = f"NLI=contradiction ({conf:.2f})"
        elif label == "entailment" and sim >= self.sim_high and conf >= self.nli_conf_min:
            verdict = Verdict.GROUNDED
            reasoning = f"NLI=entailment ({conf:.2f}), sim={sim:.2f}"
        elif label == "neutral" and sim < self.sim_low:
            verdict = Verdict.UNGROUNDED
            reasoning = f"NLI=neutral ({conf:.2f}), low sim={sim:.2f}"
        else:
            verdict = Verdict.WEAK
            reasoning = f"NLI={label} ({conf:.2f}), sim={sim:.2f} → ambiguous"

        return GroundingResult(
            claim_id=claim_id,
            similarity=sim,
            nli_label=label,
            nli_confidence=conf,
            verdict=verdict,
            reasoning=reasoning,
        )
