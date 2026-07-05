# Error notes — v2 (sentence-level retrieval + NLI), N=25

  ## False alarms (7): grounded claims that got WEAK
  
  ### C001 — ViT paraphrase
  - Claim: "Vision Transformer applies a pure transformer directly to sequences of image patches for
  classification."
  - Verdict: WEAK (sim=0.60, NLI=neutral 0.93)
  - Why it went wrong: Near-verbatim paraphrase — the abstract literally says "a pure transformer
  applied directly to sequences of image patches can perform very well on image classification tasks."
  NLI still returned neutral. Note: v1 (paragraph NLI) correctly labeled this GROUNDED — sentence
  retrieval regressed it.

  ### C007 — RAG SOTA claim
  - Claim: "RAG sets a new state of the art on three open-domain question answering tasks."
  - Verdict: WEAK (sim=0.46, NLI=neutral 0.97)
  - Why it went wrong: Abstract says "set the state-of-the-art on three open domain QA tasks" — nearly
  identical. NLI still uncertain. This is the dominant failure mode: MNLI-trained NLI is conservative
  even on lexically overlapping paraphrases.

  ### C011 — Fisher Vectors rotation robustness
  - Claim: "Fisher Vectors show natural resilience to rotations while CNNs lack a built-in mechanism
  for rotation invariance."
  - Verdict: WEAK (sim=0.50, NLI=neutral 1.00)
  - Why it went wrong: Abstract phrases it as "intrinsically resilient" vs. our "naturally resilient."
  Small lexical shift, same meaning — NLI doesn't bridge it.
  
  ### C012 — CNN vs FV no clear winner
  - Claim: "Neither CNN features nor Fisher Vectors consistently outperform the other across
  benchmarks, and combining them typically yields the best retrieval results."
  - Verdict: WEAK (sim=0.68, NLI=neutral 1.00)
  - Why it went wrong: Highest similarity of any false alarm (0.68). Heavy paraphrase — different
  sentence structure but same meaning. NLI still neutral.
  
  ### C016 — PSRD 50% claim ⚠️  threshold-tuning miss
  - Claim: "PSRD reduces the hallucination rate of LLaVA-1.5-7B by 50.0% and outperforms existing
  post-hoc methods across five hallucination benchmarks"
  - Verdict: WEAK (sim=0.40, NLI=entailment 1.00)
  - Why it went wrong: **NLI got it right — entailment with 1.00 confidence!** But sim=0.40 is below
  sim_high=0.55 threshold. Pure threshold-tuning issue. Lowering sim_high to 0.40 would flip this to
  GROUNDED without breaking fabrication catches.

  ### C017 — PSRD phase-wise pattern
  - Claim: "Vision hallucinations in LVLMs follow a temporal pattern where they spike at the onset of
  each new semantic phase during generation."
  - Verdict: WEAK (sim=0.59, NLI=neutral 1.00)
  - Why it went wrong: Uses "spike" and "temporal pattern" vs. abstract's "peaking" and "dynamic
  pattern." Same meaning, NLI treats as neutral.
  
  ### C021 — ECLIPSE AUC claim ⚠️  threshold-tuning miss
  - Claim: "On a controlled financial QA dataset with GPT-3.5-turbo, ECLIPSE achieves ROC AUC of 0.89
  and average precision of 0.90"
  - Verdict: WEAK (sim=0.46, NLI=entailment 1.00)
  - Why it went wrong: **Same pattern as C016 — NLI entailment 1.00 but similarity below threshold.**
  Second threshold-tuning miss.
  
  ## Correct catches (spot-check, 3)

  ### C003 — ViT numeric distortion (caught cleanly)
  - Claim: "Vision Transformer achieves 96.5% top-1 accuracy on ImageNet without any pretraining."
  - Verdict: UNGROUNDED (sim=0.61, NLI=contradiction 1.00)
  - Why it worked: NLI detected the "without any pretraining" as contradicting the abstract's
  pretraining requirement. Confidence 1.00 → clean flag.
  
  ### C018 — PSRD subtle number swap (caught)
  - Claim: "PSRD reduces the hallucination rate of LLaVA-1.5-7B by 65% across evaluation benchmarks"
  - Verdict: UNGROUNDED (sim=0.36, NLI=contradiction 0.99)
  - Why it worked: The single number change from 50% → 65% was caught as contradiction. Encouraging —
  NLI is sensitive to small numeric distortions.
  
  ### C023 — ECLIPSE subtle number swap (caught, weakly)
  - Claim: "On the financial QA benchmark, ECLIPSE achieves ROC AUC of 0.94 and average precision of
  0.95"
  - Verdict: UNGROUNDED (sim=0.38, NLI=contradiction 0.59)
  - Why it worked: NLI caught the 0.89→0.94 swap but with only 0.59 confidence — the subtlest
  fabrication in the dataset. Borderline but correct.
  
  ## Two failure patterns

  1. **NLI-is-conservative** (C001, C007, C011, C012, C017): NLI returns neutral on genuine paraphrases
   even with high lexical overlap. Not fixable by threshold tuning — needs a different NLI model or
  fine-tuning.
  2. **Threshold-tuning misses** (C016, C021): NLI got it RIGHT (entailment 1.00) but sim_high=0.55 
  threshold rejected them. Lowering sim_high to 0.40 would flip both to GROUNDED. Worth trying if
  there's time.
  
