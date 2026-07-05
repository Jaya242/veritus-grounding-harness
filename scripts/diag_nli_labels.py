"""Sanity check: does the NLI model's label mapping match what harness.py assumes?

harness.py hardcodes LABELS = ['contradiction', 'entailment', 'neutral']. If that
order is wrong, entailment scores get read as contradictions and vice-versa —
which would explain paraphrases getting labeled 'neutral'.

Run:  python scripts/diag_nli_labels.py
"""
from __future__ import annotations

import numpy as np
from sentence_transformers import CrossEncoder

model = CrossEncoder("cross-encoder/nli-deberta-v3-base")

# Check the model's own id2label first
try:
    id2label = model.model.config.id2label
    print(f"Model's id2label: {id2label}")
except AttributeError:
    print("(no id2label attribute — checking with known test cases)")

# Three test pairs with known answers under standard NLI
tests = [
    ("A man is playing a guitar on stage.", "Someone is performing music.", "entailment"),
    ("A man is playing a guitar on stage.", "A woman is baking bread in a kitchen.", "contradiction"),
    ("A man is playing a guitar on stage.", "The audience enjoyed the concert.", "neutral"),
]

print("\nTest predictions (each row is raw model output — 3 scores per pair):")
LABELS_GUESS = ["contradiction", "entailment", "neutral"]  # harness.py's assumption
for premise, hypothesis, expected in tests:
    raw = model.predict([(premise, hypothesis)])[0]
    exp = np.exp(raw - np.max(raw))
    probs = exp / exp.sum()
    idx = int(np.argmax(probs))
    predicted = LABELS_GUESS[idx]
    ok = "OK" if predicted == expected else "WRONG"
    print(f"  {ok:5s}  expected={expected:14s} predicted={predicted:14s} probs={dict(zip(LABELS_GUESS, probs.round(3).tolist()))}")

print("\nIf all three show OK → harness.py's label order is correct.")
print("If any show WRONG → the LABELS order in src/harness.py needs to match id2label above.")
