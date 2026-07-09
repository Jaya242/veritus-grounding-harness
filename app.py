import streamlit as st

from src.harness import CombinedHarness

st.set_page_config(page_title="Veritus Grounding Harness", page_icon="📚", layout="centered")


@st.cache_resource(show_spinner="Loading embedding + NLI models (~500MB, one-time)...")
def load_harness() -> CombinedHarness:
    return CombinedHarness()


st.title("Research Paper Grounding Harness")
st.markdown(
    "Checks whether a research claim is actually supported by a paper abstract. "
    "Combines sentence-transformer similarity with a cross-encoder NLI model — verdict is "
    "`GROUNDED`, `UNGROUNDED`, or `WEAK` (flag for human review)."
)

harness = load_harness()

claim = st.text_input(
    "Research claim",
    placeholder="Example: The model achieved 95% accuracy on ImageNet.",
)
abstract = st.text_area(
    "Paper abstract",
    height=240,
    placeholder="Paste the research paper abstract here...",
)

if st.button("Check grounding", type="primary", disabled=not (claim and abstract)):
    with st.spinner("Scoring..."):
        result = harness.evaluate(claim_id="demo", claim=claim, passage=abstract)

    verdict_color = {"GROUNDED": "green", "UNGROUNDED": "red", "WEAK": "orange"}[result.verdict.value]
    st.markdown(f"### Verdict: :{verdict_color}[**{result.verdict.value}**]")

    col1, col2, col3 = st.columns(3)
    col1.metric("Semantic similarity", f"{result.similarity:.3f}")
    col2.metric("NLI label", result.nli_label)
    col3.metric("NLI confidence", f"{result.nli_confidence:.3f}")

    st.caption(f"Reasoning: {result.reasoning}")
