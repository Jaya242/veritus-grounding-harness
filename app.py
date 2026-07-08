import gradio as gr
from src.harness import CombinedHarness

# Load the model once when the Space starts
harness = CombinedHarness()


def check_grounding(claim, abstract):
    result = harness.evaluate(
        claim_id="demo",
        claim=claim,
        passage=abstract
    )

    return (
        result.verdict.value,
        round(result.similarity, 3),
        result.nli_label,
        round(result.nli_confidence, 3),
        result.reasoning,
    )


demo = gr.Interface(
    fn=check_grounding,
    inputs=[
        gr.Textbox(
            label="Research Claim",
            placeholder="Example: The model achieved 95% accuracy on ImageNet."
        ),
        gr.Textbox(
            label="Paper Abstract",
            lines=10,
            placeholder="Paste the research paper abstract here..."
        ),
    ],
    outputs=[
        gr.Textbox(label="Verdict"),
        gr.Number(label="Semantic Similarity"),
        gr.Textbox(label="NLI Label"),
        gr.Number(label="NLI Confidence"),
        gr.Textbox(label="Reasoning"),
    ],
    title="Research Paper Grounding Harness",
    description="Checks whether a research claim is actually supported by a paper abstract.",
)

demo.launch()
