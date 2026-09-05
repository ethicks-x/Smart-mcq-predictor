"""
Deploy the Smart MCQ Solver on Modal.

    modal serve modal_app.py     # live-reloading dev URL
    modal deploy modal_app.py    # persistent URL

Expects ./deploy_artifacts/ next to this file, containing model_weights.pt,
vocab.pkl and config.json as written by the notebook export cell.
"""

from pathlib import Path
import modal

ARTIFACTS_REMOTE = "/root/deploy_artifacts"

image = (
    modal.Image.debian_slim(python_version="3.12")
    # CPU-only torch. The +cpu local version sorts above the PyPI wheel, so pip
    # picks it from the extra index and skips the ~2.5 GB CUDA build.
    .pip_install("torch", extra_index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("fastapi[standard]", "gradio")
    # Mounted at container start rather than baked into a layer, so swapping in
    # retrained weights does not trigger an image rebuild.
    .add_local_dir("deploy_artifacts", remote_path=ARTIFACTS_REMOTE)
    .add_local_python_source("mcq_model")
)

app = modal.App("smart-mcq-solver", image=image)


@app.function(
    cpu=1.0,
    memory=2048,
    scaledown_window=300,   # stay warm 5 min after the last request
    max_containers=1,       # Gradio needs sticky sessions
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def ui():
    import gradio as gr
    from fastapi import FastAPI
    from gradio.routes import mount_gradio_app
    from pydantic import BaseModel

    from mcq_model import Solver

    solver = Solver(ARTIFACTS_REMOTE)   # runs once per container
    api = FastAPI()

    class Question(BaseModel):
        prompt: str
        options: list[str]

    @api.post("/api/solve")
    def solve_api(q: Question):
        if len(q.options) != 5:
            return {"error": "exactly 5 options required"}
        prediction, confidences = solver.solve(q.prompt, q.options)
        return {"prediction": prediction, "probabilities": confidences}

    @api.get("/health")
    def health():
        return {"status": "ok", "val_map3": solver.cfg.get("val_map3")}

    def solve_ui(prompt, a, b, c, d, e):
        if not prompt or not prompt.strip():
            return "Enter a question first.", {}
        return solver.solve(prompt, [a, b, c, d, e])

    with gr.Blocks() as demo:
        gr.Markdown(
            "# Smart MCQ Solver\n"
            "BiLSTM Siamese ranker. Enter a question and five options; "
            "the model returns its top-3 ranking in MAP@3 format."
        )
        prompt = gr.Textbox(label="Question", lines=2)
        with gr.Row():
            a = gr.Textbox(label="A")
            b = gr.Textbox(label="B")
        with gr.Row():
            c = gr.Textbox(label="C")
            d = gr.Textbox(label="D")
        e = gr.Textbox(label="E")

        btn = gr.Button("Solve", variant="primary")
        out_pred = gr.Textbox(label="Top-3 prediction")
        out_probs = gr.Label(label="Per-option probability", num_top_classes=5)

        btn.click(solve_ui, [prompt, a, b, c, d, e], [out_pred, out_probs])

        gr.Examples(
            examples=[[
                "What is the proposed name for the field that is responsible for cosmic inflation and the metric expa...",
                "Inflation", "Quanta", "Scalar", "Metric", "Conformal cyclic cosmology",
            ]],
            inputs=[prompt, a, b, c, d, e],
        )

    return mount_gradio_app(app=api, blocks=demo, path="/")
