# Smart MCQ Solver — Modal deployment

BiLSTM Siamese ranker for 5-option MCQs, served on Modal as a Gradio UI plus a
JSON API.

## Layout

    modal_app.py            Modal app: image, mounts, Gradio UI, FastAPI routes
    mcq_model.py            model definition + preprocessing + Solver
    deploy_artifacts/
        model_weights.pt    state_dict of BiLSTMSiamese
        vocab.pkl           word2id
        config.json         hyperparameters + val MAP@3

## Deploy

    pip install modal
    modal setup
    modal serve modal_app.py      # dev, live reload
    modal deploy modal_app.py     # persistent URL

## Endpoints

    GET  /              Gradio UI
    GET  /health        model status and val MAP@3
    POST /api/solve     {"prompt": "...", "options": ["A","B","C","D","E"]}
