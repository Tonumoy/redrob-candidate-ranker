# HuggingFace Space deployment

The live sandbox (https://huggingface.co/spaces/Bodhi108/redrob-candidate-ranker)
is a **Docker** Space (HF deprecated the built-in Streamlit SDK on 2025-04-30, so
Streamlit now runs via the Docker template).

This folder holds the HF-specific config:

- `Dockerfile` — installs `requirements.txt` and runs `streamlit run app.py` on
  port 8501 as the non-root user HF expects. XSRF/CORS are disabled so file
  uploads work inside HF's embedding iframe.
- `requirements.txt` — `streamlit`, `numpy`, `scikit-learn` (note: unlike the
  repo-root `requirements.txt`, this one installs Streamlit explicitly).
- `README.md` — the Space card with the `sdk: docker` / `app_port: 8501` YAML.

To build the Space, this folder is combined with the repo's `app.py`, `src/`,
`demo_sample.jsonl`, and `sample.jsonl` (copied alongside these files), then
pushed to the HF Space repo (e.g. via `huggingface_hub.HfApi().upload_folder`).
The app uses the **TF-IDF** backend only — no model download, no network, runs in
seconds on free CPU.
