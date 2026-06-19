# Streamlit on HuggingFace Spaces via the Docker template (Streamlit SDK was
# deprecated 2025-04-30). Runs as the non-root user HF expects, on port 8501.
FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY --chown=user . .

EXPOSE 8501
# XSRF/CORS disabled so file uploads work inside the HF Spaces iframe/proxy
# (Streamlit's XSRF token fails cross-origin in the embedded iframe). Safe for a
# stateless public demo.
ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.port=8501", "--server.address=0.0.0.0", \
            "--server.headless=true", "--browser.gatherUsageStats=false", \
            "--server.enableXsrfProtection=false", "--server.enableCORS=false"]
