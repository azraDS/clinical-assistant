FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/app/.cache/huggingface

RUN useradd --create-home app
WORKDIR /app

# Dependencies first so code edits don't invalidate the (large) torch/docling layer.
# CPU-only torch keeps the image far smaller than the default CUDA wheels.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -e ".[app]"

COPY app.py ./
# The HF cache dir must exist (owned by app) in the image, or the named volume mounted there
# is created root-owned and model downloads fail with PermissionError.
RUN mkdir -p notebooks/data notebooks/outputs /home/app/.cache/huggingface \
    && chown -R app:app /app /home/app
USER app

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8501/_stcore/health')"

# The KDIGO guideline PDF is copyrighted and not in the repo: mount it via docker-compose.yml.
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
