FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_DIR=/app/models

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

# Train at build time on the latest data, so every deploy (including the weekly scheduled one)
# ships a model trained up to that day. The backtest results are baked into /model.
# CI overrides this with "--synthetic --days 60 --skip-backtest" so builds don't depend on the live API.
ARG TRAIN_ARGS="--days 365 --folds 6"
RUN python -m carbon_forecast.train ${TRAIN_ARGS}

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.getenv(\"PORT\",\"8000\")}/health', timeout=4)"

# Render (and most platforms) inject PORT.
CMD ["sh", "-c", "uvicorn carbon_forecast.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
