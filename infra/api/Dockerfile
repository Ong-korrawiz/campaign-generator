FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.7.19 /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml uv.lock README.md /app/
COPY src /app/src
RUN uv sync --locked --no-dev

ENV PORT=8080
CMD ["sh", "-c", "/app/.venv/bin/uvicorn campaign_generator.api.app:app --host 0.0.0.0 --port ${PORT:-8080}"]
