FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY infra/api/requirements-api.txt /app/requirements-api.txt
RUN pip install --no-cache-dir -r /app/requirements-api.txt
COPY pyproject.toml README.md /app/
COPY src /app/src
RUN pip install --no-cache-dir --no-deps -e .

ENV PORT=8080
CMD ["sh", "-c", "uvicorn campaign_generator.api.app:app --host 0.0.0.0 --port ${PORT:-8080}"]
