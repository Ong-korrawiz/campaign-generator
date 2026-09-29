FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

COPY --from=ghcr.io/astral-sh/uv:0.7.19 /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app
COPY pyproject.toml uv.lock README.md /app/
COPY src /app/src
RUN uv venv --system-site-packages && uv sync --locked --no-dev --extra training --no-install-package torch

ENTRYPOINT ["/app/.venv/bin/python", "-m", "campaign_generator.training.train_lora"]
