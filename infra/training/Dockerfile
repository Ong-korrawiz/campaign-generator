FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app
COPY infra/training/requirements-training.txt /app/requirements-training.txt
RUN pip install --no-cache-dir -r /app/requirements-training.txt
COPY pyproject.toml README.md /app/
COPY src /app/src
RUN pip install --no-cache-dir --no-deps -e .

ENTRYPOINT ["python", "-m", "campaign_generator.training.train_lora"]
