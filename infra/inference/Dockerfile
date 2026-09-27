FROM vllm/vllm-openai:v0.8.5

ARG MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct
ARG MODEL_REVISION=989aa7980e4cf806f80c7fef2b1adb7bc71aa306
ARG MODEL_DIR=/models/qwen2.5-1.5b-instruct

ENV MODEL_DIR=${MODEL_DIR} \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_OFFLINE=0 \
    TRANSFORMERS_OFFLINE=0

RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='${MODEL_ID}', revision='${MODEL_REVISION}', local_dir='${MODEL_DIR}')"

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

COPY entrypoint.sh /opt/campaign/entrypoint.sh
RUN chmod 0555 /opt/campaign/entrypoint.sh

ENTRYPOINT ["/opt/campaign/entrypoint.sh"]
