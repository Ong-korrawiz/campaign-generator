#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${1:?Usage: $0 PROJECT_ID REGION DATASET_URI TRAINING_IMAGE RUN_ID}"
REGION="${2:?Usage: $0 PROJECT_ID REGION DATASET_URI TRAINING_IMAGE RUN_ID}"
DATASET_URI="${3:?Usage: $0 PROJECT_ID REGION DATASET_URI TRAINING_IMAGE RUN_ID}"
TRAINING_IMAGE="${4:?Usage: $0 PROJECT_ID REGION DATASET_URI TRAINING_IMAGE RUN_ID}"
RUN_ID="${5:?Usage: $0 PROJECT_ID REGION DATASET_URI TRAINING_IMAGE RUN_ID}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TEMPLATE="${REPO_ROOT}/infra/training/custom-job.yaml.tpl"
CONFIG="$(mktemp --suffix=.yaml)"
trap 'rm -f "${CONFIG}"' EXIT

[[ "${TRAINING_IMAGE}" =~ @sha256:[0-9a-f]{64}$ ]] || {
  echo "TRAINING_IMAGE must be an immutable @sha256 reference" >&2
  exit 1
}
if [[ "${DATASET_URI}" != gs://* ]]; then
  echo "DATASET_URI must be a private gs:// dataset version" >&2
  exit 1
fi
TRAINING_SA="campaign-training@${PROJECT_ID}.iam.gserviceaccount.com"
OUTPUT_URI="gs://${PROJECT_ID}-model-artifacts/runs/${RUN_ID}"
if gcloud storage ls "${OUTPUT_URI}/" >/dev/null 2>&1; then
  echo "RUN_ID already has artifacts; choose a unique run ID to preserve prior results: ${RUN_ID}" >&2
  exit 1
fi
CODE_COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
python3 - "${TEMPLATE}" "${CONFIG}" "${TRAINING_SA}" "${TRAINING_IMAGE}" "${DATASET_URI}" "${OUTPUT_URI}" "${RUN_ID}" "${CODE_COMMIT}" <<'PY'
from pathlib import Path
import sys
source, target, *values = sys.argv[1:]
keys = ("${TRAINING_SERVICE_ACCOUNT}", "${TRAINING_IMAGE_URI}", "${DATASET_URI}", "${OUTPUT_URI}", "${RUN_ID}", "${CODE_COMMIT}")
content = Path(source).read_text(encoding="utf-8")
for key, value in zip(keys, values):
    content = content.replace(key, value)
Path(target).write_text(content, encoding="utf-8")
PY
gcloud ai custom-jobs create --project="${PROJECT_ID}" --region="${REGION}" \
  --display-name="campaign-lora-${RUN_ID}" --config="${CONFIG}"
