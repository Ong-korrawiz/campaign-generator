#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${1:?Usage: $0 PROJECT_ID REGION DATASET_URI TRAINING_IMAGE RUN_ID base|tuned [ADAPTER_URI]}"
REGION="${2:?Usage: $0 PROJECT_ID REGION DATASET_URI TRAINING_IMAGE RUN_ID base|tuned [ADAPTER_URI]}"
DATASET_URI="${3:?Usage: $0 PROJECT_ID REGION DATASET_URI TRAINING_IMAGE RUN_ID base|tuned [ADAPTER_URI]}"
TRAINING_IMAGE="${4:?Usage: $0 PROJECT_ID REGION DATASET_URI TRAINING_IMAGE RUN_ID base|tuned [ADAPTER_URI]}"
RUN_ID="${5:?Usage: $0 PROJECT_ID REGION DATASET_URI TRAINING_IMAGE RUN_ID base|tuned [ADAPTER_URI]}"
MODE="${6:?Prediction mode must be base or tuned}"
ADAPTER_URI="${7:-}"
[[ "${MODE}" == "base" || "${MODE}" == "tuned" ]] || { echo "MODE must be base or tuned" >&2; exit 2; }
if [[ "${MODE}" == "tuned" && -z "${ADAPTER_URI}" ]]; then
  echo "ADAPTER_URI is required for tuned prediction" >&2
  exit 2
fi
[[ "${TRAINING_IMAGE}" =~ @sha256:[0-9a-f]{64}$ ]] || { echo "TRAINING_IMAGE must be digest-pinned" >&2; exit 2; }
[[ "${DATASET_URI}" == gs://* ]] || { echo "DATASET_URI must be a GCS version URI" >&2; exit 2; }
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TEMPLATE="${REPO_ROOT}/infra/training/prediction-job.yaml.tpl"
CONFIG="$(mktemp --suffix=.yaml)"
trap 'rm -f "${CONFIG}"' EXIT
OUTPUT_URI="gs://${PROJECT_ID}-model-artifacts/runs/${RUN_ID}/evaluation/${MODE}.jsonl"
TEST_FILE_URI="${DATASET_URI%/}/test.jsonl"
TRAINING_SA="campaign-training@${PROJECT_ID}.iam.gserviceaccount.com"
if [[ -n "${ADAPTER_URI}" ]]; then
  ADAPTER_ARG="        - --adapter=${ADAPTER_URI}"
else
  ADAPTER_ARG=""
fi
python3 - "${TEMPLATE}" "${CONFIG}" "${TRAINING_SA}" "${TRAINING_IMAGE}" "${TEST_FILE_URI}" "${OUTPUT_URI}" "${ADAPTER_ARG}" <<'PY'
from pathlib import Path
import sys
source, target, service_account, image, test_uri, output_uri, adapter_arg = sys.argv[1:]
content = Path(source).read_text(encoding="utf-8")
for marker, value in (
    ("${TRAINING_SERVICE_ACCOUNT}", service_account),
    ("${TRAINING_IMAGE_URI}", image),
    ("${TEST_FILE_URI}", test_uri),
    ("${OUTPUT_URI}", output_uri),
    ("${ADAPTER_ARG}", adapter_arg),
):
    content = content.replace(marker, value)
Path(target).write_text(content, encoding="utf-8")
PY
gcloud ai custom-jobs create --project="${PROJECT_ID}" --region="${REGION}" \
  --display-name="campaign-${MODE}-predictions-${RUN_ID}" --config="${CONFIG}"
