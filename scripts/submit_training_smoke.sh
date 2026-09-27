#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${1:?Usage: $0 PROJECT_ID [REGION]}"
REGION="${2:-asia-southeast1}"
RUN_ID="smoke-$(date -u +%Y%m%d-%H%M%S)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TEMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TEMP_DIR}"' EXIT

export TRAINING_SERVICE_ACCOUNT="campaign-training@${PROJECT_ID}.iam.gserviceaccount.com"
export INPUT_URI="gs://${PROJECT_ID}-dataset/smoke/input.txt"
export OUTPUT_URI="gs://${PROJECT_ID}-model-artifacts/runs/${RUN_ID}"

printf 'vertex-training-smoke-input\n' > "${TEMP_DIR}/input.txt"
gcloud storage cp "${TEMP_DIR}/input.txt" "${INPUT_URI}" --project="${PROJECT_ID}"
envsubst < "${REPO_ROOT}/infra/training/smoke-job.yaml.tpl" > "${TEMP_DIR}/smoke-job.yaml"

gcloud ai custom-jobs create \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --display-name="campaign-training-${RUN_ID}" \
  --config="${TEMP_DIR}/smoke-job.yaml"

printf 'Expected result: %s/smoke-result.txt\n' "${OUTPUT_URI}"
