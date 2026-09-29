#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${1:?Usage: $0 PROJECT_ID REGION DATASET_URI RUN_ID}"
REGION="${2:?Usage: $0 PROJECT_ID REGION DATASET_URI RUN_ID}"
DATASET_URI="${3:?Usage: $0 PROJECT_ID REGION DATASET_URI RUN_ID}"
RUN_ID="${4:?Usage: $0 PROJECT_ID REGION DATASET_URI RUN_ID}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/training_image.sh"

[[ "${DATASET_URI}" == gs://* ]] || { echo "DATASET_URI must start with gs://" >&2; exit 2; }
[[ "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
  echo "RUN_ID may contain only letters, numbers, dots, underscores, and hyphens" >&2
  exit 2
}

run_step() {
  local label="$1"
  shift
  echo "RUN: ${label}"
  if "$@"; then
    echo "OK: ${label}"
  else
    local status=$?
    echo "FAILED: ${label} (exit ${status})" >&2
    return "${status}"
  fi
}

resolve_training_image() {
  local image
  image="$(training_image_tag "${PROJECT_ID}" "${REGION}" "${REPO_ROOT}")"
  TRAINING_IMAGE="$(gcloud artifacts docker images describe "${image}" \
    --project="${PROJECT_ID}" \
    --format='value(image_summary.fully_qualified_digest)')"
  [[ "${TRAINING_IMAGE}" =~ @sha256:[0-9a-f]{64}$ ]] || {
    echo "Unable to resolve immutable training image digest for ${image}" >&2
    return 1
  }
  echo "Training image: ${TRAINING_IMAGE}"
}

check_run_id() {
  if gcloud storage ls "gs://${PROJECT_ID}-model-artifacts/runs/${RUN_ID}/" >/dev/null 2>&1; then
    echo "RUN_ID already has artifacts: ${RUN_ID}" >&2
    return 1
  fi
}

run_step "check dataset manifest" gcloud storage ls "${DATASET_URI%/}/manifest.json"
run_step "check training split" gcloud storage ls "${DATASET_URI%/}/train.jsonl"
run_step "check validation split" gcloud storage ls "${DATASET_URI%/}/validation.jsonl"
run_step "check RUN_ID" check_run_id
run_step "build training image" "${SCRIPT_DIR}/build_training.sh" "${PROJECT_ID}" "${REGION}"
run_step "resolve training image digest" resolve_training_image
run_step "submit Vertex training job" "${SCRIPT_DIR}/submit_training.sh" \
  "${PROJECT_ID}" "${REGION}" "${DATASET_URI}" "${TRAINING_IMAGE}" "${RUN_ID}"
