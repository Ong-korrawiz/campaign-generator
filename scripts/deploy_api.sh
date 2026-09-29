#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${1:?Usage: $0 PROJECT_ID [API_IMAGE_DIGEST] [REGION]}"
API_IMAGE="${2:-}"
REGION="${3:-asia-southeast1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/api_image.sh"
STATE_BUCKET="${PROJECT_ID}-tfstate"

if [[ -z "${API_IMAGE}" ]]; then
  IMAGE="$(api_image_tag "${PROJECT_ID}" "${REGION}" "${REPO_ROOT}")"
  API_IMAGE="$(gcloud artifacts docker images describe "${IMAGE}" \
    --project="${PROJECT_ID}" \
    --format='value(image_summary.fully_qualified_digest)')" || {
    echo "Unable to resolve the digest for ${IMAGE}; run make build-api first" >&2
    exit 1
  }
  [[ -n "${API_IMAGE}" ]] || {
    echo "No digest found for ${IMAGE}; run make build-api first" >&2
    exit 1
  }
  printf 'Deploying %s\n' "${API_IMAGE}"
fi

export GOOGLE_OAUTH_ACCESS_TOKEN
GOOGLE_OAUTH_ACCESS_TOKEN="$(gcloud auth print-access-token)"

[[ "${API_IMAGE}" =~ @sha256:[0-9a-f]{64}$ ]] || {
  echo "API_IMAGE must be an immutable @sha256 reference" >&2
  exit 1
}
INFERENCE_URL="$(terraform -chdir="${REPO_ROOT}/terraform/serving" output -raw inference_url)"
[[ -n "${INFERENCE_URL}" ]] || { echo "Private inference URL is missing" >&2; exit 1; }

printf 'bucket = "%s"\nprefix = "campaign-generator/api"\n' "${STATE_BUCKET}" \
  > "${REPO_ROOT}/terraform/api/backend.hcl"
terraform -chdir="${REPO_ROOT}/terraform/api" init -reconfigure -backend-config=backend.hcl
terraform -chdir="${REPO_ROOT}/terraform/api" validate
terraform -chdir="${REPO_ROOT}/terraform/api" plan \
  -var="project_id=${PROJECT_ID}" -var="region=${REGION}" \
  -var="api_image=${API_IMAGE}" -var="inference_url=${INFERENCE_URL}"
terraform -chdir="${REPO_ROOT}/terraform/api" apply \
  -var="project_id=${PROJECT_ID}" -var="region=${REGION}" \
  -var="api_image=${API_IMAGE}" -var="inference_url=${INFERENCE_URL}" -auto-approve
terraform -chdir="${REPO_ROOT}/terraform/api" output -raw api_url
