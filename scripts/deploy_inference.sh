#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${1:?Usage: $0 PROJECT_ID IMAGE_DIGEST [REGION]}"
IMAGE_DIGEST="${2:?Usage: $0 PROJECT_ID IMAGE_DIGEST [REGION]}"
REGION="${3:-asia-southeast1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
STATE_BUCKET="${PROJECT_ID}-tfstate"

export GOOGLE_OAUTH_ACCESS_TOKEN
GOOGLE_OAUTH_ACCESS_TOKEN="$(gcloud auth print-access-token)"

if [[ ! "${IMAGE_DIGEST}" =~ @sha256:[0-9a-f]{64}$ ]]; then
  echo "IMAGE_DIGEST must be an immutable @sha256 reference" >&2
  exit 1
fi

printf 'bucket = "%s"\nprefix = "campaign-generator/serving"\n' "${STATE_BUCKET}" \
  > "${REPO_ROOT}/terraform/serving/backend.hcl"

terraform -chdir="${REPO_ROOT}/terraform/serving" init -reconfigure -backend-config=backend.hcl
terraform -chdir="${REPO_ROOT}/terraform/serving" validate
terraform -chdir="${REPO_ROOT}/terraform/serving" plan \
  -var="project_id=${PROJECT_ID}" \
  -var="region=${REGION}" \
  -var="inference_image=${IMAGE_DIGEST}"
terraform -chdir="${REPO_ROOT}/terraform/serving" apply \
  -var="project_id=${PROJECT_ID}" \
  -var="region=${REGION}" \
  -var="inference_image=${IMAGE_DIGEST}" \
  -auto-approve
