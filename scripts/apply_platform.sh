#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${1:?Usage: $0 PROJECT_ID [REGION]}"
REGION="${2:-asia-southeast1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
STATE_BUCKET="${PROJECT_ID}-tfstate"

export GOOGLE_OAUTH_ACCESS_TOKEN
GOOGLE_OAUTH_ACCESS_TOKEN="$(gcloud auth print-access-token)"

gcloud projects describe "${PROJECT_ID}" --project="${PROJECT_ID}" >/dev/null
if [[ "$(gcloud beta billing projects describe "${PROJECT_ID}" --format='value(billingEnabled)')" != "True" ]]; then
  echo "Billing is not enabled for ${PROJECT_ID}" >&2
  exit 1
fi

terraform -chdir="${REPO_ROOT}/terraform/bootstrap" init
terraform -chdir="${REPO_ROOT}/terraform/bootstrap" apply \
  -var="project_id=${PROJECT_ID}" \
  -var="region=${REGION}" \
  -auto-approve

printf 'bucket = "%s"\nprefix = "campaign-generator/platform"\n' "${STATE_BUCKET}" \
  > "${REPO_ROOT}/terraform/platform/backend.hcl"

terraform -chdir="${REPO_ROOT}/terraform/platform" init -reconfigure -backend-config=backend.hcl
terraform -chdir="${REPO_ROOT}/terraform/platform" validate
terraform -chdir="${REPO_ROOT}/terraform/platform" plan \
  -var="project_id=${PROJECT_ID}" \
  -var="region=${REGION}"
terraform -chdir="${REPO_ROOT}/terraform/platform" apply \
  -var="project_id=${PROJECT_ID}" \
  -var="region=${REGION}" \
  -auto-approve
