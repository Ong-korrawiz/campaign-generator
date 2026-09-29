#!/usr/bin/env bash
set -euo pipefail

SCOPE="${1:?Usage: $0 api|inference|platform|state|all PROJECT_ID [REGION]}"
PROJECT_ID="${2:?Usage: $0 api|inference|platform|state|all PROJECT_ID [REGION]}"
REGION="${3:-asia-southeast1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
STATE_BUCKET="${PROJECT_ID}-tfstate"
# Terraform validates these required image variables during destroy; no image is deployed.
EMPTY_DIGEST="$(printf '%064d' 0)"

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

init_remote_state() {
  local component="$1"
  local terraform_dir="${REPO_ROOT}/terraform/${component}"
  printf 'bucket = "%s"\nprefix = "campaign-generator/%s"\n' \
    "${STATE_BUCKET}" "${component}" > "${terraform_dir}/backend.hcl"
  terraform -chdir="${terraform_dir}" init -reconfigure -backend-config=backend.hcl
}

ensure_empty_state() {
  local component="$1"
  local resources
  init_remote_state "${component}" || return
  resources="$(terraform -chdir="${REPO_ROOT}/terraform/${component}" state list)" || return
  [[ -z "${resources}" ]] || {
    echo "${component} still has Terraform resources; destroy it first:" >&2
    echo "${resources}" >&2
    return 1
  }
}

read_state_resources() {
  local component="$1"
  RESOURCES="$(terraform -chdir="${REPO_ROOT}/terraform/${component}" state list)"
}

destroy_api() {
  run_step "initialize API state" init_remote_state api
  run_step "destroy public API" terraform -chdir="${REPO_ROOT}/terraform/api" destroy \
    -var="project_id=${PROJECT_ID}" -var="region=${REGION}" \
    -var="api_image=example.invalid/api@sha256:${EMPTY_DIGEST}" \
    -var="inference_url=https://example.invalid" -auto-approve
}

destroy_inference() {
  run_step "initialize inference state" init_remote_state serving
  run_step "destroy private inference" terraform -chdir="${REPO_ROOT}/terraform/serving" destroy \
    -var="project_id=${PROJECT_ID}" -var="region=${REGION}" \
    -var="inference_image=example.invalid/inference@sha256:${EMPTY_DIGEST}" -auto-approve
}

destroy_platform() {
  run_step "confirm API state is empty" ensure_empty_state api
  run_step "confirm inference state is empty" ensure_empty_state serving
  run_step "initialize platform state" init_remote_state platform
  run_step "inspect platform state" read_state_resources platform
  local targets=()
  if grep -Fxq 'google_storage_bucket.dataset' <<<"${RESOURCES}"; then
    targets+=(-target=google_storage_bucket.dataset)
  fi
  if grep -Fxq 'google_storage_bucket.model_artifacts' <<<"${RESOURCES}"; then
    targets+=(-target=google_storage_bucket.model_artifacts)
  fi
  if grep -Fxq 'google_storage_bucket.feedback' <<<"${RESOURCES}"; then
    targets+=(-target=google_storage_bucket.feedback)
  fi
  if ((${#targets[@]})); then
    run_step "allow data bucket cleanup" terraform \
      -chdir="${REPO_ROOT}/terraform/platform" apply "${targets[@]}" \
      -var="project_id=${PROJECT_ID}" -var="region=${REGION}" \
      -var="force_destroy_data_buckets=true" -auto-approve
  fi
  run_step "destroy shared platform and data buckets" terraform \
    -chdir="${REPO_ROOT}/terraform/platform" destroy \
    -var="project_id=${PROJECT_ID}" -var="region=${REGION}" \
    -var="force_destroy_data_buckets=true" -auto-approve
}

destroy_state() {
  run_step "confirm API state is empty" ensure_empty_state api
  run_step "confirm inference state is empty" ensure_empty_state serving
  run_step "confirm platform state is empty" ensure_empty_state platform
  run_step "initialize bootstrap state" terraform \
    -chdir="${REPO_ROOT}/terraform/bootstrap" init
  run_step "inspect bootstrap state" read_state_resources bootstrap
  if grep -Fxq 'google_storage_bucket.terraform_state' <<<"${RESOURCES}"; then
    run_step "allow Terraform state bucket cleanup" terraform \
      -chdir="${REPO_ROOT}/terraform/bootstrap" apply \
      -target=google_storage_bucket.terraform_state \
      -var="project_id=${PROJECT_ID}" -var="region=${REGION}" \
      -var="force_destroy_state_bucket=true" -auto-approve
  fi
  run_step "destroy Terraform state bucket" terraform \
    -chdir="${REPO_ROOT}/terraform/bootstrap" destroy \
    -var="project_id=${PROJECT_ID}" -var="region=${REGION}" \
    -var="force_destroy_state_bucket=true" -auto-approve
}

case "${SCOPE}" in
  api) destroy_api ;;
  inference) destroy_inference ;;
  platform) destroy_platform ;;
  state) destroy_state ;;
  all)
    destroy_api
    destroy_inference
    destroy_platform
    destroy_state
    ;;
  *) echo "Unknown destroy scope: ${SCOPE}" >&2; exit 2 ;;
esac
