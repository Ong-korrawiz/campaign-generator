#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${1:?Usage: $0 PROJECT_ID [REGION]}"
REGION="${2:-asia-southeast1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

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

run_make() {
  local target="$1"
  make --no-print-directory -C "${REPO_ROOT}" "${target}" \
    "PROJECT_ID=${PROJECT_ID}" "REGION=${REGION}" "IMAGE_DIGEST="
}

show_demo_url() {
  local api_url
  api_url="$(terraform -chdir="${REPO_ROOT}/terraform/api" output -raw api_url)"
  [[ -n "${api_url}" ]] || { echo "API URL is missing" >&2; return 1; }
  echo "Demo URL: ${api_url}"
}

run_step "Step 3: apply shared platform" run_make apply-platform
run_step "Step 4a: build inference image" run_make build-inference
run_step "Step 4b: deploy private inference" run_make deploy-inference
run_step "Step 5: verify private inference" run_make verify-inference
run_step "Step 6a: build API image" run_make build-api
run_step "Step 6b: deploy public API" run_make deploy-api
run_step "Step 6c: verify public API" run_make verify-api
run_step "Step 6d: show demo URL" show_demo_url
