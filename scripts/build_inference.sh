#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${1:?Usage: $0 PROJECT_ID [REGION]}"
REGION="${2:-asia-southeast1}"
MODEL_REVISION="989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
IMAGE_TAG="baseline:${MODEL_REVISION:0:12}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/campaign-generator/${IMAGE_TAG}"
BUILD_SA="projects/${PROJECT_ID}/serviceAccounts/campaign-build@${PROJECT_ID}.iam.gserviceaccount.com"
BUILD_SOURCE_BUCKET="gs://${PROJECT_ID}-build-source/source"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

gcloud builds submit "${REPO_ROOT}/infra/inference" \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --service-account="${BUILD_SA}" \
  --config="${REPO_ROOT}/infra/inference/cloudbuild.yaml" \
  --gcs-source-staging-dir="${BUILD_SOURCE_BUCKET}" \
  --substitutions="_IMAGE=${IMAGE},_MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct,_MODEL_REVISION=${MODEL_REVISION}"

IMAGE_DIGEST="$(gcloud artifacts docker images describe "${IMAGE}" \
  --project="${PROJECT_ID}" \
  --format='value(image_summary.fully_qualified_digest)')"

if [[ -z "${IMAGE_DIGEST}" ]]; then
  echo "Unable to resolve the immutable image digest" >&2
  exit 1
fi

printf '%s\n' "${IMAGE_DIGEST}"
