#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${1:?Usage: $0 PROJECT_ID [REGION]}"
REGION="${2:-asia-southeast1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
IMAGE_TAG="training:$(git -C "${REPO_ROOT}" rev-parse --short=12 HEAD)"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/campaign-generator/${IMAGE_TAG}"
BUILD_SA="projects/${PROJECT_ID}/serviceAccounts/campaign-build@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud builds submit "${REPO_ROOT}" --project="${PROJECT_ID}" --region="${REGION}" \
  --service-account="${BUILD_SA}" --gcs-source-staging-dir="gs://${PROJECT_ID}-build-source/source" \
  --config="${REPO_ROOT}/infra/training/cloudbuild.yaml" --substitutions="_IMAGE=${IMAGE}"
gcloud artifacts docker images describe "${IMAGE}" --project="${PROJECT_ID}" \
  --format='value(image_summary.fully_qualified_digest)'
