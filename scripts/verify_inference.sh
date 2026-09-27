#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${1:?Usage: $0 PROJECT_ID [REGION]}"
REGION="${2:-asia-southeast1}"
SERVICE_NAME="campaign-baseline-inference"

URL="$(gcloud run services describe "${SERVICE_NAME}" \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --format='value(status.url)')"

ANONYMOUS_STATUS="$(curl --silent --output /dev/null --write-out '%{http_code}' \
  --max-time 30 "${URL}/health")"

if [[ "${ANONYMOUS_STATUS}" != "401" && "${ANONYMOUS_STATUS}" != "403" ]]; then
  echo "Expected anonymous request to be denied, got HTTP ${ANONYMOUS_STATUS}" >&2
  exit 1
fi

TOKEN="$(gcloud auth print-identity-token)"

curl --fail --silent --show-error --max-time 600 \
  --header "Authorization: Bearer ${TOKEN}" \
  "${URL}/health" >/dev/null

curl --fail --silent --show-error --max-time 600 \
  --header "Authorization: Bearer ${TOKEN}" \
  "${URL}/v1/models" | grep --quiet '"campaign-baseline"'

RESPONSE="$(curl --fail --silent --show-error --max-time 600 \
  --header "Authorization: Bearer ${TOKEN}" \
  --header 'Content-Type: application/json' \
  --data '{"model":"campaign-baseline","messages":[{"role":"user","content":"Return only the word OK."}],"temperature":0,"max_tokens":8}' \
  "${URL}/v1/chat/completions")"

grep --quiet '"choices"' <<<"${RESPONSE}"
printf 'anonymous=%s authenticated_health=ok models=ok generation=ok\n' "${ANONYMOUS_STATUS}"
