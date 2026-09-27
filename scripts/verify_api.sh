#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${1:?Usage: $0 PROJECT_ID [REGION]}"
REGION="${2:-asia-southeast1}"
export GOOGLE_OAUTH_ACCESS_TOKEN
GOOGLE_OAUTH_ACCESS_TOKEN="$(gcloud auth print-access-token)"
API_URL="$(terraform -chdir="terraform/api" output -raw api_url)"
INFERENCE_URL="$(terraform -chdir="terraform/serving" output -raw inference_url)"

curl --fail --silent --show-error --max-time 30 "${API_URL}/health" >/dev/null
curl --fail --silent --show-error --max-time 30 "${API_URL}/" | grep --quiet 'Campaign concept demo'

ANONYMOUS_STATUS="$(curl --silent --output /dev/null --write-out '%{http_code}' \
  --max-time 30 "${INFERENCE_URL}/health")"
[[ "${ANONYMOUS_STATUS}" == "401" || "${ANONYMOUS_STATUS}" == "403" ]] || {
  echo "Expected anonymous private inference request to be denied, got ${ANONYMOUS_STATUS}" >&2
  exit 1
}

TMP_RESPONSE="$(mktemp)"
trap 'rm -f "${TMP_RESPONSE}"' EXIT
curl --fail --silent --show-error --max-time 900 \
  --header 'Content-Type: application/json' \
  --data '{"industry":"productivity software","brand":"Northstar Notes","product":"meeting notes app","target_audience":"team leads","objective":"encourage free-trial signups","channels":["LinkedIn","Email"],"proof_point":"The app turns meeting notes into assigned action items.","constraints":["Do not claim guaranteed time savings."]}' \
  "${API_URL}/generate" >"${TMP_RESPONSE}"
python3 - "${TMP_RESPONSE}" <<'PY'
import json, sys
data=json.load(open(sys.argv[1], encoding="utf-8"))
concepts=data.get("concepts")
assert isinstance(concepts, list) and len(concepts)==3
names=[item["campaign_direction"].casefold() for item in concepts]
assert len(set(names))==3
assert all(item.get("campaign_description") and item.get("proposed_kpis") for item in concepts)
assert all(item.get("budget_allocation")==[] for item in concepts)
print("api_schema=ok concepts=3 descriptions=ok private_inference=private")
PY
