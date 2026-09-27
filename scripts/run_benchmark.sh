#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${1:?Usage: $0 PROJECT_ID REGION DATASET_URI RUN_ID [REPORT_PATH] [JUDGE_MODEL]}"
REGION="${2:?Usage: $0 PROJECT_ID REGION DATASET_URI RUN_ID [REPORT_PATH] [JUDGE_MODEL]}"
DATASET_URI="${3:?Usage: $0 PROJECT_ID REGION DATASET_URI RUN_ID [REPORT_PATH] [JUDGE_MODEL]}"
RUN_ID="${4:?Usage: $0 PROJECT_ID REGION DATASET_URI RUN_ID [REPORT_PATH] [JUDGE_MODEL]}"
REPORT_PATH="${5:-experiments/runs/${RUN_ID}/report.json}"
JUDGE_MODEL="${6:-gpt-6-luna}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PREDICTION_URI="gs://${PROJECT_ID}-model-artifacts/runs/${RUN_ID}/evaluation"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT
gcloud storage cp "${DATASET_URI%/}/test.jsonl" "${TMP_DIR}/test.jsonl"
gcloud storage cp "${PREDICTION_URI}/base.jsonl" "${TMP_DIR}/base.jsonl"
gcloud storage cp "${PREDICTION_URI}/tuned.jsonl" "${TMP_DIR}/tuned.jsonl"
gcloud storage cp "gs://${PROJECT_ID}-model-artifacts/runs/${RUN_ID}/run_manifest.json" "${TMP_DIR}/training.json"
mkdir -p "$(dirname "${REPORT_PATH}")"
cp "${TMP_DIR}/training.json" "$(dirname "${REPORT_PATH}")/run_manifest.json"
cp "${TMP_DIR}/base.jsonl" "$(dirname "${REPORT_PATH}")/base.jsonl"
cp "${TMP_DIR}/tuned.jsonl" "$(dirname "${REPORT_PATH}")/tuned.jsonl"
PYTHONPATH="${REPO_ROOT}/src" "${REPO_ROOT}/.venv/bin/python" -m campaign_generator.evaluation.llm_judge \
  --test-file "${TMP_DIR}/test.jsonl" \
  --base-predictions "${TMP_DIR}/base.jsonl" \
  --tuned-predictions "${TMP_DIR}/tuned.jsonl" \
  --output "${REPORT_PATH}" --model "${JUDGE_MODEL}" \
  --expected-ideas 3 --mode concepts --limit 36 --execute
REPORT_MD="${REPORT_PATH%.*}.md"
"${REPO_ROOT}/.venv/bin/python" "${SCRIPT_DIR}/render_benchmark_report.py" \
  "${REPORT_PATH}" "${REPORT_MD}" "${TMP_DIR}/training.json" \
  "${TMP_DIR}/base.jsonl" "${TMP_DIR}/tuned.jsonl"
gcloud storage cp "${REPORT_PATH}" "${PREDICTION_URI}/report.json"
gcloud storage cp "${REPORT_MD}" "${PREDICTION_URI}/report.md"
printf 'Benchmark report: %s\nMarkdown report: %s\nGCS: %s/report.md and report.json\n' "${REPORT_PATH}" "${REPORT_MD}" "${PREDICTION_URI}"
