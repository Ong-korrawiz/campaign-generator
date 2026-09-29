PYTHON ?= .venv/bin/python
PROJECT_ID ?= campaign-generator-509812
REGION ?= asia-southeast1
IMAGE_DIGEST ?=
DATASET_DIR ?= data/processed/dataset-v3
REVIEWS ?= data/review/baseline-v3-reviews.jsonl
DATASET_URI ?=
TRAINING_IMAGE ?=
RUN_ID ?=
PREDICTION_MODE ?=
ADAPTER_URI ?=
REPORT_PATH ?=
JUDGE_MODEL ?= gpt-6-luna

.PHONY: help dataset-v3 dataset-v3-prepare dataset-v3-finalize dataset-v3-schema-only upload-dataset deploy-demo train destroy-api destroy-inference destroy-platform destroy-state destroy-all apply-platform build-inference build-api build-training deploy-inference deploy-api submit-training submit-predictions benchmark verify-inference verify-api check-project

help:
	@echo "Usage: make <target> PROJECT_ID=<gcp-project-id> [REGION=asia-southeast1]"
	@echo
	@echo "Infrastructure targets:"
	@echo "  deploy-demo            Run Steps 3-6 in order and report each step's status"
	@echo "  train                  Build training image and submit a Vertex job (requires DATASET_URI, RUN_ID)"
	@echo "  destroy-api           Destroy the public API"
	@echo "  destroy-inference     Destroy the private GPU inference service"
	@echo "  destroy-platform      Destroy shared resources and dataset/artifact buckets"
	@echo "  destroy-state         Destroy the Terraform state bucket"
	@echo "  destroy-all           Run all destroy targets in dependency order"
	@echo "  apply-platform         Create Terraform state and shared GCP resources"
	@echo "  build-inference        Build the pinned vLLM/Qwen image and print its digest"
	@echo "  deploy-inference       Deploy the private Cloud Run GPU service (auto-resolves IMAGE_DIGEST)"
	@echo "  verify-inference       Test private access, health, model listing, and generation"
	@echo "  build-api              Build the public API image and print its digest"
	@echo "  deploy-api             Deploy public API (auto-resolves IMAGE_DIGEST)"
	@echo "  verify-api             Verify public API and private inference access"
	@echo "  build-training         Build pinned LoRA training image"
	@echo "  submit-training        Submit dataset to Vertex Spot L4"
	@echo "  submit-predictions     Run baseline or tuned holdout inference on Vertex Spot L4"
	@echo "  benchmark              Judge matched baseline/tuned predictions and upload report"
	@echo
	@echo "Data targets:"
	@echo "  dataset-v3        Generate 240 description-aware review candidates"
	@echo "  dataset-v3-prepare  Write deterministic synthetic briefs without API calls"
	@echo "  dataset-v3-finalize  Assemble splits after human decisions"
	@echo "  dataset-v3-schema-only  Prepare training splits after schema validation only"
	@echo "  upload-dataset    Retry automatic v3 GCS upload after a network failure"
	@echo
	@echo "Examples:"
	@echo "  make deploy-demo PROJECT_ID=campaign-generator-509812"
	@echo "  make train PROJECT_ID=campaign-generator-509812 DATASET_URI=gs://.../versions/<hash> RUN_ID=<run-id>"
	@echo "  make destroy-all PROJECT_ID=campaign-generator-509812"
	@echo "  make apply-platform PROJECT_ID=campaign-generator-509812"
	@echo "  make build-inference PROJECT_ID=campaign-generator-509812"
	@echo "  make deploy-inference PROJECT_ID=campaign-generator-509812"
	@echo "  make verify-inference PROJECT_ID=campaign-generator-509812"
	@echo "  make build-api PROJECT_ID=campaign-generator-509812"
	@echo "  make deploy-api PROJECT_ID=campaign-generator-509812"

dataset-v3:
	PYTHONPATH=src "$(PYTHON)" -m campaign_generator.dataset_v3.pipeline generate --output-dir "$(DATASET_DIR)" --project-id "$(PROJECT_ID)"

dataset-v3-prepare:
	PYTHONPATH=src "$(PYTHON)" -m campaign_generator.dataset_v3.pipeline prepare-briefs --output-dir "$(DATASET_DIR)"

dataset-v3-finalize:
	PYTHONPATH=src "$(PYTHON)" -m campaign_generator.dataset_v3.pipeline finalize --output-dir "$(DATASET_DIR)" --reviews "$(REVIEWS)" --project-id "$(PROJECT_ID)"

dataset-v3-schema-only:
	PYTHONPATH=src "$(PYTHON)" -m campaign_generator.dataset_v3.pipeline prepare-schema-only-training --output-dir "$(DATASET_DIR)" --project-id "$(PROJECT_ID)"

upload-dataset: check-project
	PYTHONPATH=src "$(PYTHON)" -m campaign_generator.dataset_v3.pipeline sync --output-dir "$(DATASET_DIR)" --project-id "$(PROJECT_ID)"

apply-platform: check-project
	./scripts/apply_platform.sh "$(PROJECT_ID)" "$(REGION)"

deploy-demo: check-project
	bash ./scripts/deploy_demo.sh "$(PROJECT_ID)" "$(REGION)"

train: check-project
	@test -n "$(DATASET_URI)" || { echo "DATASET_URI is required" >&2; exit 2; }
	@test -n "$(RUN_ID)" || { echo "RUN_ID is required" >&2; exit 2; }
	bash ./scripts/train.sh "$(PROJECT_ID)" "$(REGION)" "$(DATASET_URI)" "$(RUN_ID)"

destroy-api destroy-inference destroy-platform destroy-state destroy-all: check-project
	bash ./scripts/destroy_infra.sh "$(patsubst destroy-%,%,$@)" "$(PROJECT_ID)" "$(REGION)"

build-inference: check-project
	./scripts/build_inference.sh "$(PROJECT_ID)" "$(REGION)"

deploy-inference: check-project
	./scripts/deploy_inference.sh "$(PROJECT_ID)" "$(IMAGE_DIGEST)" "$(REGION)"

build-api: check-project
	./scripts/build_api.sh "$(PROJECT_ID)" "$(REGION)"

deploy-api: check-project
	./scripts/deploy_api.sh "$(PROJECT_ID)" "$(IMAGE_DIGEST)" "$(REGION)"

verify-api: check-project
	./scripts/verify_api.sh "$(PROJECT_ID)" "$(REGION)"

build-training: check-project
	./scripts/build_training.sh "$(PROJECT_ID)" "$(REGION)"

submit-training: check-project
	@test -n "$(DATASET_URI)" || { echo "DATASET_URI is required" >&2; exit 2; }
	@test -n "$(TRAINING_IMAGE)" || { echo "TRAINING_IMAGE is required" >&2; exit 2; }
	@test -n "$(RUN_ID)" || { echo "RUN_ID is required" >&2; exit 2; }
	./scripts/submit_training.sh "$(PROJECT_ID)" "$(REGION)" "$(DATASET_URI)" "$(TRAINING_IMAGE)" "$(RUN_ID)"

submit-predictions: check-project
	@test -n "$(DATASET_URI)" || { echo "DATASET_URI is required" >&2; exit 2; }
	@test -n "$(TRAINING_IMAGE)" || { echo "TRAINING_IMAGE is required" >&2; exit 2; }
	@test -n "$(RUN_ID)" || { echo "RUN_ID is required" >&2; exit 2; }
	@test -n "$(PREDICTION_MODE)" || { echo "PREDICTION_MODE=base or tuned is required" >&2; exit 2; }
	./scripts/submit_predictions.sh "$(PROJECT_ID)" "$(REGION)" "$(DATASET_URI)" "$(TRAINING_IMAGE)" "$(RUN_ID)" "$(PREDICTION_MODE)" "$(ADAPTER_URI)"

benchmark: check-project
	@test -n "$(DATASET_URI)" || { echo "DATASET_URI is required" >&2; exit 2; }
	@test -n "$(RUN_ID)" || { echo "RUN_ID is required" >&2; exit 2; }
	./scripts/run_benchmark.sh "$(PROJECT_ID)" "$(REGION)" "$(DATASET_URI)" "$(RUN_ID)" "$(REPORT_PATH)" "$(JUDGE_MODEL)"

verify-inference: check-project
	./scripts/verify_inference.sh "$(PROJECT_ID)" "$(REGION)"

check-project:
	@test -n "$(PROJECT_ID)" || { echo "PROJECT_ID is required. Example: make apply-platform PROJECT_ID=campaign-generator-509812" >&2; exit 2; }
