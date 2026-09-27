PYTHON ?= .venv/bin/python
INPUT ?= examples/briefs.jsonl
OUTPUT ?= data/processed/teacher-$(shell date +%Y%m%d-%H%M%S)
CACHE ?= data/cache/brief-teacher.sqlite3
PROJECT_ID ?=
REGION ?= asia-southeast1
IMAGE_DIGEST ?=

.PHONY: help dataset apply-platform build-inference deploy-inference submit-training-smoke verify-inference check-project check-image-digest

help:
	@echo "Usage: make <target> PROJECT_ID=<gcp-project-id> [REGION=asia-southeast1]"
	@echo
	@echo "Infrastructure targets:"
	@echo "  apply-platform         Create Terraform state and shared GCP resources"
	@echo "  build-inference        Build the pinned vLLM/Qwen image and print its digest"
	@echo "  deploy-inference       Deploy the private Cloud Run GPU service (requires IMAGE_DIGEST)"
	@echo "  submit-training-smoke  Verify Vertex AI can read the dataset and write artifacts"
	@echo "  verify-inference       Test private access, health, model listing, and generation"
	@echo
	@echo "Data targets:"
	@echo "  dataset           Generate dataset_v2 teacher drafts"
	@echo
	@echo "Examples:"
	@echo "  make apply-platform PROJECT_ID=campaign-generator-509812"
	@echo "  make build-inference PROJECT_ID=campaign-generator-509812"
	@echo "  make deploy-inference PROJECT_ID=campaign-generator-509812 IMAGE_DIGEST=asia-southeast1-docker.pkg.dev/...@sha256:..."
	@echo "  make submit-training-smoke PROJECT_ID=campaign-generator-509812"
	@echo "  make verify-inference PROJECT_ID=campaign-generator-509812"

dataset:
	@test -f "$(INPUT)" || { echo "Brief file not found: $(INPUT)" >&2; exit 1; }
	PYTHONPATH=src "$(PYTHON)" -m campaign_generator.dataset_v2.generate --input-file "$(INPUT)" --output-dir "$(OUTPUT)" --cache "$(CACHE)" $(if $(strip $(LIMIT)),--limit "$(LIMIT)",)

apply-platform: check-project
	./scripts/apply_platform.sh "$(PROJECT_ID)" "$(REGION)"

build-inference: check-project
	./scripts/build_inference.sh "$(PROJECT_ID)" "$(REGION)"

deploy-inference: check-project check-image-digest
	./scripts/deploy_inference.sh "$(PROJECT_ID)" "$(IMAGE_DIGEST)" "$(REGION)"

submit-training-smoke: check-project
	./scripts/submit_training_smoke.sh "$(PROJECT_ID)" "$(REGION)"

verify-inference: check-project
	./scripts/verify_inference.sh "$(PROJECT_ID)" "$(REGION)"

check-project:
	@test -n "$(PROJECT_ID)" || { echo "PROJECT_ID is required. Example: make apply-platform PROJECT_ID=campaign-generator-509812" >&2; exit 2; }

check-image-digest:
	@test -n "$(IMAGE_DIGEST)" || { echo "IMAGE_DIGEST is required. Use the digest printed by: make build-inference PROJECT_ID=<project-id>" >&2; exit 2; }
