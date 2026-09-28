# Campaign Generator Infrastructure

The infrastructure is split into bootstrap, platform, private serving, and public API Terraform states so each dependency can be created before the next service is deployed.

## Target

- Project: supplied explicitly to every command
- Default region: `asia-southeast1`
- Baseline model: `Qwen/Qwen2.5-1.5B-Instruct`
- Model revision: `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`

## Prerequisites

Terraform 1.12+, Google Cloud CLI, an authenticated account with permission to enable services and create IAM/storage/Cloud Run/Vertex resources, an existing project with billing enabled, and Cloud Run L4 quota.

## Deploy

Each script in `scripts/` has a matching Make target. Run `make help` to see
the commands and required variables.

```bash
PROJECT_ID=campaign-generator-509812
REGION=asia-southeast1

# Bootstrap the state bucket first when setting up from a clean project.
terraform -chdir=terraform/bootstrap init
terraform -chdir=terraform/bootstrap apply -var="project_id=$PROJECT_ID" -var="region=$REGION"

# Set terraform/platform/backend.hcl to the state bucket and prefix before init.
terraform -chdir=terraform/platform init -backend-config=backend.hcl
make apply-platform PROJECT_ID="$PROJECT_ID" REGION="$REGION"
make build-inference PROJECT_ID="$PROJECT_ID" REGION="$REGION"
make deploy-inference PROJECT_ID="$PROJECT_ID" REGION="$REGION" IMAGE_DIGEST="<digest printed by build-inference>"
```

The Make targets are thin wrappers; the implementation remains in the
same-named shell script under `scripts/`.

The inference service is private. No `allUsers` IAM binding is created. Test as an authorized operator:

```bash
URL="$(terraform -chdir=terraform/serving output -raw inference_url)"
TOKEN="$(gcloud auth print-identity-token)"
curl -H "Authorization: Bearer $TOKEN" "$URL/health"
curl -H "Authorization: Bearer $TOKEN" "$URL/v1/models"
```

Run the complete smoke check with:

```bash
make verify-inference PROJECT_ID="$PROJECT_ID" REGION="$REGION"
```

An anonymous request must return HTTP 401 or 403.

## Training plumbing

Render `infra/training/custom-job.yaml.tpl` with `TRAINING_SERVICE_ACCOUNT`, `TRAINING_IMAGE_URI`, `DATASET_URI`, `OUTPUT_URI`, and `RUN_ID`, then submit it with:

```bash
gcloud ai custom-jobs create \
  --project="$PROJECT_ID" \
  --region="$REGION" \
  --display-name="campaign-lora-$RUN_ID" \
  --config=custom-job.yaml
```

The template uses one Spot L4. Training code must checkpoint to `OUTPUT_URI` because Spot workers can be preempted.

Build the pinned trainer image with `make build-training`. The v3 generator
uploads drafts to GCS staging automatically. `make dataset-v3-schema-only`
validates the splits and uploads an immutable training version; this keeps
semantic review pending and is not promotion-ready. `make upload-dataset`
retries a failed upload. The training job
reads only train/validation; test remains reserved for final evaluation.

## Public API

After private inference is deployed, build and deploy the public CPU service:

```bash
make build-api PROJECT_ID="$PROJECT_ID" REGION="$REGION"
make deploy-api PROJECT_ID="$PROJECT_ID" REGION="$REGION" IMAGE_DIGEST="<api-image-digest>"
make verify-api PROJECT_ID="$PROJECT_ID" REGION="$REGION"
```

Terraform creates a separate `campaign-generator/api` state and exposes only
the API. The serving stack grants `campaign-api` `roles/run.invoker` on the
private inference service. The API uses the inference service base URL for
both the HTTP endpoint and Google identity-token audience.

## Cleanup

For a future teardown, destroy API before serving, then platform, and bootstrap last. Empty the versioned state bucket only after all remote states have been saved, because it contains the state needed for Terraform cleanup.

```bash
terraform -chdir=terraform/api destroy
terraform -chdir=terraform/serving destroy
terraform -chdir=terraform/platform destroy
terraform -chdir=terraform/bootstrap destroy
```
