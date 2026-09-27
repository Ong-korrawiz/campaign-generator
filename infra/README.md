# Campaign Generator Infrastructure

The infrastructure is split into three Terraform states so the state bucket can be created before the remote backends and the inference image can be built before Cloud Run is deployed.

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

Run the CPU-only permissions smoke test with:

```bash
make submit-training-smoke PROJECT_ID="$PROJECT_ID" REGION="$REGION"
```

## Cleanup

Destroy serving before platform. The state bucket is intentionally retained and must be emptied and removed separately after both remote states are no longer needed.

```bash
terraform -chdir=terraform/serving destroy
terraform -chdir=terraform/platform destroy
terraform -chdir=terraform/bootstrap destroy
```
