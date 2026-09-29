# Infrastructure

Run these commands from the repository root. Before running `make deploy-demo`, you need an existing GCP project with billing enabled and Cloud Run L4 quota in the selected region. You also need Terraform 1.12+ and the Google Cloud CLI.

## 1. ADC

Sign in with an account that can manage the project's IAM, storage, Cloud Run, and Vertex resources. Terraform uses Application Default Credentials (ADC); the build scripts use the `gcloud` login.

```bash
PROJECT_ID=campaign-generator-509812
REGION=asia-southeast1
unset GOOGLE_APPLICATION_CREDENTIALS
gcloud auth login
gcloud auth application-default login
gcloud auth application-default print-access-token >/dev/null
```

If Terraform reports `invalid_grant`, rerun `gcloud auth application-default login`. See [Google's Terraform authentication guide](https://docs.cloud.google.com/docs/terraform/authentication).

The local API instructions in the main README create **impersonated** ADC credentials in the same default file. If you followed them, restore your own Google account's ADC before running any Terraform or `make deploy-demo`/`make destroy-*` command: `unset GOOGLE_APPLICATION_CREDENTIALS && gcloud auth application-default login`. The account you sign in with must have access to the project's Terraform state bucket and resources.

## 2. Deploy demo

```bash
make deploy-demo PROJECT_ID="$PROJECT_ID" REGION="$REGION"
```

Creates the Terraform state bucket and shared resources, builds and deploys private GPU inference and the public API, verifies both services, then prints the demo URL. Each operation reports `OK` or `FAILED`; the command stops on failure.

To get the public demo URL again later, read it from Cloud Run using the `gcloud` login (no Terraform state or local ADC impersonation needed):

```bash
gcloud run services describe campaign-api --project="$PROJECT_ID" --region="$REGION" --format='value(status.url)'
```

Open that URL in a browser. Visitors do not need GCP credentials.

## 3. Training

First upload the committed dataset with `make upload-dataset PROJECT_ID="$PROJECT_ID" DATASET_DIR=dataset`, or prepare a new one using the [dataset steps](../README.md#dataset). The upload prints `Dataset uploaded: gs://<project-id>-dataset/versions/<64-character-hash>`. Copy the **entire URI** after `Dataset uploaded:` into `DATASET_URI`. The last part is the SHA-256 hash of the chosen dataset's `manifest.json`; run `sha256sum dataset/manifest.json` for the committed version or `sha256sum data/processed/dataset-v3/manifest.json` for a new run.

Choose `RUN_ID` yourself for this training run; it names the output folder under `gs://$PROJECT_ID-model-artifacts/runs/`. Use a new value for every run, for example `lora-20260929-01`, so earlier results are not overwritten.

```bash
DATASET_URI="gs://$PROJECT_ID-dataset/versions/<manifest-hash-from-upload>"
RUN_ID="lora-$(date -u +%Y%m%dT%H%M%SZ)"
make train PROJECT_ID="$PROJECT_ID" REGION="$REGION" DATASET_URI="$DATASET_URI" RUN_ID="$RUN_ID"
```

The `date` command above creates a new `RUN_ID` from the current UTC time. `make train` checks the uploaded dataset, builds the training image, resolves its digest, snapshots current thumb-up feedback into an immutable dataset version, and submits a Vertex Spot L4 job. It returns after submission; training continues in Vertex. The public API stores feedback in a separate private versioned bucket created by `apply-platform`.

## 4. Destroy

If you used local service account impersonation, restore the deployment account's ADC as described in [Step 1](#1-adc) before destroying resources. An `iam.serviceAccounts.getAccessToken` 403 during `terraform init` means initialization stopped before any resource was deleted.

To remove each layer separately, run these commands in order:

```bash
make destroy-api PROJECT_ID="$PROJECT_ID" REGION="$REGION"
make destroy-inference PROJECT_ID="$PROJECT_ID" REGION="$REGION"
make destroy-platform PROJECT_ID="$PROJECT_ID" REGION="$REGION"
make destroy-state PROJECT_ID="$PROJECT_ID" REGION="$REGION"
```

`destroy-api` removes the public API; `destroy-inference` removes the private GPU service. `destroy-platform` removes shared resources and all dataset, feedback, and model artifacts. `destroy-state` removes the versioned Terraform state bucket.

To run the same teardown in one command:

```bash
make destroy-all PROJECT_ID="$PROJECT_ID" REGION="$REGION"
```

`destroy-all` runs the four destroy steps in dependency order and permanently deletes Terraform-managed resources, dataset/feedback/model bucket contents, and state history. The GCP project and enabled APIs remain. Stop any active Vertex CustomJobs separately. Each operation reports `OK` or `FAILED` and stops on failure.

Before running the destroy command, switch back to your personal Google account and refresh your local credentials:
```
PROJECT_ID=campaign-generator-509812
REGION=asia-southeast1
unset GOOGLE_APPLICATION_CREDENTIALS
gcloud auth login
gcloud auth application-default login
gcloud auth application-default print-access-token >/dev/null
```
