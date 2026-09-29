# Campaign Generator

Marketing campaign ideation prototype for the Jenosize AI & Data Engineer assignment. The public app takes an industry, audience, objective, channels, product proof, constraints, and an optional budget. It returns three English concepts with campaign descriptions, key messages, channel and asset plans, and proposed KPIs.

**Prototype status:** the Cloud Run API and GPU inference services have been decommissioned. See [code review guide](docs/code-review-guide.md) and the deployment instructions below to review or redeploy the project.

At deployment, the API used a private Qwen2.5-1.5B-Instruct baseline through vLLM. The LoRA adapter is experimental and was not deployed: on the 36-brief holdout, the original baseline produced three schema-valid concepts for 7/36 briefs and the tuned model for 0/36. The revised baseline API passed structural checks on 20/20 additional briefs, but manual review still found generic ideas and unsupported claims. These are separate evaluations; see [experiment results](experiments/README.md).

## Quickstart: local demo

Requires Python 3.10+, [uv](https://docs.astral.sh/uv/), the Google Cloud CLI, and access to the project's private inference service. Generating teacher drafts also requires an OpenAI API key; set `OPENAI_API_KEY` in `.env`. The hosted API and inference services have been decommissioned, so campaign generation requires redeploying the infrastructure first; follow the [IaC setup guide](infra/README.md).

For the simplest demo after deployment, open the public API URL printed by `make deploy-demo`. If you missed it, run `gcloud run services describe campaign-api --project="$PROJECT_ID" --region="$REGION" --format='value(status.url)'`. The public page does not require local ADC setup. The steps below are for running the API locally.

### 1. Clone the repo and install dependencies

Create your local environment file:

```bash
git clone https://github.com/Ong-korrawiz/campaign-generator.git
cd campaign-generator
uv sync --locked --group dev
cp .env.example .env
```

The `dev` dependency group includes `pre-commit`, which the Git hook runs from `.venv`. On a fresh clone, install the hook with `uv run pre-commit install`.

### 2. Get the private inference URL

After deploying the infrastructure, set your project and region and get the URL from Cloud Run:

```bash
PROJECT_ID="your-project-id"
REGION="asia-southeast1"
gcloud run services describe campaign-baseline-inference --project="$PROJECT_ID" --region="$REGION" --format='value(status.url)'
```

Put that URL in `.env` as `INFERENCE_URL` and `INFERENCE_AUDIENCE` (spelled **INFERENCE**, not `INVERENCE`; the audience defaults to the URL if omitted).

### 3. Allow local authentication

Local authentication needs permission to impersonate the `campaign-api` service account. Have a project administrator grant your Google account the Service Account Token Creator role **on that service account**:

```bash
USER_EMAIL="your-google-account@example.com"
gcloud iam service-accounts add-iam-policy-binding \
  "campaign-api@${PROJECT_ID}.iam.gserviceaccount.com" \
  --project="${PROJECT_ID}" \
  --member="user:${USER_EMAIL}" \
  --role="roles/iam.serviceAccountTokenCreator"
```

### 4. Sign in with ADC

Use the same Google account for ADC login. Set the credentials path **after** login; this application's ID token library reads that path directly:

```bash
unset GOOGLE_APPLICATION_CREDENTIALS
gcloud auth application-default login \
  --impersonate-service-account="campaign-api@${PROJECT_ID}.iam.gserviceaccount.com"
export GOOGLE_APPLICATION_CREDENTIALS="$(gcloud info --format='value(config.paths.global_config_dir)')/application_default_credentials.json"
```

### 5. Start the local API

Login only saves credentials; it does not prove the ID token permission works. Load `.env`, check token creation without printing the token, then start the API:

```bash
set -a
source .env
set +a
uv run python -c 'import os; from google.auth.transport.requests import Request; from google.oauth2 import id_token; id_token.fetch_id_token(Request(), os.environ["INFERENCE_AUDIENCE"]); print("ID token OK")'
uv run uvicorn campaign_generator.api.app:app --port 8080
```

If the token check returns `iam.serviceAccounts.getOpenIdToken` 403, confirm that `USER_EMAIL` is the account used at ADC login and that its role binding is on the `campaign-api` service account. See [Google's ADC impersonation guide](https://docs.cloud.google.com/docs/authentication/set-up-adc-local-dev-environment).

This local setup replaces your default ADC file with impersonated credentials. Before later Terraform, deployment, or destroy commands, run `unset GOOGLE_APPLICATION_CREDENTIALS && gcloud auth application-default login` to restore your own account's ADC.

### 6. Use the demo

Open [http://localhost:8080](http://localhost:8080) and submit a brief. `pyproject.toml` declares runtime, development, and training dependencies; the committed `uv.lock` pins their resolved versions. To add the training stack locally, run `uv sync --locked --extra training`. When dependency declarations change, run `uv lock` and commit both files.

## Dataset

[`dataset/`](dataset/) is the single committed fine-tuning dataset: 240 records split 168/24/48, with a manifest, quality report, and 120 source briefs needed to regenerate v3. The training labels were generated by `gpt-6-luna` and passed schema validation; human semantic review is still pending. Its former immutable GCS copy was used for the documented run; cloud resources have since been removed. The version hash can be recalculated from the committed manifest.

The v3 generator automatically uploads drafts to GCS staging. After schema validation or human review, it uploads the split files to an immutable `versions/<manifest-hash>` URI and removes staging. A failed upload leaves local files intact; `make upload-dataset` retries. Use `--no-upload` only for offline development.

```bash
make dataset-v3 PROJECT_ID=campaign-generator-509812
make dataset-v3-schema-only PROJECT_ID=campaign-generator-509812
# Or review all rows, then: make dataset-v3-finalize REVIEWS=path/to/reviews.jsonl PROJECT_ID=...
make upload-dataset PROJECT_ID=campaign-generator-509812
```

New runs write to ignored `data/processed/dataset-v3`; the committed dataset is not overwritten. Each upload verifies hashes and row counts. Source and teacher provenance are retained in the manifest and records.

The upload prints `Dataset uploaded: gs://<project-id>-dataset/versions/<hash>`. Use that whole URI as `DATASET_URI` for training. The `<hash>` is the SHA-256 of `data/processed/dataset-v3/manifest.json` (the first value from `sha256sum data/processed/dataset-v3/manifest.json`), calculated after the final splits are prepared. Choose `RUN_ID` yourself for each training run, such as `lora-20260929-01`; it names the output folder under `gs://<project-id>-model-artifacts/runs/` and must be new for every run.

## Fine-tuning and evaluation

The training job uses completion-only LoRA loss on assistant messages. Build a digest-pinned training image and submit a Vertex CustomJob using the immutable dataset URI:

```bash
make build-training PROJECT_ID=campaign-generator-509812
make submit-training PROJECT_ID=campaign-generator-509812 DATASET_URI=gs://campaign-generator-509812-dataset/versions/<hash> TRAINING_IMAGE=<image@sha256:...> RUN_ID=<unique-run-id>
make submit-predictions PROJECT_ID=campaign-generator-509812 DATASET_URI=gs://campaign-generator-509812-dataset/versions/<hash> TRAINING_IMAGE=<image@sha256:...> RUN_ID=<unique-run-id> PREDICTION_MODE=base
make submit-predictions PROJECT_ID=campaign-generator-509812 DATASET_URI=gs://campaign-generator-509812-dataset/versions/<hash> TRAINING_IMAGE=<image@sha256:...> RUN_ID=<unique-run-id> PREDICTION_MODE=tuned ADAPTER_URI=<adapter-gcs-uri>
make benchmark PROJECT_ID=campaign-generator-509812 DATASET_URI=gs://campaign-generator-509812-dataset/versions/<hash> RUN_ID=<unique-run-id>
```

Results for the completed v3 training run, including the training manifest, baseline/tuned outputs, benchmark, and diagnostics, are in [`experiments/runs/schema-only-seed42-20260927/`](experiments/runs/schema-only-seed42-20260927/). The adapter and checkpoint were archived locally under ignored `doc/cloud-archive/` before their GCS bucket was deleted; job IDs and original URIs are indexed in [`experiments/README.md`](experiments/README.md).

## API and deployment

`GET /` serves the demo, `GET /health` reports API health, and `POST /generate` accepts a JSON brief. On success the response contains exactly three `concepts`. Each concept has `campaign_direction`, `campaign_description`, `audience_insight`, `key_message`, `channel_plan`, `asset_plan`, `proposed_kpis`, and optional `budget_allocation`. KPI and budget numbers are planning proposals, not measured forecasts.

```bash
curl --max-time 900 -sS http://localhost:8080/generate \
  -H 'Content-Type: application/json' \
  -d '{"industry":"productivity software","brand":"Northstar Notes","product":"meeting notes app","target_audience":"team leads","objective":"increase trial sign-ups","channels":["LinkedIn","Email"],"proof_point":"The app turns meeting notes into assigned action items."}'
```

The public Cloud Run service authenticates to a private GPU inference service with a Google ID token. Terraform, build, and deployment instructions are in [`infra/README.md`](infra/README.md). The model limitations are recorded in [`docs/model-card.md`](docs/model-card.md).
