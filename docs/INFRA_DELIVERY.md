# Infrastructure delivery result

## Deployment

- Project: `campaign-generator-509812`
- Region: `asia-southeast1`
- Private inference service: `campaign-baseline-inference`
- Service URL: `https://campaign-baseline-inference-ivrph5gd4a-as.a.run.app`
- Image: `asia-southeast1-docker.pkg.dev/campaign-generator-509812/campaign-generator/baseline@sha256:418132d0e77e7f48216d65872d6ff6ff7f8c83e6445ed33fe1c296552bdf914c`
- Scaling: minimum 0, maximum 1 L4 instance
- Public access: disabled; `campaign-api` is the only dedicated runtime invoker

## Verification

- Terraform bootstrap, platform, and serving configurations validate successfully.
- A repeat serving plan/apply reported no drift.
- Anonymous inference returned HTTP 403.
- Authenticated `/health`, `/v1/models`, and chat completion succeeded.
- Scale-from-zero was observed in Cloud Run logs with reason `AUTOSCALING`; the measured cold verification took 2m34s.
- Vertex CPU smoke job `366435064822628352` completed with `JOB_STATE_SUCCEEDED`.
- The training identity read the dataset input and wrote `gs://campaign-generator-509812-model-artifacts/runs/smoke-20260926-155202/smoke-result.txt`.
- Existing Python tests pass: 5 passed.

## Ongoing cost

No training worker remains after the smoke job. Cloud Run can scale to zero, so ongoing charges are limited primarily to stored container layers, GCS objects, and brief compute while requests are active.
