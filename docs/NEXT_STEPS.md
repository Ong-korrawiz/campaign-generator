# Next steps after infrastructure

## Public API

1. Implement FastAPI endpoints `GET /`, `GET /health`, and `POST /generate`.
2. Reuse the existing campaign request and response schemas.
3. Fetch a Google identity token for the private inference URL and call vLLM's OpenAI-compatible API.
4. Enforce request-size, token, timeout, and exactly-three-concepts constraints.
5. Add a CPU Cloud Run service to Terraform, use the existing `campaign-api` service account, set `max-instances=1`, and allow public invocation only on this API service.
6. Test public request → API validation → authenticated private inference → validated response.

## Fine-tuning and evaluation

1. Implement the LoRA trainer and package it as the training image consumed by the existing Vertex template.
2. Upload the approved dataset to the private dataset bucket.
3. Run the Spot L4 job with checkpoint resume and persist the adapter plus run manifest.
4. Compare base and fine-tuned outputs with the existing judge and quality gates.
5. Merge only a passing adapter, build an immutable serving image, and deploy it as a new Cloud Run revision.
