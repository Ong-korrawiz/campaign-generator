# Model card: Campaign Generator baseline

## Deployment

- Public API: redeployed on 2026-09-30 at https://campaign-api-ivrph5gd4a-as.a.run.app.
- Inference model: Qwen2.5-1.5B-Instruct, revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- Serving: private Cloud Run L4 behind a public API that authenticates to inference with a Google ID token.
- Fine-tuned adapter: exploratory LoRA artifact from Vertex CustomJob `4823415399175421952`; not promoted after failing the output-format gate.

## Intended use

Generate three structured English campaign concept proposals from a supplied
brief. Each concept includes a direction, a 2–3 sentence campaign description,
an audience insight, key message, channel and asset plans, and proposed KPI
guidance. Budget allocation is included only when the caller supplies a budget
range.

## Limitations

Outputs can be generic or inaccurate even when they pass schema validation.
The revised public API passed structural checks on 20/20 holdout briefs, but a
five-brief manual spot check found unsupported claims and generic ideas. KPI and
budget figures are planning proposals, not forecasts or guarantees. A human
should check factual claims, brand fit, legal constraints, and execution
details before publication.

## Fine-tuning data status

Dataset `baseline-v3-description-seed42` contains 240 teacher-generated labels
with train/validation/test splits of 168/24/48. Every row passed the current
CampaignDirectionV2 and TrainingRecordV2 schemas. The manifest explicitly has
`human_semantic_review: false`; semantic grounding and label quality were not
reviewed. The dataset is for exploratory training only and is not eligible for
promotion until it passes the planned human review and evaluation gates.

Baseline and tuned inference were benchmarked on 36 synthetic holdout briefs.
The adapter produced zero schema-valid three-concept API outputs in the original
36-brief benchmark. With the exact training prompt and stochastic sampling, it
produced three schema-valid candidate concepts on 6/6 validation briefs, but
only 3/6 met the added description-length and near-duplicate-name rules. Keep
the baseline deployed; these results remain exploratory while semantic review
is pending.
