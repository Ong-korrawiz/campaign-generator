# Baseline vs LoRA benchmark

- Run: `schema-only-seed42-20260927`
- Dataset: `gs://campaign-generator-509812-dataset/versions/2368fab94e9eb94deb91cdd79526a3dc8107240e5c97ee424ad35317aec8620d`; 36 matched synthetic holdout briefs
- Base model: `Qwen/Qwen2.5-1.5B-Instruct` revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- Judge: `gpt-6-luna`; 72 paired-order calls
- Dataset semantic review: pending; this is an exploratory schema-only experiment.

## Pairwise result

- Overall: base 7 wins, tuned 4 wins, 15 ties (26 resolved briefs).
- Tuned net win: **-11.5 pp**; 95% bootstrap CI **[-34.6, 11.5] pp**.
- Order disagreement: 27.8%.

| Criterion | Base wins | Tuned wins | Ties | Tuned net |
|---|---:|---:|---:|---:|
| goal alignment | 7 | 0 | 29 | -19.4 pp |
| groundedness | 3 | 14 | 13 | +30.6 pp |
| idea distinctness | 1 | 0 | 26 | -2.8 pp |
| execution fit | 7 | 0 | 26 | -19.4 pp |
| overall | 7 | 4 | 15 | -8.3 pp |

## Output quality and runtime

| Check | Baseline | LoRA |
|---|---:|---:|
| Full format pass | 11.1% | 0.0% |
| Description present | 19.4% | 0.0% |
| Description has 2–3 sentences | 11.1% | 0.0% |
| Descriptions distinct (exact check) | 19.4% | 0.0% |
| Proposed KPIs present | 19.4% | 0.0% |
| Validation failure: `invalid_model_concept_after_retry:channel_plan.0.role:missing,channel_plan.1.role:missing,key_message:missing` | 0/36 | 1/36 |
| Validation failure: `invalid_model_concept_after_retry:channel_plan:missing` | 0/36 | 29/36 |
| Validation failure: `invalid_model_concept_after_retry:duplicate_campaign_direction` | 13/36 | 0/36 |
| Validation failure: `invalid_model_concept_after_retry:key_message:missing` | 0/36 | 1/36 |
| Validation failure: `invalid_model_concept_after_retry:model_used_unapproved_channel` | 16/36 | 0/36 |
| Validation failure: `invalid_model_concept_after_retry:response:invalid_json` | 0/36 | 5/36 |
- Base inference: mean 53.8s; p95 81.7s; mean output 1263 tokens per brief.
- Tuned inference: mean 44.1s; p95 58.2s; mean output 497 tokens per brief.
- Training: 3 epochs; LoRA r=16, alpha=32; train loss 1.524; validation loss 1.279.

## Promotion gate

- Net win ≥10 pp: **False**
- 95% CI excludes zero: **False**
- Format pass ≥98%: **False**
- Critical rubric not down >2 pp: **False**
- Human review: **pending**; ready to promote: **False**

**Conclusion:** retain the baseline. The LoRA adapter failed the output-format gate, so this schema-only experiment is not eligible for promotion. The dataset was not semantically reviewed; the result is exploratory.

Full pairwise evidence, per-brief outputs, hashes, and token usage: `schema-only-seed42-20260927-benchmark.json`.
