# Campaign Ideation API

A three-day implementation plan for a marketing campaign ideation service. The intended API accepts a structured campaign brief and returns three distinct campaign concepts in English. Two additional days are reserved for integration and delivery issues.

**Project status:** Shared campaign schemas, the dataset_v2 teacher generator, and evaluation scaffold are implemented. Model training and the API remain planned work.

## Repository structure

```text
campaign-generator/
├── README.md
├── pyproject.toml                    # Dependencies and CLI package configuration
├── .gitignore                        # Excludes downloaded data and model outputs
├── configs/
│   ├── data.yaml                     # Dataset revisions, filters, and split settings
│   └── train.yaml                    # Base model, LoRA settings, seed, and output paths
├── src/campaign_generator/
│   ├── dataset_v2/                   # Structured campaign draft generation
│   ├── io.py                         # Shared JSONL reading and hashing
│   ├── training/                     # Fine-tuning modules (to be implemented)
│   ├── evaluation/
│   │   └── baseline.py               # Base-model inference and metrics
│   ├── prompts/                      # Shared model prompts
│   ├── schemas.py                    # Shared data and model contracts
│   └── __init__.py
├── tests/
│   ├── test_schemas.py               # Shared input and output contracts
│   └── test_api.py                   # Response shape, constraints, and budget checks
├── data/                              # Generated locally; not committed
│   ├── raw/                           # Version-pinned source snapshots
│   └── processed/                     # Normalized train/validation/test files
├── artifacts/                         # Generated locally; not committed
│   ├── adapters/                      # LoRA adapter checkpoints
│   └── reports/                       # Data quality and evaluation results
└── resource/
    ├── DEVELOPMENT_PLAN.md           # Detailed project plan (existing)
    ├── marketting-101.md             # Marketing research notes (existing)
    └── Test Assignment ... .pdf      # Assignment brief (existing)
```

The dataset_v2 package turns validated campaign briefs into reviewable teacher drafts. Training code belongs under training/, while baseline inference and scoring belong under evaluation/. Shared schemas and prompts stay at package level. Generated datasets, checkpoints, and reports stay out of Git.

## Quality checks

Install the hooks after installing the project dependencies:

    .venv/bin/pre-commit install
    .venv/bin/pre-commit run --all-files

The hooks check repository hygiene, Ruff linting and formatting, Python compilation, and the offline unit tests.

## Compare brainstorming models with the OpenAI LLM judge

The judge compares base and fine-tuned predictions on held-out examples. Each
prediction JSONL row needs a source ID and generated answer, for
example:

    {"source_id":"zarn_creative_brief_to_asset_plan_test_0001","output":"1. **Idea name** — Description..."}

Use output, response, or raw_response for the answer field. Answer values may
be text or a JSON object. Base and tuned files must contain the same IDs, and
those IDs must exist in the held-out test JSONL.

The default run evaluates up to 30 examples, selected round-robin by industry,
and makes two judge calls per example to check response-order sensitivity:

    PYTHONPATH=src python -m campaign_generator.evaluation.llm_judge \
      --test-file path/to/brainstorming-compatible/test.jsonl \
      --base-predictions artifacts/reports/base_predictions.jsonl \
      --tuned-predictions artifacts/reports/tuned_predictions.jsonl \
      --output artifacts/reports/llm_judge.json \
      --execute

Set OPENAI_API_KEY in .env; the default judge is gpt-6-astra. Choose a
different model with --model. The report stores the judge model, input hashes,
prompt hash, token usage, per-example evidence, pairwise preferences, and order
disagreement. Deterministic checks report the 10-idea count, unique named
ideas, and presence of prioritization notes. No API calls occur without
--execute; an existing report path is never overwritten.

The judge currently evaluates the 10-idea brainstorming output. KPI, budget,
three-concept API checks should be added when the API response schema is
implemented; the current SFT target does not contain KPI labels.

## Generate campaign drafts with dataset_v2

Each UTF-8 JSONL row contains an id and structured campaign brief. The
generator uses gpt-6-luna with Pydantic output validation and marks drafts
pending review. Set OPENAI_API_KEY in the environment or .env.

    make dataset

For custom inputs and output paths:

    make dataset INPUT=path/to/briefs.jsonl OUTPUT=data/processed/my-run LIMIT=10

The manifest records model, prompt and input hashes, token use, and cache
information. Review generated labels before using them for training.

## Data and model workflow

1. Prepare diverse briefs across industries, objectives, audiences, channels, and constraints.
2. Generate candidate campaign directions with dataset_v2 and retain model, prompt, and input provenance.
3. Review and approve candidates before using them as training labels; split by brief into train, validation, and test.
4. Train a baseline adapter and compare it with the base model on held-out test inputs.
5. Implement the campaign API, validate structured responses, and treat KPI or budget suggestions as proposals.

Teacher outputs are synthetic candidates; review semantic quality before training. Keep model, prompt, source, hashes, and token usage with each dataset run.

## Planned API contract

`POST /generate` accepts a structured JSON brief containing the brand/product, audience, objective, market, permitted channels, constraints, and optional budget with currency. It returns exactly three English concepts. Each concept includes an idea, audience insight, key message, channel roles, an asset/execution plan, and proposed KPIs. If a budget is supplied, any suggested allocation must stay within it.

The exact JSON field names are defined in schemas.py and shared by training and API validation. Numeric KPIs and budget suggestions are estimates: the source data does not provide verified numeric targets.

## Delivery sequence

| Day | Deliverable |
| --- | --- |
| 1 | Versioned data pipeline, normalized dataset, quality report, shared schema, and base-model baseline |
| 2 | Colab CLI fine-tuning, adapter artifact, and base-versus-tuned evaluation |
| 3 | API, response validation, tests, run instructions, and demonstration |
| 4–5 | Buffer for data, training, or integration issues |

See [the development plan](resource/DEVELOPMENT_PLAN.md) for acceptance criteria and limitations. The dataset_v2 generation command is documented above; commands for remaining planned components will be added as they are implemented.
