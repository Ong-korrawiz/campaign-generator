# Campaign Ideation API

A three-day implementation plan for a marketing campaign ideation service. The intended API accepts a structured campaign brief and returns three distinct campaign concepts in English. Two additional days are reserved for integration and delivery issues.

**Project status:** Day 1 data preparation, shared Pydantic schemas, and OpenAI target enrichment are implemented. Model training and the API remain planned work.

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
│   ├── data/                         # Preparation, enrichment, and verification
│   │   ├── pipeline.py               # Download, normalize, filter, and manifest
│   │   ├── enrich.py                 # OpenAI conversion to Brainstorming SFT
│   │   └── verify.py                 # Data quality and split checks
│   ├── training/                     # Fine-tuning modules (to be implemented)
│   ├── evaluation/
│   │   └── baseline.py               # Base-model inference and metrics
│   ├── prompts/                      # Shared model prompts
│   ├── schemas.py                    # Shared data and model contracts
│   └── constants.py                  # Shared constants
├── tests/
│   ├── test_data.py                  # Data shape, deduplication, and split leakage
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

The data package owns dataset preparation and AI-assisted SFT conversion. Training code belongs under training/, while baseline inference and scoring belong under evaluation/. Shared schemas, prompts, and constants stay at package level. The training package is currently a scaffold because fine-tuning code has not been implemented yet. Configuration is separate from code. Downloaded datasets, checkpoints, and generated reports stay out of Git.

## Quality checks

Install the hooks after installing the project dependencies:

    .venv/bin/pre-commit install
    .venv/bin/pre-commit run --all-files

The hooks check repository hygiene, Ruff linting and formatting, Python compilation, and the offline unit tests.

## Generate brainstorming-style targets with OpenAI

Zarn's reference outputs are asset plans, while the candidate Brainstorming dataset uses a natural-language request and a numbered list of ten named campaign ideas. The enrichment CLI uses the OpenAI Responses API with Pydantic structured output to convert each Zarn row into that prompt-and-response style. The source brief and split are preserved; generated labels include model and prompt provenance and are not human-verified. A local SQLite cache lets reruns reuse completed rows for the same input, model, and prompt.

    cp .env.example .env
    # Set OPENAI_API_KEY in .env; optionally change OPENAI_MODEL.
    python -m pip install -e .
    PYTHONPATH=src python -m campaign_generator.dataset.enrich \
      --input-dir data/processed/day1-refactor \
      --output-dir data/processed/zarn-brainstorm-v1 \
      --split train \
      --limit-per-split 1

Remove --limit-per-split to process all rows. Each output must use a new directory. The cache is stored under ignored data/cache/; API cost depends on the selected model and token usage, which the run manifest records.

## Data and model workflow

1. Pin the revisions of [Zarn Creative Brief to Asset Plan](https://huggingface.co/datasets/zarnite/zarn-creative-brief-to-asset-plan) and the candidate [Brainstorming Ideation SFT](https://huggingface.co/datasets/stindardlogic/brainstorming-ideation-sft-100k) source.
2. Convert each usable Zarn brief and asset plan into **one** brainstorming-style campaign concept with its key message, channel roles, and execution plan. Asset-order variants are not separate concepts.
3. Filter contradictory or unsupported claims, deduplicate, and preserve Zarn's published train/validation/test split. Add Brainstorming examples only if enough distinct, valid marketing cases remain; group duplicates before splitting added examples.
4. Compare the base Qwen2.5-1.5B-Instruct model with a LoRA/QLoRA adapter trained through Python CLI scripts on Colab. Evaluate both on the same held-out inputs and save the configuration and results.
5. Generate three concepts with separate inference calls, validate the response, and label KPI targets and budget allocations as proposals. Training examples contain one concept each; producing three is an inference-time behavior.

The enrichment stage creates model-generated labels from Zarn references; they remain synthetic and are not human-reviewed. Record the model, prompt, revisions, hashes, and token usage, and report semantic quality as a limitation.

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

See [the development plan](resource/DEVELOPMENT_PLAN.md) for acceptance criteria and limitations. The OpenAI enrichment command is documented above; commands for remaining planned components will be added as they are implemented.
