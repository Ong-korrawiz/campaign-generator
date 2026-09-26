# Brief-to-campaign draft generator

Each UTF-8 JSONL input row contains a stable id and a brief matching
campaign_generator.schemas.CampaignBrief:

    {"id":"bakery-001","brief":{"industry":"food","target_audience":"office workers","objective":"increase weekday visits","brand":"Example Bakery","channels":["Instagram"],"constraints":["No discount offers"]}}

The command validates the whole input before making API requests:

    PYTHONPATH=src python -m campaign_generator.dataset_v2.generate \
      --input-file path/to/briefs.jsonl \
      --output-dir data/processed/teacher-run-1 \
      --limit 10

Set OPENAI_API_KEY in the environment or .env. The teacher uses gpt-6-luna
through the Responses API with structured CampaignDirection output and no web
search. The output directory must be new. Successful generations are cached
under data/cache so a rerun to a new output directory can reuse them.

The run writes drafts.jsonl and manifest.json. Every row has
review_status=pending and must be approved before it becomes a training label.
The model receives only the supplied brief; inferred audience insights are
hypotheses, not verified research.

Use the root Makefile to generate from the two checked-in example briefs:

    make dataset

For a custom JSONL, output directory, or row limit:

    make dataset INPUT=path/to/briefs.jsonl OUTPUT=data/processed/my-run LIMIT=10

Each run needs a new OUTPUT path. The default OUTPUT includes the current timestamp.
