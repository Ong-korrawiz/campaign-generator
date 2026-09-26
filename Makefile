PYTHON ?= .venv/bin/python
INPUT ?= examples/briefs.jsonl
OUTPUT ?= data/processed/teacher-$(shell date +%Y%m%d-%H%M%S)
CACHE ?= data/cache/brief-teacher.sqlite3

.PHONY: dataset

dataset:
	@test -f "$(INPUT)" || { echo "Brief file not found: $(INPUT)" >&2; exit 1; }
	PYTHONPATH=src "$(PYTHON)" -m campaign_generator.dataset_v2.generate --input-file "$(INPUT)" --output-dir "$(OUTPUT)" --cache "$(CACHE)" $(if $(strip $(LIMIT)),--limit "$(LIMIT)",)
