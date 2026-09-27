# Fine-tuning dataset v3

This is the only committed training dataset. `train.jsonl`, `validation.jsonl`, and `test.jsonl` contain 168, 24, and 48 `TrainingRecordV2` rows. `manifest.json` records source/model/prompt provenance and file hashes; `quality_report.md` records schema-only review status. `source_briefs.jsonl` contains the 120 pinned-source briefs and split assignments needed by the generator; the 120 fictional briefs are generated deterministically by code.

Teacher labels have not received human semantic review. Use this dataset for the documented experiment, not as evidence of production-ready campaign quality. The training run used the immutable GCS URI named in the root README.
