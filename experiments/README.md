# Dataset v3 experiment evidence

One LoRA training run has been completed on dataset v3: `schema-only-seed42-20260927`, Vertex CustomJob `4823415399175421952`. Its dataset URI is `gs://campaign-generator-509812-dataset/versions/2368fab94e9eb94deb91cdd79526a3dc8107240e5c97ee424ad35317aec8620d`.

The run directory contains the training manifest, baseline and tuned predictions, pairwise benchmark report, prompt diagnostic outputs, and three-concept smoke outputs. Original adapter, checkpoint, and result files remain at `gs://campaign-generator-509812-model-artifacts/runs/schema-only-seed42-20260927/`. Related Vertex jobs: base predictions `7616667514935705600`, tuned predictions `5049721280450789376`, prompt diagnostic `1795870549675606016`, and three-concept diagnostic `7611038015401492480`. The earlier training connectivity smoke job `366435064822628352` did not train a model.

On the 36-brief holdout, the original baseline yielded three schema-valid concepts for 7/36 briefs and the tuned model for 0/36; the tuned adapter was not promoted. The later public API structural check passed 20/20 separate briefs, while human spot checks found generic ideas and unsupported claims. See `api-acceptance.md` and `public-api-20.json` for that test. Subsequent runs should use a unique `RUN_ID` so every result remains separate.
