# Config Reference

ตารางนี้อธิบายค่าทุกตัวใน [`src/campaign_generator/config.py`](../src/campaign_generator/config.py) ค่าในคอลัมน์ “ค่าปัจจุบัน” ใช้เป็นค่าเริ่มต้นหรือพฤติกรรมของโค้ด ณ ตอนนี้

## Models และ provenance

| Config | ค่าปัจจุบัน | คำอธิบาย |
| --- | --- | --- |
| `OPENAI_API_KEY_ENV` | `OPENAI_API_KEY` | ชื่อตัวแปร environment ที่เก็บ API key; โค้ดอ่านค่าตอนเรียกใช้งาน |
| `CODE_COMMIT_ENV` | `CODE_COMMIT` | ชื่อตัวแปร environment สำหรับบันทึก Git commit ใน training manifest |
| `GCP_PROJECT_ENV` | `CAMPAIGN_GCP_PROJECT` | ชื่อตัวแปร environment สำหรับกำหนด GCP project ให้ dataset pipeline |
| `BASE_MODEL_ID` | `Qwen/Qwen2.5-1.5B-Instruct` | ID ของโมเดลพื้นฐานที่ใช้ inference, evaluation และเป็นฐานสำหรับ LoRA |
| `BASE_MODEL_REVISION` | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` | revision แบบ pin ของโมเดลพื้นฐาน เพื่อให้โหลดรุ่นเดิมซ้ำได้ |
| `TEACHER_MODEL_ID` | `gpt-6-luna` | โมเดลที่สร้าง label สำหรับ dataset |
| `JUDGE_MODEL_ID` | `gpt-6-astra` | โมเดล judge เริ่มต้นสำหรับเปรียบเทียบผล |
| `PROMPT_VERSION` | `brief-to-campaign-description-v2` | รหัสเวอร์ชัน prompt ที่บันทึกใน provenance ของ dataset |

## API และ inference

| Config | ค่าปัจจุบัน | คำอธิบาย |
| --- | --- | --- |
| `API_TITLE` | `Campaign Generation API` | ชื่อ API ที่แสดงใน FastAPI metadata และเอกสาร OpenAPI |
| `API_VERSION` | `1.0.0` | เวอร์ชัน API ที่แสดงใน metadata |
| `INFERENCE_URL_ENV` | `INFERENCE_URL` | ชื่อตัวแปร environment สำหรับ URL ของ private inference service |
| `INFERENCE_AUDIENCE_ENV` | `INFERENCE_AUDIENCE` | ชื่อตัวแปร environment สำหรับ audience ของ Google identity token |
| `API_MAX_BODY_BYTES` | `8192` bytes | ขนาด body สูงสุดที่ API ยอมรับ |
| `API_TIMEOUT_SECONDS` | `900` วินาที | เวลาสูงสุดที่ API รอการสร้าง concepts ต่อ request |
| `API_MAX_CHANNELS` | `10` | จำนวน channel สูงสุดใน brief |
| `API_MAX_CONSTRAINTS` | `20` | จำนวน constraint สูงสุดใน brief |
| `API_MAX_BRIEF_TEXT_LENGTH` | `1000` ตัวอักษร | ความยาวสูงสุดของแต่ละ field ข้อความใน brief ที่ API ตรวจ |
| `API_REJECTED_LOG_LENGTH` | `500` ตัวอักษร | ความยาวสูงสุดของ error ที่บันทึกใน log เมื่อปฏิเสธ output |
| `API_NAME_SIMILARITY_THRESHOLD` | `0.82` | เกณฑ์ความคล้ายของชื่อ concept ที่ใช้ตรวจชื่อซ้ำ |
| `API_DESCRIPTION_CHANNEL_LIMIT` | `3` ช่องทาง | จำนวนช่องทางสูงสุดที่เติมในประโยคเสริมของคำบรรยาย |
| `API_DESCRIPTION_MIN_SENTENCES` | `2` ประโยค | จำนวนประโยคขั้นต่ำของ `campaign_description` |
| `API_DESCRIPTION_MAX_SENTENCES` | `3` ประโยค | จำนวนประโยคสูงสุดของ `campaign_description` |
| `API_CONCEPT_ATTEMPTS` | `3` ครั้ง | จำนวนครั้งที่ลองสร้าง concept หนึ่งแนวคิด รวมครั้งแรก |
| `API_CONCEPT_COUNT` | `3` concepts | จำนวน concepts ที่การประเมิน LoRA คาดหวังให้สร้างได้ |
| `API_MAX_OUTPUT_TOKENS` | `1536` tokens | เพดาน token ต่อคำตอบที่สร้างจาก inference |
| `INFERENCE_TIMEOUT_SECONDS` | `300.0` วินาที | timeout ของ HTTP client ที่เรียก private inference |
| `INFERENCE_MODEL_NAME` | `campaign-baseline` | ชื่อโมเดลที่ส่งให้ vLLM ใน chat completion request |
| `INFERENCE_TEMPERATURE` | `0.2` | ค่า temperature ของ API inference |

## Dataset v3

| Config | ค่าปัจจุบัน | คำอธิบาย |
| --- | --- | --- |
| `DATASET_NAME` | `baseline-v3-description-seed42` | ชื่อ dataset ที่ใส่ใน manifest และ quality report |
| `DATASET_SEED` | `42` | seed สำหรับสุ่มลำดับและแบ่ง synthetic briefs |
| `DATASET_SYNTHETIC_BRIEF_COUNT` | `120` briefs | จำนวน brief สมมติที่ generator สร้าง |
| `DATASET_CANDIDATE_COUNT` | `240` แถว | จำนวน candidate ทั้งชุดหลังรวม brief ต้นทางกับ brief สมมติ |
| `DATASET_TEACHER_WORKERS` | `4` workers | จำนวน worker ที่เรียก teacher พร้อมกัน |
| `DATASET_PROGRESS_INTERVAL` | `20` แถว | ช่วงจำนวนงานที่ใช้พิมพ์สถานะความคืบหน้า |
| `DATASET_LEGACY_SPLIT_COUNTS` | train 96, validation 12, test 12 | จำนวนแถวเดิมในแต่ละ split ที่ pipeline คาดหวัง |
| `DATASET_SYNTHETIC_SPLIT_COUNTS` | train 72, validation 12, test 36 | จำนวน brief สมมติที่กำหนดให้แต่ละ split |
| `DATASET_SPLIT_COUNTS` | train 168, validation 24, test 48 | จำนวนแถวเป้าหมายของ dataset v3 ทั้งชุด |
| `DATASET_DEFAULT_SOURCE_BRIEFS` | `dataset/source_briefs.jsonl` | path เริ่มต้นของ brief ต้นทาง |
| `DATASET_DEFAULT_OUTPUT_DIR` | `data/processed/dataset-v3` | path เริ่มต้นสำหรับผลสร้างหรือ finalize dataset |
| `DATASET_DEFAULT_CACHE` | `data/cache/baseline-v3-description.sqlite3` | path เริ่มต้นของ cache คำตอบ teacher |
| `DEFAULT_GCP_PROJECT_ID` | `campaign-generator-509812` | project ID เริ่มต้นเมื่อไม่ได้กำหนดผ่าน environment หรือ CLI |
| `DATASET_BUCKET_SUFFIX` | `-dataset` | suffix ที่ต่อท้าย project ID เพื่อสร้างชื่อ GCS bucket |
| `DATASET_FINAL_FILES` | `manifest.json`, `train.jsonl`, `validation.jsonl`, `test.jsonl`, `quality_report.md` | รายชื่อไฟล์ที่อัปโหลดไปยัง dataset version ที่สมบูรณ์ |
| `DATASET_DRAFT_FILES` | `manifest.json`, `drafts.jsonl`, `review_queue.jsonl`, `review_template.jsonl`, `synthetic_briefs.jsonl`, `brief_manifest.json`, `HUMAN_REVIEW.md` | รายชื่อไฟล์ที่อัปโหลดไปยัง GCS staging |

## LoRA training

| Config | ค่าปัจจุบัน | คำอธิบาย |
| --- | --- | --- |
| `TRAIN_MAX_SEQUENCE_LENGTH` | `4096` tokens | ความยาว sequence สูงสุดในการ train |
| `TRAIN_SEQUENCE_PERCENTILE` | `0.95` | percentile ของความยาวตัวอย่างที่ใช้เลือก sequence length |
| `TRAIN_SEQUENCE_ROUND_TO` | `256` tokens | ปัด sequence length ขึ้นเป็นช่วงละเท่านี้ |
| `TRAIN_SMOKE_SEQUENCE_LIMIT` | `1024` tokens | เพดาน sequence length ใน smoke training |
| `TRAIN_SMOKE_ROW_COUNT` | `8` แถว | จำนวนตัวอย่างที่ใช้ใน smoke overfit |
| `TRAIN_SMOKE_MAX_STEPS` | `20` steps | จำนวน training steps ใน smoke overfit |
| `TRAIN_SMOKE_MICROBATCH` | `1` | batch size ต่อ GPU สำหรับ smoke training |
| `TRAIN_SMOKE_GRADIENT_ACCUMULATION` | `1` step | จำนวน accumulation steps ใน smoke training |
| `TRAIN_DEFAULT_MAX_STEPS` | `-1` | ค่า Transformers ที่หมายถึงให้ train ตามจำนวน epoch แทนการกำหนด max steps |
| `TRAIN_LABEL_IGNORE_INDEX` | `-100` | label value ที่ loss function ข้าม เช่น token ของ prompt |
| `TRAIN_LORA_R` | `16` | rank ของ LoRA adapter |
| `TRAIN_LORA_ALPHA` | `32` | scaling parameter ของ LoRA |
| `TRAIN_LORA_DROPOUT` | `0.05` | dropout ของ LoRA adapter |
| `TRAIN_LORA_BIAS` | `none` | วิธีฝึก bias parameters ของ LoRA |
| `TRAIN_LORA_TASK_TYPE` | `CAUSAL_LM` | ประเภทงานของ PEFT adapter |
| `TRAIN_LORA_TARGET_MODULES` | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` | โมดูลของ transformer ที่ติดตั้ง LoRA adapter |
| `TRAIN_LEARNING_RATE` | `0.0002` | learning rate ของ training |
| `TRAIN_EPOCHS` | `3` | จำนวน epoch ปกติ |
| `TRAIN_MICROBATCH` | `4` | batch size ต่อ GPU ใน training ปกติ |
| `TRAIN_EVAL_BATCH` | `4` | batch size ต่อ GPU ระหว่าง validation |
| `TRAIN_GRADIENT_ACCUMULATION` | `8` steps | จำนวน accumulation steps ใน training ปกติ |
| `TRAIN_LOGGING_STEPS` | `5` steps | ความถี่ในการบันทึก training logs |
| `TRAIN_CHECKPOINT_STEPS` | `25` steps | ความถี่ในการบันทึก checkpoint และประเมิน validation |
| `TRAIN_CHECKPOINT_LIMIT` | `2` checkpoints | จำนวน checkpoint ที่เก็บไว้ในพื้นที่ training output |
| `TRAIN_EVAL_METRIC` | `eval_loss` | metric ที่ใช้เลือก checkpoint ที่ดีที่สุด |
| `TRAIN_SEED` | `42` | seed สำหรับการฝึกและการสุ่มที่เกี่ยวข้อง |
| `TRAIN_DEFAULT_WORK_DIR` | `/tmp/campaign-training` | path เริ่มต้นสำหรับไฟล์ชั่วคราวและ output ของ training |

## Evaluation และ promotion gate

| Config | ค่าปัจจุบัน | คำอธิบาย |
| --- | --- | --- |
| `EVAL_HOLDOUT_COUNT` | `36` briefs | จำนวน brief ใน primary synthetic holdout |
| `EVAL_LIVE_LIMIT` | `20` briefs | จำนวน brief เริ่มต้นสำหรับตรวจ live API |
| `EVAL_LIVE_TIMEOUT_SECONDS` | `900.0` วินาที | timeout ของ HTTP client สำหรับ live API check |
| `EVAL_LIVE_BUDGET_MIN` | `100000` | งบต่ำสุดของ brief ตัวอย่างที่ใช้ทดสอบ API |
| `EVAL_LIVE_BUDGET_MAX` | `300000` | งบสูงสุดของ brief ตัวอย่างที่ใช้ทดสอบ API |
| `EVAL_LIVE_BUDGET_CURRENCY` | `THB` | สกุลเงินของงบใน brief ตัวอย่าง |
| `EVAL_JUDGE_EXPECTED_IDEAS` | `10` | จำนวน ideas ที่ judge คาดหวังในโหมดเดิม |
| `EVAL_JUDGE_MODE` | `ideas` | โหมดประเมินเริ่มต้นของ judge |
| `EVAL_JUDGE_LIMIT` | `30` briefs | จำนวนตัวอย่างสูงสุดที่ judge ประเมินโดยปริยาย |
| `EVAL_BOOTSTRAP_SEED` | `42` | seed ของ bootstrap confidence interval |
| `EVAL_BOOTSTRAP_SAMPLES` | `10000` samples | จำนวน bootstrap resamples สำหรับ confidence interval |
| `EVAL_DIAGNOSTIC_PER_SOURCE` | `6` briefs | จำนวนตัวอย่างต่อ source ใน prompt diagnostic |
| `EVAL_LORA_PER_SOURCE` | `3` briefs | จำนวนตัวอย่างต่อ source ใน LoRA three-concept check |
| `EVAL_LORA_MAX_ATTEMPTS` | `6` attempts | จำนวน generation attempts สูงสุดต่อ brief ใน LoRA check |
| `EVAL_LORA_TEMPERATURE` | `0.7` | temperature สำหรับ generation แบบสุ่มใน LoRA check |
| `EVAL_LORA_TOP_P` | `0.9` | top-p สำหรับ generation แบบสุ่มใน LoRA check |
| `PROMOTION_MIN_NET_WIN_POINTS` | `10` percentage points | net win ขั้นต่ำที่ tuned model ต้องทำได้ |
| `PROMOTION_MIN_FORMAT_PASS_RATE` | `0.98` หรือ `98%` | format pass rate ขั้นต่ำของ tuned model |
| `PROMOTION_MAX_RUBRIC_REGRESSION_POINTS` | `2` percentage points | rubric สำคัญถดถอยได้ไม่เกินค่านี้ |
