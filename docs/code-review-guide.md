# Code Review Guide

คู่มือนี้ช่วยรีวิว repository ของ Campaign Generator ก่อน merge หรือส่งมอบ โดยให้ตรวจ contract, ความปลอดภัย และหลักฐานการประเมินควบคู่กับโค้ด

## 1. เริ่มจากภาพรวม

อ่าน [README](../README.md), [model card](model-card.md), [dataset report](../dataset/quality_report.md) และ [ผลการทดลอง](../experiments/README.md) ก่อน แล้วดู diff ทั้งหมด:

```bash
git status --short
git diff --check
git diff --stat
git diff
```

ตรวจว่า change มีขอบเขตตรงกับงาน ไม่มี `.env`, token, key, ไฟล์ชั่วคราว หรือ dataset สำเนาเพิ่มเข้ามา

## 2. ตามเส้นทางหลักของระบบ

| ส่วน | จุดเริ่มอ่าน | สิ่งที่ต้องยืนยัน |
| --- | --- | --- |
| Config | `src/campaign_generator/config.py` | ค่า API, dataset, model, LoRA และเกณฑ์ประเมินมีจุดอ้างอิงเดียว; อ่าน environment ตอนใช้งาน |
| API | `src/campaign_generator/api/app.py`, `backend.py`, `service.py` | `POST /generate` ตรวจ brief, คืน 3 concepts ตาม schema, map timeout/error ชัดเจน และไม่เปิดเผยข้อมูลภายใน |
| Contract และ prompt | `schemas.py`, `prompts/campaign.py` | field และข้อกำหนดตรงกัน; `campaign_description` มีรายละเอียดจาก brief และไม่มีการอ้างผลลัพธ์เกินหลักฐาน |
| Dataset v3 | `dataset/`, `dataset_v3/pipeline.py`, `dataset_v3/gcs.py` | split 168/24/48, schema/hash/manifest ตรงกัน, provenance ครบ และการอัปโหลดซ้ำไม่เขียนทับข้อมูลที่ hash ต่างกัน |
| Training | `training/train_lora.py`, `infra/training/` | อ่านเฉพาะ train/validation, คิด loss เฉพาะ assistant completion, บันทึก checkpoint และ manifest พร้อม resume ได้ |
| Evaluation | `evaluation/`, `experiments/runs/` | เทียบ baseline กับ tuned บน brief และ schema เดียวกัน; รายงาน format pass แยกจากคุณภาพเชิงความหมาย |
| Deployment | `terraform/`, `scripts/`, `infra/` | API เรียก inference แบบ authenticated, ค่า timeout/scaling สมเหตุผล, ไม่มี public access ที่ไม่ตั้งใจ และ deployment ใช้ image digest |

## 3. ตรวจคุณภาพพฤติกรรม

สุ่มอ่าน brief และผลลัพธ์จาก train, validation และ test แยกกัน แล้วถามว่า:

- ทั้งสามแนวคิดต่างกันที่ insight หรือกลไกแคมเปญจริง ไม่ใช่เปลี่ยนแค่ชื่อ
- คำบรรยายอธิบายแนวคิด วิธีนำไปใช้ และหน้าที่ของช่องทาง โดยอิงข้อมูล brief
- KPI และ budget ระบุชัดว่าเป็นข้อเสนอ ไม่ได้แสดงเป็นผลลัพธ์ที่วัดแล้ว
- ไม่มีสถิติ ข้อเท็จจริง หรือคุณสมบัติสินค้าเพิ่มเอง
- การทดสอบไม่ใช้ test split เพื่อปรับ prompt หรือฝึกโมเดล

Schema pass อย่างเดียวไม่ยืนยันว่าไอเดียดีหรือถูกต้อง ให้อ่านตัวอย่างและดูผล benchmark ประกอบเสมอ

## 4. รัน checks

```bash
.venv/bin/python -m pytest -q
python -m pip install ruff
ruff check src tests
.venv/bin/python -m campaign_generator.dataset_v3.pipeline --help
```

รันเฉพาะคำสั่งที่ตรงกับไฟล์ซึ่งเปลี่ยน และระบุใน review ว่า check ใดผ่านหรือยังไม่ได้รัน การตรวจ infrastructure ใช้ `terraform fmt -check -recursive terraform` และ `terraform validate` ในแต่ละ state หลัง init ด้วย backend ที่ถูกต้อง โดยไม่ apply ระหว่าง code review

## 5. สรุปผลรีวิว

เขียนข้อพบก่อน โดยอ้างไฟล์และบรรทัด ระบุผลกระทบกับผู้ใช้หรือข้อมูล แล้วแยกเป็น:

1. **Blocker** — เสี่ยงข้อมูล/credential รั่ว, API ใช้งานไม่ได้, หรือทำลายข้อมูลผิดชุด
2. **ควรแก้ก่อน merge** — contract พัง, ผลประเมินคลาดเคลื่อน, หรือขาด validation สำคัญ
3. **ข้อเสนอแนะ** — ปรับความอ่านง่ายหรือดูแลรักษา โดยไม่ขัดขวางการส่งมอบ

ถ้าไม่พบปัญหา ให้บอกขอบเขตที่ตรวจและ checks ที่รันแล้ว พร้อมระบุข้อจำกัดของหลักฐาน เช่น ยังไม่มี human semantic review
