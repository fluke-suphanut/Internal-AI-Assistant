## Internal AI Assistant (LLM + FAISS)

ระบบ AI Assistant แบบ API-first
ใช้ Large Language Models (LLMs) ร่วมกับ Vector Search (FAISS) เพื่อ

- ตอบคำถามจากเอกสารภายในองค์กร
- สรุปปัญหา / bug / feedback ให้อยู่ในรูปแบบข้อมูลเชิงโครงสร้าง (JSON)

### Features

- ถาม–ตอบจากเอกสาร (Document Q&A) ด้วย Vector Search + LLM
- AI Agent เลือกเครื่องมือที่เหมาะสมให้อัตโนมัติ
- Structured Output (JSON) ลดปัญหา hallucination
- FAISS สำหรับจัดเก็บและค้นหา embeddings แบบเรียบง่ายและรวดเร็ว
- FastAPI + Swagger UI ใช้งานและทดสอบ
- รองรับ Docker พร้อม deploy

### Project Structure

```
ai-assistant/
├─ app/
│  ├─ main.py
│  ├─ core/
│  │  ├─ config.py
│  │  └─ logging.py
│  ├─ schemas/
│  │  ├─ requests.py
│  │  └─ responses.py
│  ├─ ingestion/
│  │  ├─ loader.py
│  │  ├─ splitter.py
│  │  ├─ embeddings.py
│  │  └─ build_index.py
│  ├─ retriever/
│  │  ├─ faiss_store.py
│  │  └─ search.py
│  ├─ tools/
│  │  ├─ internal_qa_tool.py
│  │  └─ issue_summary_tool.py
│  ├─ agent/
│  │  ├─ prompts.py
│  │  ├─ router.py
│  │  └─ agent.py
│  └─ utils/
│     ├─ json_guard.py
│     └─ trace.py
│
├─ scripts/
│  ├─ ingest.py
│  └─ smoke_test.py
│
├─ data/
│  ├─ ai_test_bug_report.
│  └─ ai_test_user_feedback.
│
├─ storage/
│  ├─ faiss_index/
│  └─ manifest.json
│
├─ Dockerfile
├─ requirements.txt
├─ .env.example
└─ README.md
```

### Tools

1. Internal Q&A Tool

- ใช้ FAISS Vector Search ค้นหาข้อมูลที่เกี่ยวข้อง
- ส่ง context ให้ LLM ตอบคำถาม
- คืนค่า:
  - คำตอบ
  - แหล่งอ้างอิง (citations)
  - ระดับความมั่นใจ

2. Issue Summary Tool

- ใช้ LLM สรุปข้อความปัญหา
- ดึงข้อมูลออกมาเป็นโครงสร้าง:
  - รายการปัญหา
  - ส่วนที่ได้รับผลกระทบ
  - ระดับความรุนแรง
  - คืนค่าเป็น JSON

### หลักการทำงานของ AI Agent

1. รับ input จากผู้ใช้
2. ใช้ LLM ตัดสินใจเลือก tool ที่เหมาะสม
3. อธิบายเหตุผลในการเลือก
4. เรียก tool และส่งผลลัพธ์กลับเป็น structured output

### การติดตั้งและรัน

1. สร้าง Virtual Environment

```bash
python -m venv .venv

.venv\Scripts\activate
```

2. ติดตั้ง Dependencies

```bash
pip install -r requirements.txt
```

3. แก้ไข .env

```bash
OPENAI_API_KEY=xxxxxxxx
```

สร้าง FAISS Index

```bash
python -m scripts.ingest
```

ระบบจะ

- โหลดเอกสาร
- ตัดเป็น chunk
- สร้าง embeddings
- build FAISS index
- สร้างไฟล์ manifest.json

รัน API

```bash
uvicorn app.main:app --reload
```

เปิด Swagger UI

```bash
http://localhost:8000/docs
```

Smoke Test

```bash
python -m scripts.smoke_test
```

### ตัวอย่างการเรียกใช้งาน API

POST /`ask`

```bash
{
  "query": "What issues were reported on email notification?",
  "top_k": 5
}
```

POST /`summarize`

```bash
{
  "issue_text": "Users report email notifications are delayed during peak hours."
}
```

### เหตุผลในการออกแบบระบบนี้

- เป็น API-first เชื่อมต่อระบบอื่นได้ง่าย
- ใช้ structured output ลดความผิดพลาดจาก LLM
- แยกส่วน ingestion / agent / tools ชัดเจน
- สามารถ rebuild index ได้ (reproducible)
- รองรับการ deploy จริงด้วย Docker
