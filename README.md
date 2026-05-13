# TalkToDB — Backend API

A FastAPI backend that converts natural language questions into SQL queries using semantic search (Qdrant) and a large language model (vLLM).

---

## How It Works

```
User Question
     │
     ▼
[Input Sanitizer]       — rejects SQL, JSON, code input
     │
     ▼
[Embedding Service]     — all-MiniLM-L6-v2 → 384-dim vector
     │
     ├──▶ [Qdrant: Q&A Collection]    top-5 similar question-SQL pairs
     ├──▶ [Qdrant: DDL Collection]    top-5 relevant table schemas
     └──▶ [Qdrant: Docs Collection]   top-5 relevant documentation chunks
              │
              ▼
        [LLM Service]                 system prompt + all context → SQL
              │
              ▼
        [SQL Validator]               rejects non-SELECT queries
              │
              ▼
        JSON Response  { userMessage, generatedSQL }
```

---

## Project Structure

```
backend/
├── main.py                         # App entry point, lifespan preloading
├── requirements.txt
├── .env                            # All config (never commit this)
├── .env.example                    # Template for env variables
│
├── app/
│   ├── config.py                   # Pydantic settings — reads from .env
│   │
│   ├── prompts/
│   │   └── system_prompt.txt       # System prompt sent to LLM on every request
│   │
│   ├── routes/
│   │   ├── chat.py                 # POST /chat/
│   │   ├── qdrant.py               # GET  /qdrant/search
│   │   ├── training.py             # POST /train/*
│   │   └── collection.py           # GET/PUT/DELETE /collection/*
│   │
│   ├── schemas/
│   │   ├── chat_schema.py
│   │   ├── qdrant_schema.py
│   │   ├── training_schema.py
│   │   └── collection_schema.py
│   │
│   └── services/
│       ├── chat_service.py         # Full pipeline orchestration
│       ├── embedding_service.py    # SentenceTransformer wrapper
│       ├── qdrant_service.py       # All Qdrant operations
│       ├── llm_service.py          # vLLM / OpenAI-compatible call
│       ├── sql_validator.py        # Ensures only SELECT queries pass
│       └── input_sanitizer.py      # Blocks SQL/JSON/code input
│
└── logs/                           # Auto-created. Full LLM prompt saved per request.
```

---

## Qdrant Collections

| Collection | Purpose | Payload Fields |
|------------|---------|----------------|
| `your_question_sql_collection` | Similar Q&A pairs | `question`, `sql`, `timestamp` |
| `your_ddl_collection` | Table schemas | `table_name`, `ddl`, `description`, `type`, `timestamp` |
| `your_docs_collection` | Documentation | `content`, `category`, `type`, `timestamp` |

---

## Setup

### 1. Clone & create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux/Mac
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment
Copy `.env.example` to `.env` and fill in all values:
```bash
cp .env.example .env
```

```env
DEBUG=False

QDRANT_URL=http://<qdrant-host>:<port>
QDRANT_COLLECTION_NAME=<your_question_sql_collection>
QDRANT_DDL_COLLECTION_NAME=<your_ddl_collection>
QDRANT_DOCS_COLLECTION_NAME=<your_docs_collection>
QDRANT_API_KEY=

EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384

OLLAMA_URL=http://<llm-host>:<port>/v1/chat/completions
MODEL_NAME=<your-model-id>
LLM_API_KEY=
TEMPERATURE=0.7
MAX_TOKENS=2048
```

> All variables are **required**. Missing any will log an error and stop startup.

### 4. Run development server
```bash
uvicorn main:app --reload
```

### 5. Run production server
```bash
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

API docs available at: `http://localhost:8000/docs`

---

## API Reference

### Chat

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/chat/` | Convert natural language to SQL |

**Request:**
```json
{ "userMessage": "How many PACS are in Karnataka?" }
```
**Response:**
```json
{
  "userMessage": "How many PACS are in Karnataka?",
  "generatedSQL": "SELECT COUNT(*) FROM ...",
  "error": null
}
```

---

### Qdrant Search

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/qdrant/search?question=...` | Top 5 similar Q&A from collection |

---

### Training — Add data to collections

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/train/question-sql` | Single Q&A pair |
| `POST` | `/train/question-sql/bulk` | Bulk Q&A pairs |
| `POST` | `/train/ddl` | Single DDL record |
| `POST` | `/train/ddl/bulk` | Bulk DDL records |
| `POST` | `/train/docs` | Single documentation record |
| `POST` | `/train/docs/bulk` | Bulk documentation records |

**Bulk Q&A example:**
```json
{
  "records": [
    { "question": "Total PACS count", "sql": "SELECT COUNT(*) FROM ..." },
    { "question": "State-wise PACS",  "sql": "SELECT state_name, COUNT(*) FROM ..." }
  ]
}
```

---

### Collection Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/collection/question-sql` | List all Q&A points |
| `PUT` | `/collection/question-sql/{id}` | Update Q&A point |
| `DELETE` | `/collection/question-sql/{id}` | Delete Q&A point |
| `GET` | `/collection/ddl` | List all DDL points |
| `PUT` | `/collection/ddl/{id}` | Update DDL point |
| `DELETE` | `/collection/ddl/{id}` | Delete DDL point |
| `GET` | `/collection/docs` | List all docs points |
| `PUT` | `/collection/docs/{id}` | Update docs point |
| `DELETE` | `/collection/docs/{id}` | Delete docs point |

---

## Customizing the LLM Behavior

Edit [app/prompts/system_prompt.txt](app/prompts/system_prompt.txt) to change how the LLM responds. This file is loaded fresh on every request — no restart needed.

---

## Logs

Every `/chat/` request saves a `.txt` file in `logs/` with the **full prompt** sent to the LLM:

```
logs/
└── 20260513_154446_How_many_PACS_are_in_Karnataka.txt
```

Each file contains:
```
==== SYSTEM PROMPT ====
...

==== USER PROMPT ====
### Database Schema (DDL):
...

### Documentation Context:
...

### Similar Question-SQL Examples:
...

### Task:
Generate a SQL query for: <userMessage>
```

---

## Input Validation

The `/chat/` endpoint rejects:
- SQL queries (`SELECT`, `INSERT`, `UPDATE`, etc.)
- JSON objects/arrays
- Code snippets (`import`, `def`, `function`, etc.)
- Messages shorter than 3 characters or longer than 500 characters

The generated SQL is also validated — only `SELECT` statements are returned. Any other operation returns an error in the response.

---

## Environment Variables Reference

| Variable | Required | Description |
|----------|----------|-------------|
| `DEBUG` | No | Enables debug mode (default: False) |
| `QDRANT_URL` | ✅ | Qdrant server URL |
| `QDRANT_COLLECTION_NAME` | ✅ | Q&A collection name |
| `QDRANT_DDL_COLLECTION_NAME` | ✅ | DDL collection name |
| `QDRANT_DOCS_COLLECTION_NAME` | ✅ | Docs collection name |
| `QDRANT_API_KEY` | No | Qdrant API key (if enabled) |
| `EMBEDDING_MODEL` | ✅ | SentenceTransformer model name |
| `EMBEDDING_DIMENSION` | ✅ | Embedding vector dimension |
| `OLLAMA_URL` | ✅ | vLLM chat completions endpoint |
| `MODEL_NAME` | ✅ | Model ID used for generation |
| `LLM_API_KEY` | No | Bearer token for LLM (if required) |
| `TEMPERATURE` | ✅ | LLM temperature (0.0–1.0) |
| `MAX_TOKENS` | ✅ | Max tokens in LLM response |
