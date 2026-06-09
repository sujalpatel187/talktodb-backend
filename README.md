# TalkToDB — Backend API

<p align="center">
  <img src="https://img.shields.io/badge/FastAPI-0.115.12-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Qdrant-1.13.3-FF007F?style=for-the-badge&logo=qdrant&logoColor=white" alt="Qdrant" />
  <img src="https://img.shields.io/badge/PostgreSQL-15+-4169E1?style=for-the-badge&logo=postgresql&logoColor=white" alt="PostgreSQL" />
  <img src="https://img.shields.io/badge/CrossEncoder-BAAI%2Fbge--reranker--base-FF6B35?style=for-the-badge&logo=huggingface&logoColor=white" alt="CrossEncoder" />
</p>

A state-of-the-art FastAPI backend designed to translate natural language user questions into safe, optimized SQL queries and execute them against a PostgreSQL database. It leverages semantic search over schemas and documentation via **Qdrant**, a **Cross-Encoder re-ranking stage** for retrieval precision, and advanced text generation using an OpenAI-compatible large language model (e.g., vLLM, Ollama).

---

## 🛠️ Architecture & Pipeline Flow

The backend employs a multi-stage, defense-in-depth pipeline to ensure that only valid, safe queries are processed and executed.

```
                  User Question
                       │
                       ▼
            ┌──────────────────────┐
            │   Input Sanitizer    │  ◄── Rejects SQL, JSON, code injections
            └──────────┬───────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │  Glossary Resolution │  ◄── Expands acronyms & business terms
            │  (business_glossary) │      from the Glossary Qdrant collection
            └──────────┬───────────┘
                       │  expanded_query
                       ▼
            ┌──────────────────────┐
            │  Embedding Service   │  ◄── Bi-encoder (all-MiniLM-L6-v2)
            └──────────┬───────────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
     ┌──────────┐ ┌──────────┐ ┌──────────┐
     │  Qdrant  │ │  Qdrant  │ │  Qdrant  │
     │ (Q&A×50) │ │(DDL×100) │ │ (Doc×5)  │
     └────┬─────┘ └────┬─────┘ └────┬─────┘
          │            │  BM25       │  passed
          │            │  pre-filter │  directly
          │            ▼  (→ 20)     │  to LLM
          │     ┌──────────────┐    │
          │     │  BM25 Filter │    │
          │     └──────┬───────┘    │
          │            │            │
          └────────────┼────────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │  Cross-Encoder       │  ◄── BAAI/bge-reranker-base
            │  Re-Ranker           │      QA: 50→10  |  DDL: BM25(20)→5
            └──────────┬───────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │   LLM Generation     │  ◄── Generates optimized SQL query
            └──────────┬───────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │    SQL Validator     │  ◄── Enforces SELECT-only validation
            └──────────┬───────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │ PostgreSQL Execution │  ◄── Read-only transactional environment
            └──────────┬───────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │   NL Summary (LLM)   │  ◄── No system prompt, summary only
            └──────────┬───────────┘
                       │
                       ▼
                 JSON Response
     { columns, rows, generatedSQL, userMessage, error }
```

---

## 📂 Project Structure

```
backend/
├── main.py                         # Application entrypoint & startup preloading
├── requirements.txt                # Python dependencies
├── RERANKER.md                     # Re-ranking deep-dive documentation
├── .env                            # Local configuration (Ignored by Git)
├── .env.example                    # Template for environment settings
│
├── app/
│   ├── config.py                   # Pydantic Settings management
│   │
│   ├── prompts/
│   │   └── system_prompt.txt       # System prompt loaded fresh for every request
│   │
│   ├── routes/
│   │   ├── chat.py                 # POST /chat/ — Core NL-to-SQL orchestration endpoint
│   │   ├── qdrant.py               # GET  /qdrant/search — Vector search debug tool
│   │   ├── training.py             # POST /train/* — Q&A, DDL, Docs ingestion
│   │   ├── glossary.py             # POST /train/glossary/* — Business glossary ingestion
│   │   ├── collection.py           # GET/PUT/DELETE /collection/* — Vector data management
│   │   └── execute.py              # POST /execute/ — Direct SQL execution endpoint
│   │
│   ├── schemas/
│   │   ├── chat_schema.py          # Request and response Pydantic schemas
│   │   ├── qdrant_schema.py        # Qdrant search request/response schemas
│   │   ├── training_schema.py      # Q&A, DDL, Docs training record schemas
│   │   ├── glossary_schema.py      # Glossary record and response schemas
│   │   ├── collection_schema.py    # Collection list/update/delete response schemas
│   │   └── execute_schema.py       # Execute request/response schemas
│   │
│   └── services/
│       ├── chat_service.py         # Main pipeline flow coordinator
│       ├── embedding_service.py    # SentenceTransformers bi-encoder (all-MiniLM-L6-v2)
│       ├── reranker_service.py     # Cross-encoder re-ranking + BM25 pre-filter
│       ├── glossary_service.py     # Business glossary resolution & query expansion
│       ├── qdrant_service.py       # Vector DB interaction (CRUD + search)
│       ├── llm_service.py          # OpenAI-compatible API connector
│       ├── pg_service.py           # PostgreSQL transactional query execution
│       ├── sql_validator.py        # Strict AST/regex-based SQL checks
│       └── input_sanitizer.py      # Input sanitation & prompt injection defense
│
└── logs/                           # Auto-generated audit trail per request
    ├── YYYYMMDD_HHMMSS_question.txt                   # System + LLM prompt log
    └── YYYYMMDD_HHMMSS_question_retrieval_debug.txt   # Full retrieval debug log
```

---

## 🗄️ Qdrant Collections

Four dedicated collections store semantic context within the vector database:

| Collection Name | Purpose | Candidates Retrieved | Key Payload Fields |
|:---|:---|:---|:---|
| `your_question_sql_collection` | Verified Question-to-SQL pairs for few-shot examples | 50 → CE → **top 10** | `question`, `sql`, `timestamp` |
| `your_ddl_collection` | Physical and logical database schemas (DDL statements) | 100 → BM25 → 20 → CE → **top 5** | `table_name`, `ddl`, `description`, `type`, `timestamp` |
| `your_docs_collection` | Supplemental text documentation and data dictionaries | **top 5** (sent directly to LLM) | `content`, `category`, `type`, `timestamp` |
| `your_glossary_collection` | Business acronyms, domain terms, and SQL hints | top 5 (resolved before embedding) | `term`, `meaning`, `sql_hint`, `category`, `timestamp` |

The QA and DDL candidates are passed to the **Cross-Encoder Re-Ranker**. DDL candidates are additionally pre-filtered with a **BM25 keyword filter** before cross-encoding. Docs are retrieved directly and injected into the LLM prompt without re-ranking. The Glossary collection is searched before embedding to expand the user query with domain meanings.

---

## � Cross-Encoder Re-Ranking

After Qdrant retrieval, all candidates are re-scored using a **Cross-Encoder** model that reads the query and each document together — producing a much more accurate relevance score than cosine similarity alone.

| Property | Value |
|:---|:---|
| Primary model | `BAAI/bge-reranker-base` |
| Fallback model | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Input token limit | 512 |
| Batch size | 64 pairs per forward pass |
| QA: candidates in → out | 50 → **top 10** |
| DDL: candidates in → BM25 → out | 100 → BM25(20) → **top 5** |
| Docs | top 5 (no re-ranking, sent directly to LLM) |

The model is loaded **once at startup** and cached for the lifetime of the process. Override the model via `.env`:

```env
RERANKER_MODEL=BAAI/bge-reranker-base
```

> See [RERANKER.md](RERANKER.md) for full technical documentation.

---

## 🔒 Security & Safety Layers

To prevent malicious activities, SQL injection, and database alteration, the backend implements multiple layers of protection:

1. **Input Sanitizer**: Blocks inputs that look like direct SQL statements, structured JSON, code blocks (`import`, `def`, `function`), or values outside of the 3–500 character limits.
2. **SQL Validator**: Analyzes generated SQL and strictly rejects anything that is not a pure `SELECT` statement.
3. **Database Guardrails**:
   - Connection established explicitly in **Read-Only Mode** (`conn.set_session(readonly=True, autocommit=False)`).
   - Strict statement timeouts (`statement_timeout = 30000ms`) to protect against resource starvation from runaway Cartesian products.
   - Result limit capped at **500 rows** to avoid memory overload.

---

## 🚀 Getting Started

### 1. Prerequisite Checklist
- **Python 3.10** or higher installed.
- Running instance of **Qdrant Vector Database**.
- Access to a **PostgreSQL Database** (read-only credentials recommended).
- Access to an **OpenAI-compatible LLM provider** (vLLM, Ollama, OpenAI).

### 2. Install & Configure

Clone the repository, initialize your virtual environment, and install dependencies:

```bash
# Initialize and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows PowerShell/CMD
source venv/bin/activate     # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 3. Setup Environment Variables

Copy `.env.example` to `.env` and fill in all variables:

```bash
cp .env.example .env
```

```env
# Application
DEBUG=False

# Qdrant Vector DB
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_NAME=your_question_sql_collection
QDRANT_DDL_COLLECTION_NAME=your_ddl_collection
QDRANT_DOCS_COLLECTION_NAME=your_docs_collection
QDRANT_GLOSSARY_COLLECTION_NAME=your_glossary_collection
QDRANT_API_KEY=

# Embedding Model settings
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384

# Re-Ranking Model (optional — defaults to BAAI/bge-reranker-base)
RERANKER_MODEL=BAAI/bge-reranker-base

# LLM Configurations
OLLAMA_URL=http://localhost:11434/v1/chat/completions
MODEL_NAME=deepseek-coder:6.7b
LLM_API_KEY=
TEMPERATURE=0.0
MAX_TOKENS=2048

# PostgreSQL Target Database
PG_HOST=localhost
PG_PORT=5432
PG_DATABASE=my_production_db
PG_USER=talktodb_readonly
PG_PASSWORD=secure_password_here
```

> [!IMPORTANT]
> A temperature of `0.0` is highly recommended for SQL generation to guarantee deterministic, exact output.

### 4. Running the Servers

* **Development Server (with hot reload):**
  ```bash
  uvicorn main:app --reload
  ```

* **Production Server (multi-worker Gunicorn):**
  ```bash
  gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
  ```

Interactive API documentation is automatically generated and accessible at:
- Swagger UI: `http://localhost:8000/docs`
- Redoc: `http://localhost:8000/redoc`

---

## 📡 API Reference

### 1. Chat Orchestration
* **Endpoint:** `POST /chat/`
* **Request Payload:**
  ```json
  {
    "userMessage": "List all branches with more than 50 employees."
  }
  ```
* **Response Payload:**
  ```json
  {
    "userMessage": "List all branches with more than 50 employees.",
    "generatedSQL": "SELECT branch_name, COUNT(*) FROM ...",
    "columns": ["branch_name", "employee_count"],
    "rows": [
      ["Bengaluru Central", 84],
      ["Hubli North", 52]
    ],
    "row_count": 2,
    "error": null
  }
  ```

### 2. Knowledge Ingestion (Training)
Upload context and schema representations into the semantic indexes:

| Method | Endpoint | Ingested Object |
| :--- | :--- | :--- |
| `POST` | `/train/question-sql` | Single Q&A Example |
| `POST` | `/train/question-sql/bulk` | Multiple Q&A Examples |
| `POST` | `/train/ddl` | Single Table DDL Schema |
| `POST` | `/train/ddl/bulk` | Multiple Table DDL Schemas |
| `POST` | `/train/docs` | Documentation Text Snippet |
| `POST` | `/train/docs/bulk` | Multiple Documentation Snippets |
| `POST` | `/train/glossary` | Single Business Glossary Term |
| `POST` | `/train/glossary/bulk` | Multiple Business Glossary Terms |

**Glossary Record fields:** `term` (acronym/abbreviation), `meaning` (human-readable expansion), `sql_hint` (optional filter hint for SQL generation), `category` (optional grouping label).

### 3. Collection Management
Explore or edit your vectorized data programmatically:
- `GET /collection/{type}` — List all vectorized data points of `{type}` (`question-sql`, `ddl`, `docs`, `glossary`).
- `PUT /collection/{type}/{id}` — Update specific vectorized details.
- `DELETE /collection/{type}/{id}` — Delete outdated data from Qdrant.

### 4. Direct SQL Execution
* **Endpoint:** `POST /execute/`
* **Description:** Execute a raw read-only SQL query directly against PostgreSQL without going through the NL-to-SQL pipeline. Useful for testing and validation.
* **Request Payload:**
  ```json
  { "sql": "SELECT COUNT(*) FROM cooperatives" }
  ```
* **Response Payload:**
  ```json
  { "columns": ["count"], "rows": [[142]], "rowCount": 1, "error": null }
  ```

### 5. Vector Search Debug
* **Endpoint:** `GET /qdrant/search?question=<your+question>`
* **Description:** Performs a raw semantic search against the Q&A collection and returns the top candidates with cosine scores. Useful for tuning retrieval quality.

---

## 🗒️ Logging & Prompt Audits

For every processed `/chat/` query, two debug files are saved in the `logs/` folder.

### Prompt Log — `YYYYMMDD_HHMMSS_question.txt`
```
==== SYSTEM PROMPT ====
You are a highly precise PostgreSQL specialist...

==== USER PROMPT ====
Question: List all branches with more than 50 employees.

### Similar Question-SQL Examples:
...

### Documentation Context:
...

### Database Schema (DDL):
...
```

### Retrieval Debug Log — `YYYYMMDD_HHMMSS_question_retrieval_debug.txt`

| Section | Contents |
|:---|:---|
| 0 — Glossary Resolution | Matched glossary terms with scores, meanings, sql_hints, and the expanded query |
| 1 — Raw QA | All 50 Qdrant QA candidates with cosine score, question |
| 2 — Raw DDL | All 100 Qdrant DDL candidates with cosine score, table, description (BM25 pre-filter applied before CrossEncoder) |
| 3 — Raw Docs | Top 5 Qdrant Docs candidates sent directly to the LLM |
| 4 — Re-ranked QA | Top 10 QA hits that survived cross-encoder re-ranking, with rerank score |
| 5 — Re-ranked DDL | Top 5 DDL hits that survived BM25 + cross-encoder re-ranking, with rerank score |

---

## 🎨 System Prompt Tuning

Modify [app/prompts/system_prompt.txt](app/prompts/system_prompt.txt) to adapt the LLM's query generation style. This file is read **dynamically on every request**, enabling on-the-fly tuning without restarting the server!
