# Cross-Encoder Re-Ranking in TalkToDB

## What is Re-Ranking?

The retrieval pipeline uses **two separate models** working in sequence:

| Stage | Model Type | Purpose |
|---|---|---|
| 1 — Retrieval | Bi-Encoder (`all-MiniLM-L6-v2`) | Embed query and documents independently; fast ANN search in Qdrant |
| 2 — Re-Ranking | Cross-Encoder (`BAAI/bge-reranker-base`) | Score every (query, document) pair together; much more accurate but slower |

A **bi-encoder** maps query and document to separate vectors and measures cosine distance. It is fast but imprecise because the two texts never "see" each other during encoding.

A **cross-encoder** receives the query and document concatenated and passes them through a transformer together, producing a single relevance score. Because both texts interact through attention, the score is far more accurate — at the cost of being too slow to run over millions of documents. Re-ranking solves this by first narrowing to a small candidate set with the bi-encoder, then scoring that set precisely with the cross-encoder.

---

## Models Used

### Primary — `BAAI/bge-reranker-base`

- Published by the Beijing Academy of Artificial Intelligence (BAAI).
- Fine-tuned specifically for **passage re-ranking** from the BGE family.
- Architecture: `bert-base` (110 M parameters).
- Input limit: 512 tokens.
- Output: a single raw logit (higher = more relevant); no sigmoid applied.
- Best choice: English and Chinese text, general-purpose retrieval.
- HuggingFace: https://huggingface.co/BAAI/bge-reranker-base

### Fallback — `cross-encoder/ms-marco-MiniLM-L-6-v2`

- Trained on the Microsoft MARCO passage ranking dataset.
- Architecture: `MiniLM-L6` (22 M parameters) — 5× smaller, faster on CPU.
- Output: raw logit compatible with the primary model's score range.
- Used automatically if the primary model fails to download or load.
- HuggingFace: https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2

### Model Resolution Order

```
1. RERANKER_MODEL env variable  (optional override in .env)
2. BAAI/bge-reranker-base       (primary hard-coded default)
3. cross-encoder/ms-marco-MiniLM-L-6-v2  (fallback)
```

If all three fail to load, the application raises a `RuntimeError` at startup.

---

## Pipeline Position

```
User Question
     │
     ▼
[1] Bi-Encoder embedding  (all-MiniLM-L6-v2)
     │
     ▼
[2] Qdrant ANN search  —  retrieves candidates
     │   • QA collection   : top 50
     │   • DDL collection  : top 10
     │   • Docs collection : top 10
     │   Total: up to 70 candidates
     ▼
[3] ★ Cross-Encoder Re-Ranking ★  (BAAI/bge-reranker-base)
     │   Scores every (query, document) pair in one batch
     │   Sorts all 70 by rerank_score descending
     │   Returns top 10 across all collections
     ▼
[4] Partition by source  →  qa_hits / ddl_hits / docs_hits
     │
     ▼
[5] Build DDL / Docs / QA context blocks
     │
     ▼
[6] LLM prompt  →  SQL generation  →  execution  →  NL summary
```

---

## How Each Collection's Document is Formatted for the Cross-Encoder

The cross-encoder needs a single text string per candidate. The function
`_extract_document_text()` converts each Qdrant payload as follows:

| Collection | Input text sent to cross-encoder |
|---|---|
| `qa` | `"Question: <question>\nSQL: <sql>"` |
| `ddl` | `"Table: <name>\nDescription: <desc>\nDDL: <ddl>"` |
| `docs` | `"<content>"` (raw content field) |

---

## Key Configuration

All tuneable values live in `reranker_service.py` as module-level constants:

| Constant | Default | Description |
|---|---|---|
| `_PRIMARY_MODEL` | `BAAI/bge-reranker-base` | First model tried |
| `_FALLBACK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Used if primary fails |
| `_MAX_LENGTH` | `512` | Token limit passed to `CrossEncoder()` |
| `_BATCH_SIZE` | `64` | Pairs per forward pass; reduce if OOM |

Override the model via `.env`:

```env
RERANKER_MODEL=BAAI/bge-reranker-base
```

---

## Implementation Details

### Singleton Loading (`get_reranker`)

The `CrossEncoder` is loaded **once** during FastAPI's `lifespan` startup hook
and stored as a module-level variable `_reranker`. Every subsequent call to
`get_reranker()` returns the cached instance with no I/O.

```
main.py  →  lifespan()  →  get_reranker()  →  CrossEncoder loaded into RAM
```

### Synchronous Path (`rerank_sync`)

Used by `run_chat_pipeline` (the current synchronous pipeline):

1. Builds a list of `(query_text, document_text)` pairs.
2. Calls `reranker.predict(pairs, batch_size=64)` — returns a NumPy array of scores.
3. Zips scores back to their `(ScoredPoint, source)` tuples.
4. Sorts descending by score.
5. Returns the top-`k` as `RerankedHit` dataclass instances.

### Async-Safe Path (`rerank`)

An `async def` wrapper that submits `rerank_sync` to a dedicated
`ThreadPoolExecutor` (2 workers, named `reranker-*`). This ensures the
FastAPI event loop is never blocked by CPU-bound inference, making it
ready for any future async migration of the pipeline.

```python
loop.run_in_executor(_executor, rerank_sync, query, candidates, top_k)
```

### `RerankedHit` Dataclass

```python
@dataclass(frozen=True)
class RerankedHit:
    hit: ScoredPoint   # original Qdrant result — payload fully preserved
    source: str        # "qa" | "ddl" | "docs"
    rerank_score: float  # raw cross-encoder logit
```

The original `ScoredPoint` (with all Qdrant payload fields) is preserved
untouched. The `source` tag allows the caller to partition results back by
collection after re-ranking.

---

## Debug Logging

Every query produces two files in `backend/logs/`:

| File | Contents |
|---|---|
| `{timestamp}_{question}.txt` | System prompt + full LLM user prompt |
| `{timestamp}_{question}_retrieval_debug.txt` | 6-section retrieval debug log (see below) |

### `_retrieval_debug.txt` Sections

| # | Section | What it shows |
|---|---|---|
| 1 | Raw QA candidates | All 50 Qdrant QA hits with cosine score, question, SQL |
| 2 | Raw DDL candidates | All 10 Qdrant DDL hits with cosine score, table, DDL |
| 3 | Raw Docs candidates | All 10 Qdrant Docs hits with cosine score, category, content |
| 4 | Re-ranked QA → LLM | QA hits that survived re-ranking, with cross-encoder score |
| 5 | Re-ranked DDL → LLM | DDL hits that survived re-ranking, with cross-encoder score |
| 6 | Re-ranked Docs → LLM | Docs hits that survived re-ranking, with cross-encoder score |

---

## File Locations

```
backend/
├── app/
│   ├── services/
│   │   └── reranker_service.py   ← all re-ranking logic
│   ├── config.py                 ← reranker_model setting
├── main.py                       ← startup preloading
└── RERANKER.md                   ← this file
```
