# SHL Conversational Assessment Recommender

## Deployment & Repository

- **Live API:** https://shl-assignment-f2wd.onrender.com
- **Health Endpoint:** https://shl-assignment-f2wd.onrender.com/health
- **API Docs:** https://shl-assignment-f2wd.onrender.com/docs
- **GitHub:** https://github.com/Khushdeep17/SHL-Assignment

---

## Overview

A retrieval-grounded conversational agent that helps hiring managers find the right SHL Individual Test Solutions through dialogue. The system supports clarification for vague queries, grounded recommendations, mid-conversation refinement, comparison queries, and refusal handling for out-of-scope requests.

The API exposes two endpoints — `GET /health` and `POST /chat` — and is fully stateless: every request carries the complete conversation history.

---

## System Design

```
POST /chat (full history)
        │
        ▼
  Intent Detection          ← rule-based: compare / clarify / refuse / end
        │
        ▼
  Hybrid Retrieval          ← BM25 (0.35) + FAISS semantic (0.65) + heuristic boosting
        │
        ▼
  LLM Response Generation   ← grounded on retrieved catalog context only
        │
        ▼
  URL Safety Filter         ← strips any recommendation not in retrieval pool
        │
        ▼
  FastAPI Response          ← { reply, recommendations[], end_of_conversation }
```

The most important architectural decision was shifting recommendation control away from the LLM and into the retrieval layer. Earlier versions relied too heavily on generative reasoning, which caused hallucinated assessments and inconsistent refinement behavior. The final system improved reliability by grounding all recommendations in retrieved catalog candidates and validating every returned URL before generating the final response.

---

## Catalog Ingestion

The provided SHL catalog JSON was used directly. Each item was enriched with a normalized `test_type` code mapped from its `keys` array (`K`, `P`, `A`, `B`, `S`, `C`, `D`) and a composite `search_text` field combining name, description, job levels, and keys — used for both embedding and BM25 indexing.

**Embedding model:** `sentence-transformers/all-MiniLM-L6-v2` → FAISS `IndexFlatIP` (cosine via L2 normalization). BM25 via `rank-bm25`. Both indexes are built at startup and cached.

---

## Retrieval Design

**Why hybrid search?** Semantic search captures intent but misses exact names. BM25 catches abbreviations (OPQ, GSA, DSI, SVAR) but misses paraphrase. Combining both at 65/35 weight outperformed either alone on the public traces.

**Query expansion:** Domain-specific trigger→synonym maps for leadership, graduate, contact center, safety, admin, and healthcare roles — plus common SHL abbreviations. Tuned iteratively using the provided sample traces.

**Heuristic boosting:** Post-combination score adjustments for high-signal patterns — e.g. `"cxo"` → +0.30 for OPQ32r, `"contact center"` → +0.30 for SVAR, `"safety"` → +0.30 for DSI. Learned from trace patterns.

**Diversity filter:** Deduplicates by normalized base name to avoid returning multiple report variants when an instrument is more relevant.

---

## Conversational Agent & Prompt Design

Intent is classified rule-first for latency and reliability:

| Intent | Trigger | Action |
|--------|---------|--------|
| `refuse` | Prompt injection / off-topic | Static refusal, no retrieval |
| `legal_refuse` | "legally required", "does this satisfy" | Specific legal disclaimer |
| `compare` | "difference between" / "vs" + assessment keyword | Retrieve both items, compare from catalog only |
| `end` | Confirmation phrases | `end_of_conversation: true`, preserve prior shortlist |
| `clarify` | Vague first turn | Ask ONE focused question |
| `recommend` | Default | Recommend from retrieval pool |

Clarification behavior was tuned iteratively using the provided traces. The system avoids over-asking via lightweight heuristics for leadership, graduate, safety, and customer-service scenarios, and commits to a recommendation after 2–3 turns regardless.

On final confirmation turns, the system extracts the previously committed shortlist from conversation history instead of re-running retrieval — preventing shortlist drift on turns like "confirmed" or "locking it in."

Low-temperature decoding was used for stable and deterministic outputs. Every recommendation is validated against the retrieval pool before returning, structurally preventing hallucinated URLs.

LLM inference runs on **LLaMA 3.3 70B** (primary) with **LLaMA 3.1 8B** as a rate-limit fallback via the Groq API.

---

## Evaluation Approach

Tested against all 10 public traces by replaying each turn and checking:

- Schema compliance on every response
- Correct items in final shortlist (Recall@10)
- Clarification triggered only for vague queries on turns 1–2
- Refinements (`"add X"`, `"drop Y"`) preserved prior items correctly
- Legal refusal triggered on HIPAA compliance question
- `end_of_conversation: true` on all confirmation turns
- No hallucinated URLs in any response

---

## What Didn't Work

- **Pure LLM recommendations** caused hallucinated assessments and inconsistent refinement. Fixed by retrieval-first grounding.
- **Semantic-only retrieval** failed on abbreviations (GSA, DSI, SVAR). Fixed by hybrid search.
- **Aggressive clarification** hurt flow on clear queries. Fixed by role-specific skip logic.
- **Re-running retrieval on confirmation turns** caused shortlist drift. Fixed by extracting the prior committed shortlist from conversation history.

---

## Technology Stack

FastAPI · Groq API · LLaMA 3.3 70B · FAISS · BM25 · sentence-transformers · Render · GitHub

---

AI-assisted tools were used for debugging, code refactoring, prompt iteration, and deployment troubleshooting. The retrieval architecture, ranking logic, conversational workflow, and evaluation strategy were manually designed and validated throughout development.