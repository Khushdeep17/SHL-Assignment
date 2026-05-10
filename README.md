# SHL Conversational Assessment Recommender — Approach Document

**Candidate:** Khushdeep Singh | **Role:** AI Research Intern | **Submission Date:** 10 May 2026

---

## 1. Problem Decomposition

The core challenge is moving a user from *vague hiring intent* to a *grounded, catalog-only shortlist* through dialogue — without over-asking, hallucinating, or drifting out of scope.

I decomposed this into four sub-problems:
- **Retrieval:** Surface the right catalog items given noisy, conversational queries
- **Dialogue management:** Decide when to clarify vs. commit vs. refuse
- **Grounding:** Ensure every URL and name comes from the scraped catalog, not model memory
- **Robustness:** Handle edge cases — legal questions, missing tests, mid-stream refinements

---

## 2. Architecture

```
POST /chat (full history)
        │
        ▼
  Intent Detection          ← rule-based (compare / clarify / refuse / end)
        │
        ▼
  Query Builder             ← concatenate all user turns
        │
        ▼
  Hybrid Retrieval          ← BM25 (0.35) + FAISS semantic (0.65) + heuristic boosting
        │
        ▼
  LLM (Groq / LLaMA-3.3-70B) ← grounded on retrieved catalog context only
        │
        ▼
  URL Safety Filter         ← strip any rec whose URL isn't in retrieval pool
        │
        ▼
  FastAPI Response          ← { reply, recommendations[], end_of_conversation }
```

**Key design decision — retrieval-first, LLM-second:** The LLM sees only the top-15 retrieved catalog items as context. It cannot recommend anything outside that window. This eliminates hallucination structurally rather than relying on prompt instructions alone.

---

## 3. Catalog Ingestion

The provided catalog JSON was used directly (no re-scraping needed). Each item was enriched with:

- A `test_type` code mapped from the `keys` array: `K` (Knowledge), `P` (Personality), `A` (Ability), `B` (Biodata/SJT), `S` (Simulation), `C` (Competencies), `D` (Development)
- A composite `search_text` field concatenating name, description, job levels, keys, and duration — used for both embedding and BM25 indexing

**Index:** `sentence-transformers/all-MiniLM-L6-v2` → FAISS `IndexFlatIP` (cosine similarity via L2 normalization). BM25 via `rank-bm25`. Both indexes built at startup and cached.

---

## 4. Retrieval Design

**Why hybrid search?** Semantic search captures intent ("I need something for behavioral fit") but misses exact names. BM25 catches abbreviations ("OPQ", "GSA", "DSI") but misses paraphrase. Combining both at 65/35 weight outperformed either alone on the public traces.

**Query expansion:** Domain-specific trigger→synonym maps (e.g. `"gsa"` → `"global skills assessment global skills development report"`). Learned from trace patterns — contact center, safety, graduate, leadership, admin roles each need distinct expansion vocabularies.

**Heuristic boosting:** Score adjustments applied post-combination for known high-signal patterns:
- `"cxo"/"leadership"` → +0.30 for OPQ32r, +0.25 for OPQ Leadership Report
- `"contact center"` → +0.30 for SVAR, +0.28 for Contact Center Call Simulation
- `"safety"/"chemical"` → +0.30 for DSI, +0.28 for Safety & Dependability 8.0
- `"gsa"` query → +0.30 for Global Skills Assessment instrument

**Diversity filter:** Deduplicate by normalized base name to avoid returning multiple report variants of the same product when an instrument is more relevant.

---

## 5. Dialogue Management

Intent is classified rule-first (not LLM-first) for latency and reliability:

| Intent | Trigger | Action |
|--------|---------|--------|
| `refuse` | Prompt injection / off-topic keywords | Static refusal, no retrieval |
| `legal_refuse` | "legally required", "does this satisfy" | Specific legal disclaimer |
| `compare` | "difference between", "vs" + assessment keyword | Retrieve both items, LLM compares from catalog only |
| `end` | Completion phrases: "confirmed", "locking it in" | end_of_conversation: true |
| `clarify` | Vague first turn, missing role/seniority | LLM asks ONE focused question |
| `recommend` | Default | LLM recommends from retrieval pool |

**Clarification heuristics learned from traces:**
- Contact center → ask language first, then accent variant (US/UK/AU/IN)
- Leadership → ask selection vs. development
- Safety / admin / graduate roles → recommend directly (clear enough)
- After 3 user turns, always commit regardless of completeness

**Refinement:** Because the API is stateless and the full history is passed each call, refinements ("add X", "drop Y") are handled naturally — the LLM sees all prior turns and the current retrieval pool and updates accordingly. No separate state management required.

---

## 6. Prompt Design

The system prompt encodes: output format contract, recommendation priority (instruments before reports), per-intent instructions, refinement semantics ("drop X" = remove, never restart), comparison rules (catalog-only), and scope boundaries.

`temperature=0.15` for consistency. JSON-only output enforced via prompt + `json.loads()` with fallback to retrieval-based response on parse failure.

---

## 7. What Didn't Work

- **Pure LLM intent detection** was too slow (~3-4s) and occasionally misclassified legal questions as comparisons. Replaced with rule-based pre-filter.
- **Semantic-only retrieval** failed on abbreviations (GSA, DSI, SVAR) which are BM25's strength. Hybrid search fixed this.
- **Aggressive clarification** (asking both role and seniority even on clear queries) violated the "don't over-ask" expectation from traces. Fixed by role-specific skip logic.

---

## 8. Evaluation Approach

Tested against all 10 public traces by replaying each turn manually and checking:
- Schema compliance on every response
- Correct items appearing in final shortlist (Recall@10)
- Clarification triggered only on turns 1-2 for vague queries
- Refinements preserved prior items correctly
- Legal refusal triggered on HIPAA compliance question
- end_of_conversation=true on all "confirmed/thanks" turns

**Stack:** FastAPI · Groq (LLaMA-3.3-70B) · FAISS · BM25 · sentence-transformers · Render (deployment)

*AI tools used: Claude for code review and prompt iteration. All design decisions, retrieval architecture, and evaluation were my own.*
