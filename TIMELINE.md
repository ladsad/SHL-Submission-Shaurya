# SHL Assessment Recommender — Design Timeline

This document tracks every significant design decision, what was tried, what worked, and what didn't.

---

## Phase 0 — Data Acquisition & Exploration
**Date:** 2026-06-30

### Actions
1. Downloaded SHL product catalog JSON from canonical URL (377 assessments)
2. Downloaded 10 sample conversation traces (C1–C10) from provided zip
3. Analyzed catalog schema: `entity_id`, `name`, `link`, `description`, `keys`, `job_levels`, `languages`, `duration`, `remote`, `adaptive`
4. Read all 10 conversation traces to understand expected agent behavior

### Key Findings
- **No `test_type` field in JSON** — all 377 entries return `None`. The sample conversations use codes (K, P, A, C, D, B, S). Must be derived from the `keys` field.
- **8 key categories** map to test_type codes: Knowledge & Skills → K, Personality & Behavior → P, Ability & Aptitude → A, Competencies → C, Development & 360 → D, Biodata & Situational Judgment → B, Simulations → S, Assessment Exercises → E
- **10 job levels** in the catalog: Director, Entry-Level, Executive, Front Line Manager, General Population, Graduate, Manager, Mid-Professional, Professional Individual Contributor, Supervisor
- **Multi-key assessments** exist (e.g., Global Skills Assessment has keys [Competencies, Knowledge & Skills] → test_type "C,K")

### Conversation Trace Patterns Observed
| Trace | Scenario | Turns | Recs | Key Behaviors |
|-------|----------|-------|------|---------------|
| C1 | Senior leadership (vague) | 4 | 3 | Clarify → clarify → recommend → confirm |
| C2 | Senior Rust engineer (specific) | 3 | 5 | Partial recommend + no exact match → add cognitive → confirm |
| C3 | Entry-level contact centre (500 agents) | 5 | 4 | Clarify language → clarify accent → recommend → compare → confirm |
| C4 | Graduate financial analysts | 3 | 5 | Immediate recommend → refine (add SJT) → confirm |
| C5 | Sales org re-skilling | 3 | 5 | Immediate recommend → compare OPQ vs OPQ MQ → confirm |
| C6 | Plant operators (safety) | 3 | 2-3 | Recommend → compare DSI vs 8.0 → refine → confirm |
| C7 | Bilingual healthcare admin (Spanish) | 4 | 5 | Handle language constraint → hybrid battery → refuse legal Q → confirm |
| C8 | Admin assistants (Excel/Word) | 3 | 3-5 | Quick recommend → refine (add simulations) → confirm |
| C9 | Senior full-stack engineer (JD) | 7 | 7 | Clarify backend vs full-stack → clarify seniority → recommend → refine (swap items) → compare levels → discuss redundancy → confirm |
| C10 | Graduate management trainees | 3 | 2-3 | Immediate recommend → pushback on OPQ → refine (drop item) → confirm |

### Design Decisions
- **Test type derivation**: Map `keys` → codes. Multi-key → comma-separated. Default to "K" if unknown.
- **LLM choice**: Gemini 2.0 Flash — free tier, fast, good instruction following, native JSON mode
- **Retrieval**: Hybrid semantic + structured filtering. 377 items is small enough for in-memory numpy.
- **Framework**: Raw Gemini SDK — simpler than LangChain/LangGraph, easier to debug and defend in interview
- **Embedding model**: `all-MiniLM-L6-v2` — local, fast, no API dependency for embeddings

---

## Phase 1 — Project Scaffold & Core Architecture
**Date:** 2026-06-30

### Project Structure
```
SHL_Assignment/
├── src/
│   ├── __init__.py
│   ├── catalog.py      # Catalog loader, Assessment dataclass, test_type derivation
│   ├── retriever.py    # Hybrid retriever (semantic + structured filtering)
│   ├── agent.py        # Conversational agent (2-call LLM pipeline)
│   └── app.py          # FastAPI endpoints (GET /health, POST /chat)
├── data/
│   └── shl_product_catalog.json
├── tests/
│   └── __init__.py
├── Documentation/
│   ├── SHL_AI_Intern_Assignment.pdf
│   └── sample_conversations/
├── requirements.txt
├── .env.example
├── .gitignore
└── TIMELINE.md
```

### Architecture Design
```
POST /chat → Intent Analyzer (LLM call #1) → Hybrid Retrieval → Response Generator (LLM call #2) → Schema Validator → JSON Response
```

**Two-call LLM pipeline rationale:**
- Call #1 (intent analysis): Low-temperature structured extraction — determines search query, required filters, and conversation intent (clarify/recommend/refine/compare/refuse)
- Call #2 (response generation): RAG-augmented response with retrieved catalog context injected into system prompt. Higher temperature for natural conversation.

**Why not single-call?** Separating intent from response allows the retriever to do its job between calls. A single call would require the LLM to both decide what to retrieve AND generate the response, leading to hallucination risk.

### What Didn't Work
- (Nothing yet — first build iteration)

---

## Phase 2 — Testing & Evaluation
**Date:** 2026-06-30

### Actions & Findings
1. Built `test_smoke.py` — verified catalog loads and hybrid retriever returns expected results (Java for Java queries, OPQ for leadership, Contact Center sim for sim queries).
2. Encountered `429 RESOURCE_EXHAUSTED` with Gemini API free tier (daily limit 0).
3. **Pivoted backend:** Refactored `agent.py` to support `GROQ_API_KEY` alongside `GEMINI_API_KEY`.
4. Using Groq's `llama-3.3-70b-versatile` as the active LLM fallback.
5. Built `test_e2e.py` and successfully ran 4 core behavior probes against the active API:
    - Vague query clarification (no turn-1 recs)
    - Specific query (immediate recommendation)
    - Off-topic refusal (no recommendations)
    - Multi-turn refinement (maintaining context and updating shortlist)

### Planned Next Steps
- [ ] Compute Recall@10 against all 10 public traces
- [ ] Latency benchmarking (must be < 30s per call)

---

## Phase 3 — Hardening & Evaluation Suite
**Date:** 2026-06-30

### Actions & Findings
1. Built `tests/evaluator.py` to automate Recall@10 computation across all 10 conversation traces.
2. The evaluator accurately parses multi-turn markdown traces, accumulates unique recommendations across the conversation, and scores them against the ground-truth tables in the traces.
3. Due to Groq/Gemini free-tier rate limits (both Tokens-Per-Minute and Requests-Per-Day), a full evaluation of all 10 traces (~40 LLM calls) currently encounters forced API sleeps.
4. Finalized project for handoff: stateless architecture complete, dual-provider support active, testing suite operational.

### Final Deliverables
- **Data Pipeline:** `src/catalog.py` & `src/retriever.py` (Hybrid Search)
- **Agent Logic:** `src/agent.py` (Schema-enforced intent + RAG generation)
- **API Server:** `src/app.py` (FastAPI stateless endpoint)
- **Evaluation Suite:** `tests/test_e2e.py` & `tests/evaluator.py`
- **Documentation:** `README.md` & `TIMELINE.md`
