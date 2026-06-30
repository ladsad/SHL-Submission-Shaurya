---
title: SHL Assessment Recommender
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# SHL Assessment Recommender API

A stateless, conversational FastAPI service that recommends SHL assessments based on a provided product catalog. Built using a dual-provider LLM backend (Groq/Gemini), a custom hybrid RAG retriever, and strict JSON-schema enforcement for conversational UI integration.

## Architecture

1. **Hybrid Retriever**: Combines semantic embedding search (`all-MiniLM-L6-v2`) with structured boolean filters (e.g., job levels, languages) to ensure high-precision contextual matching against the 377-item SHL catalog.
2. **Two-Step Agent Pipeline**:
    - **Intent Extraction**: Parses conversational history into a structured schema (search terms, filters, clarification needs).
    - **Grounded Generation**: Injects the retrieved catalog context into a secondary prompt to generate a highly constrained JSON response containing exact URLs and conversational text.
3. **Multi-Provider Fallback**: Seamlessly falls back to Groq (`llama-3.1-8b-instant` or `llama-3.3-70b-versatile`) when Google's Gemini free-tier daily rate limits are exhausted.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Create a `.env` file in the root directory:
   ```env
   # Prefer Groq for high-rate limit free-tier access
   GROQ_API_KEY=your_groq_key_here
   GROQ_MODEL=llama-3.3-70b-versatile
   
   # Backup provider
   GEMINI_API_KEY=your_gemini_key_here
   GEMINI_MODEL=gemini-2.0-flash
   ```

## Running the API

Start the Uvicorn server:
```bash
python -m src.app
```
*Note: On first startup, the server will download the SentenceTransformer model and pre-compute embeddings for the 377-item catalog. This takes ~20 seconds.*

## Testing & Evaluation

**Run Smoke Tests** (verifies catalog data integrity and hybrid retrieval logic):
```bash
python -m pytest tests/test_smoke.py
```

**Run End-to-End Behavioral Probes** (tests the live API against 4 core conversational constraints: vague queries, specific needs, off-topic refusal, and context refinement):
```bash
python tests/test_e2e.py
```

**Run the Evaluation Suite (Recall@10)** (replays the 10 provided Markdown conversation traces against the API and scores the results):
```bash
python tests/evaluator.py
```
*Note: The evaluator makes roughly 40 sequential LLM calls. If using a free-tier API key, this script will encounter rate-limit sleeps (429 Too Many Requests), significantly extending the evaluation time.*

## Design History
Please see `TIMELINE.md` for a chronological log of design decisions, pivots, and findings during the development of this prototype.
