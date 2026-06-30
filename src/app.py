"""
SHL Assessment Recommender — FastAPI application.

Exposes:
  GET  /health  → {"status": "ok"}
  POST /chat    → Stateless conversational endpoint
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv()

from src.agent import Agent
from src.catalog import load_catalog
from src.retriever import Retriever

logger = logging.getLogger("shl_recommender")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ── Global state ─────────────────────────────────────────────────────────────
_agent: Optional[Agent] = None

CATALOG_PATH = Path(__file__).parent.parent / "data" / "shl_product_catalog.json"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load catalog, build retriever and agent."""
    global _agent
    t0 = time.time()
    logger.info("Loading SHL catalog...")
    catalog = load_catalog(CATALOG_PATH)
    logger.info(f"Loaded {len(catalog)} assessments in {time.time()-t0:.1f}s")

    t1 = time.time()
    logger.info("Building retriever (this may take a moment on first run)...")
    retriever = Retriever(catalog)
    # Pre-warm embeddings
    retriever._ensure_embeddings()
    logger.info(f"Retriever ready in {time.time()-t1:.1f}s")

    _agent = Agent(retriever)
    logger.info(f"Agent ready. Total startup: {time.time()-t0:.1f}s")

    yield

    logger.info("Shutting down.")


# ── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="SHL Assessment Recommender",
    description="Conversational agent for recommending SHL Individual Test Solutions",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ──────────────────────────────────────────────────────────────────

class Message(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message text")


class ChatRequest(BaseModel):
    messages: list[Message] = Field(..., description="Full conversation history")


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation]
    end_of_conversation: bool


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent not ready")

    if not request.messages:
        raise HTTPException(status_code=400, detail="messages cannot be empty")

    # Validate message roles
    for msg in request.messages:
        if msg.role not in ("user", "assistant"):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid role: {msg.role}. Must be 'user' or 'assistant'.",
            )

    # Convert to plain dicts
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    t0 = time.time()
    try:
        result = _agent.process(messages)
    except Exception as e:
        logger.error(f"Agent error: {e}", exc_info=True)
        # Return a safe fallback rather than crashing
        result = {
            "reply": "I encountered an issue processing your request. Could you rephrase your question about SHL assessments?",
            "recommendations": [],
            "end_of_conversation": False,
        }

    elapsed = time.time() - t0
    logger.info(
        f"Chat processed in {elapsed:.2f}s | "
        f"turns={len(messages)} | "
        f"recs={len(result.get('recommendations', []))} | "
        f"eoc={result.get('end_of_conversation', False)}"
    )

    return ChatResponse(
        reply=result["reply"],
        recommendations=[Recommendation(**r) for r in result["recommendations"]],
        end_of_conversation=result["end_of_conversation"],
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=True)
