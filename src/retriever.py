"""
SHL Assessment Recommender — Retrieval engine.

Hybrid retriever combining:
1. Semantic search via sentence-transformers embeddings + cosine similarity
2. Structured filtering on job_levels, keys, languages

The catalog is small (377 items) so we keep everything in-memory with numpy
rather than adding a vector DB dependency.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from src.catalog import Assessment

# ── Configuration ───────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CACHE_DIR = Path(__file__).parent.parent / "embeddings_cache"
TOP_K_SEMANTIC = 30  # candidates from semantic search
TOP_K_FINAL = 15     # returned to the agent after re-ranking


class Retriever:
    """Hybrid retriever over the SHL catalog."""

    def __init__(self, catalog: list[Assessment]) -> None:
        self.catalog = catalog
        self._name_index: dict[str, Assessment] = {
            a.name.lower(): a for a in catalog
        }
        self._id_index: dict[str, Assessment] = {a.entity_id: a for a in catalog}
        self._embeddings: Optional[NDArray] = None
        self._model = None

    # ── Lazy-load the embedding model ────────────────────────────────────

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(EMBEDDING_MODEL)

    def _ensure_embeddings(self) -> None:
        if self._embeddings is not None:
            return

        self._ensure_model()

        # Compute a cache key from catalog content
        content_hash = hashlib.md5(
            json.dumps([a.embedding_text for a in self.catalog]).encode()
        ).hexdigest()[:12]
        cache_path = CACHE_DIR / f"embeddings_{content_hash}.pkl"

        if cache_path.exists():
            self._embeddings = pickle.loads(cache_path.read_bytes())
            return
            
        # Fallback for OS newline differences (Windows vs Linux hash mismatch)
        if CACHE_DIR.exists():
            pkl_files = list(CACHE_DIR.glob("*.pkl"))
            if pkl_files:
                self._embeddings = pickle.loads(pkl_files[0].read_bytes())
                return

        texts = [a.embedding_text for a in self.catalog]
        self._embeddings = self._model.encode(texts, normalize_embeddings=True)

        CACHE_DIR.mkdir(exist_ok=True)
        cache_path.write_bytes(pickle.dumps(self._embeddings))

    # ── Search methods ───────────────────────────────────────────────────

    def semantic_search(
        self,
        query: str,
        top_k: int = TOP_K_SEMANTIC,
    ) -> list[tuple[Assessment, float]]:
        """Return top-k assessments by cosine similarity to the query."""
        self._ensure_embeddings()

        q_emb = self._model.encode([query], normalize_embeddings=True)
        scores = (self._embeddings @ q_emb.T).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [(self.catalog[i], float(scores[i])) for i in top_indices]

    def filter_by_keys(
        self, assessments: list[Assessment], required_keys: list[str]
    ) -> list[Assessment]:
        """Keep assessments whose `keys` overlap with required_keys."""
        req = {k.lower() for k in required_keys}
        return [a for a in assessments if req & {k.lower() for k in a.keys}]

    def filter_by_job_levels(
        self, assessments: list[Assessment], levels: list[str]
    ) -> list[Assessment]:
        """Keep assessments whose job_levels overlap with required levels."""
        req = {l.lower() for l in levels}
        return [a for a in assessments if req & {l.lower() for l in a.job_levels}]

    def filter_by_language(
        self, assessments: list[Assessment], language: str
    ) -> list[Assessment]:
        """Keep assessments available in the given language."""
        lang = language.lower()
        return [
            a for a in assessments
            if any(lang in l.lower() for l in a.languages) or not a.languages
        ]

    def lookup_by_name(self, name: str) -> Optional[Assessment]:
        """Exact or fuzzy lookup by assessment name."""
        key = name.lower().strip()
        if key in self._name_index:
            return self._name_index[key]
        # Fuzzy: substring match
        for n, a in self._name_index.items():
            if key in n or n in key:
                return a
        return None

    def lookup_by_id(self, entity_id: str) -> Optional[Assessment]:
        return self._id_index.get(entity_id)

    def retrieve(
        self,
        query: str,
        job_levels: Optional[list[str]] = None,
        key_categories: Optional[list[str]] = None,
        language: Optional[str] = None,
        top_k: int = TOP_K_FINAL,
    ) -> list[Assessment]:
        """Full hybrid retrieval pipeline.

        1. Semantic search for broad recall
        2. Optional structured filters to boost precision
        3. Merge: filtered results first, then remaining semantic hits
        """
        semantic_results = self.semantic_search(query, top_k=TOP_K_SEMANTIC)
        candidates = [a for a, _ in semantic_results]

        # Apply structured filters (intersection, not elimination)
        filtered = candidates[:]
        if key_categories:
            key_filtered = self.filter_by_keys(candidates, key_categories)
            if key_filtered:
                filtered = key_filtered
        if job_levels:
            level_filtered = self.filter_by_job_levels(filtered, job_levels)
            if level_filtered:
                filtered = level_filtered
        if language:
            lang_filtered = self.filter_by_language(filtered, language)
            if lang_filtered:
                filtered = lang_filtered

        # Merge: filtered first, then fill from semantic
        seen = {a.entity_id for a in filtered}
        merged = filtered[:]
        for a in candidates:
            if a.entity_id not in seen:
                merged.append(a)
                seen.add(a.entity_id)
            if len(merged) >= top_k:
                break

        return merged[:top_k]

    def get_all(self) -> list[Assessment]:
        """Return the full catalog."""
        return self.catalog
