"""
SHL Assessment Recommender — Catalog loader and processor.

Loads the SHL product catalog JSON, derives test_type codes from the `keys`
field, and exposes the processed catalog as a list of Assessment dataclass
instances.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ── Key → Test-type code mapping (derived from sample conversation traces) ──
_KEY_TO_CODE: dict[str, str] = {
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Ability & Aptitude": "A",
    "Competencies": "C",
    "Development & 360": "D",
    "Biodata & Situational Judgment": "B",
    "Simulations": "S",
    "Assessment Exercises": "E",
}

_CATALOG_URL = (
    "https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json"
)


@dataclass
class Assessment:
    """Single SHL assessment product."""

    entity_id: str
    name: str
    url: str
    description: str
    test_type: str  # e.g. "K", "P", "K,S"
    keys: list[str]
    job_levels: list[str]
    languages: list[str]
    duration: str
    remote: str
    adaptive: str

    # pre-built text for embedding
    embedding_text: str = field(default="", repr=False)

    def to_recommendation(self) -> dict:
        """Return the schema-compliant recommendation dict."""
        return {
            "name": self.name,
            "url": self.url,
            "test_type": self.test_type,
        }

    def to_context_str(self) -> str:
        """Rich string for injecting into LLM context."""
        langs = self.languages[:5]
        lang_str = ", ".join(langs)
        if len(self.languages) > 5:
            lang_str += f" (+{len(self.languages) - 5} more)"
        return (
            f"[{self.entity_id}] {self.name}\n"
            f"  Test Type: {self.test_type} | Keys: {', '.join(self.keys)}\n"
            f"  Job Levels: {', '.join(self.job_levels)}\n"
            f"  Duration: {self.duration or 'N/A'} | Remote: {self.remote} | Adaptive: {self.adaptive}\n"
            f"  Languages: {lang_str}\n"
            f"  URL: {self.url}\n"
            f"  Description: {self.description[:300]}"
        )


def _derive_test_type(keys: list[str]) -> str:
    """Map keys list → comma-separated test-type codes."""
    codes = []
    for k in keys:
        code = _KEY_TO_CODE.get(k)
        if code and code not in codes:
            codes.append(code)
    return ",".join(codes) if codes else "K"  # default to K if unknown


def _build_embedding_text(a: dict) -> str:
    """Build a composite text string for embedding."""
    parts = [
        a.get("name", ""),
        a.get("description", ""),
        ", ".join(a.get("keys", [])),
        ", ".join(a.get("job_levels", [])),
    ]
    return " | ".join(p for p in parts if p)


def _parse_duration(raw: str) -> str:
    """Normalise the duration field."""
    if not raw:
        return ""
    # Extract minutes if present
    m = re.search(r"(\d+)\s*minutes?", raw, re.IGNORECASE)
    if m:
        return f"{m.group(1)} minutes"
    return raw.strip()


def load_catalog(path: Optional[str | Path] = None) -> list[Assessment]:
    """Load and process the SHL product catalog.

    If *path* is ``None`` the catalog is fetched from the canonical URL.
    """
    if path and Path(path).exists():
        raw = Path(path).read_bytes()
    else:
        import urllib.request

        raw = urllib.request.urlopen(_CATALOG_URL).read()

    entries: list[dict] = json.loads(raw, strict=False)

    assessments: list[Assessment] = []
    seen_ids: set[str] = set()

    for e in entries:
        eid = e.get("entity_id", "")
        if not eid or eid in seen_ids:
            continue
        seen_ids.add(eid)

        # Skip entries with bad status
        if e.get("status") != "ok":
            continue

        keys = e.get("keys", [])
        test_type = _derive_test_type(keys)

        a = Assessment(
            entity_id=eid,
            name=e.get("name", ""),
            url=e.get("link", ""),
            description=e.get("description", ""),
            test_type=test_type,
            keys=keys,
            job_levels=e.get("job_levels", []),
            languages=e.get("languages", []),
            duration=_parse_duration(e.get("duration", "")),
            remote=e.get("remote", ""),
            adaptive=e.get("adaptive", ""),
            embedding_text=_build_embedding_text(e),
        )
        assessments.append(a)

    return assessments


# ── Quick sanity check ──────────────────────────────────────────────────────
if __name__ == "__main__":
    catalog = load_catalog(Path(__file__).parent.parent / "data" / "shl_product_catalog.json")
    print(f"Loaded {len(catalog)} assessments")
    for a in catalog[:3]:
        print(f"  {a.name} -> test_type={a.test_type}, keys={a.keys}")
