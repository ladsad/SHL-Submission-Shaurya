"""Quick smoke test for catalog + retriever — no API key needed."""

import sys
import time
from pathlib import Path

# Run from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.catalog import load_catalog
from src.retriever import Retriever

DATA_PATH = Path(__file__).parent / "data" / "shl_product_catalog.json"


def test_catalog():
    catalog = load_catalog(DATA_PATH)
    print(f"[OK] Loaded {len(catalog)} assessments")

    # Check test_type derivation
    for a in catalog:
        assert a.test_type, f"Missing test_type for {a.name}"
        assert a.url.startswith("https://www.shl.com/"), f"Bad URL for {a.name}: {a.url}"

    # Check known assessments from traces
    names = {a.name for a in catalog}
    expected = [
        "Occupational Personality Questionnaire OPQ32r",
        "SHL Verify Interactive G+",
        "Global Skills Assessment",
    ]
    for name in expected:
        assert name in names, f"Expected assessment not found: {name}"

    print(f"[OK] All assessments have valid test_type and URLs")
    print(f"[OK] Known assessments found in catalog")


def test_retriever():
    catalog = load_catalog(DATA_PATH)
    retriever = Retriever(catalog)

    t0 = time.time()
    retriever._ensure_embeddings()
    print(f"[OK] Embeddings built in {time.time()-t0:.1f}s")

    # Test: "Java developer" should return Java-related assessments
    results = retriever.retrieve("Java developer mid-level")
    names = [a.name for a in results]
    java_hits = [n for n in names if "java" in n.lower()]
    print(f"[OK] 'Java developer' query -> {len(results)} results, {len(java_hits)} Java hits")
    for a in results[:5]:
        print(f"     {a.name} (test_type={a.test_type})")

    # Test: "personality assessment senior leadership"
    results = retriever.retrieve("personality assessment senior leadership", job_levels=["Director", "Executive"])
    names = [a.name for a in results]
    opq_hits = [n for n in names if "opq" in n.lower()]
    print(f"[OK] 'personality senior leadership' -> {len(results)} results, {len(opq_hits)} OPQ hits")
    for a in results[:5]:
        print(f"     {a.name} (test_type={a.test_type})")

    # Test: lookup by name
    a = retriever.lookup_by_name("Occupational Personality Questionnaire OPQ32r")
    assert a is not None, "Failed to look up OPQ32r"
    print(f"[OK] Lookup OPQ32r: {a.name}, URL={a.url[:60]}...")

    # Test: lookup by partial name
    a = retriever.lookup_by_name("OPQ32r")
    assert a is not None, "Failed to fuzzy-lookup OPQ32r"
    print(f"[OK] Fuzzy lookup 'OPQ32r': {a.name}")

    # Test: "contact center simulation"
    results = retriever.retrieve("contact center call simulation entry level")
    names = [a.name for a in results]
    print(f"[OK] 'contact center simulation' -> top 3:")
    for a in results[:3]:
        print(f"     {a.name} (test_type={a.test_type})")


if __name__ == "__main__":
    test_catalog()
    print()
    test_retriever()
    print("\nAll smoke tests passed!")
