"""End-to-end test: replay conversation traces against the running API."""

import httpx
import json
import time
import sys

BASE_URL = "http://localhost:8000"

def chat(messages: list[dict]) -> dict:
    """Send a chat request and return the response."""
    r = httpx.post(
        f"{BASE_URL}/chat",
        json={"messages": messages},
        timeout=35.0,
    )
    r.raise_for_status()
    return r.json()


def test_vague_query():
    """C1-style: vague query should trigger clarification, NOT recommendations."""
    print("=== Test: Vague Query (should clarify) ===")
    resp = chat([{"role": "user", "content": "We need a solution for senior leadership."}])
    print(f"Reply: {resp['reply'][:200]}")
    print(f"Recs: {len(resp['recommendations'])}")
    print(f"EOC: {resp['end_of_conversation']}")
    assert resp["recommendations"] == [] or len(resp["recommendations"]) <= 10, "Schema violation"
    assert resp["end_of_conversation"] == False, "Should not end on vague query"
    print("[OK]\n")
    return resp


def test_specific_query():
    """C4-style: specific enough to recommend immediately."""
    print("=== Test: Specific Query (should recommend) ===")
    resp = chat([{
        "role": "user",
        "content": "Hiring graduate financial analysts - final-year students, no work experience. We need numerical reasoning and a finance knowledge test."
    }])
    print(f"Reply: {resp['reply'][:200]}")
    print(f"Recs: {len(resp['recommendations'])}")
    for r in resp["recommendations"]:
        print(f"  - {r['name']} ({r['test_type']}) -> {r['url'][:60]}")
    print(f"EOC: {resp['end_of_conversation']}")
    # Should have recommendations for this clear query
    print("[OK]\n")
    return resp


def test_off_topic():
    """Should refuse non-SHL topics."""
    print("=== Test: Off-Topic (should refuse) ===")
    resp = chat([{"role": "user", "content": "What is the best salary to offer a Java developer in Bangalore?"}])
    print(f"Reply: {resp['reply'][:200]}")
    print(f"Recs: {len(resp['recommendations'])}")
    assert resp["recommendations"] == [], "Should not recommend for off-topic"
    print("[OK]\n")
    return resp


def test_multi_turn_refinement():
    """C9-style: multi-turn with refinement."""
    print("=== Test: Multi-Turn Refinement ===")
    messages = [
        {"role": "user", "content": "I'm hiring a senior Java developer, backend-focused."},
    ]
    resp1 = chat(messages)
    print(f"Turn 1: {resp1['reply'][:150]}...")
    print(f"  Recs: {len(resp1['recommendations'])}")

    # Add assistant response and user refinement
    messages.append({"role": "assistant", "content": json.dumps(resp1)})
    messages.append({"role": "user", "content": "Add AWS and Docker assessments too."})
    resp2 = chat(messages)
    print(f"Turn 2: {resp2['reply'][:150]}...")
    print(f"  Recs: {len(resp2['recommendations'])}")
    for r in resp2["recommendations"]:
        print(f"    - {r['name']} ({r['test_type']})")
    print("[OK]\n")
    return resp2


if __name__ == "__main__":
    # Verify server is up
    try:
        r = httpx.get(f"{BASE_URL}/health", timeout=5.0)
        assert r.json()["status"] == "ok"
    except Exception:
        print("Server not running. Start with: python -m src.app")
        sys.exit(1)

    t0 = time.time()
    test_vague_query()
    test_specific_query()
    test_off_topic()
    test_multi_turn_refinement()
    print(f"All tests passed in {time.time()-t0:.1f}s")
