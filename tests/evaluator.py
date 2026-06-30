"""
SHL Assessment Recommender — Evaluator Script

Replays the 10 conversation traces against the API and calculates Recall@10
based on the expected recommendations found in the markdown tables.
"""

import re
import sys
import time
from pathlib import Path

import httpx

BASE_URL = "http://localhost:8000"
TRACES_DIR = Path(__file__).parent.parent / "Documentation" / "sample_conversations" / "GenAI_SampleConversations"


def parse_trace(filepath: Path) -> tuple[list[str], list[str]]:
    """Parse a markdown trace file into a list of user messages and expected URLs."""
    content = filepath.read_text(encoding="utf-8")

    # Extract all user queries
    user_queries = []
    blocks = content.split("**User**")
    for block in blocks[1:]:
        match = re.search(r"> (.*?)\n", block, re.DOTALL)
        if match:
            user_queries.append(match.group(1).strip().replace("\n", " "))

    # Extract all expected URLs from markdown tables
    expected_urls = []
    # Find all links starting with https://www.shl.com/
    urls = re.findall(r"<(https://www\.shl\.com/[^>]+)>", content)
    # Deduplicate while preserving order
    seen = set()
    for u in urls:
        if u not in seen:
            expected_urls.append(u)
            seen.add(u)

    return user_queries, expected_urls


def evaluate_trace(filename: str, user_queries: list[str], expected_urls: list[str]) -> float:
    """Replay conversation and compute Recall@10 based on all unique recs."""
    print(f"\n--- Evaluating {filename} ---")
    messages = []
    agent_urls_all = set()
    
    for i, query in enumerate(user_queries):
        messages.append({"role": "user", "content": query})
        
        try:
            r = httpx.post(f"{BASE_URL}/chat", json={"messages": messages}, timeout=150.0)
            r.raise_for_status()
            resp = r.json()
            
            # Accumulate recommendations from this turn
            recs = resp.get("recommendations", [])
            print(f"  [Turn {i+1}] Query: {query}")
            print(f"  [Turn {i+1}] Recs: {len(recs)}")
            for rec in recs:
                print(f"    - {rec.get('name')}")
                if rec.get("url"):
                    agent_urls_all.add(rec["url"])
                    
            messages.append({"role": "assistant", "content": r.text})
            
            if i == len(user_queries) - 1:
                agent_urls = list(agent_urls_all)
                print(f"Agent returned {len(agent_urls)} unique recommendations across all turns.")
                print(f"Expected {len(expected_urls)} recommendations.")
                
                hits = 0
                for expected_url in expected_urls:
                    if expected_url in agent_urls:
                        hits += 1
                    else:
                        expected_slug = expected_url.strip("/").split("/")[-1]
                        if any(expected_slug in a_url for a_url in agent_urls):
                            hits += 1
                        else:
                            print(f"  [MISS] {expected_slug}")
                
                recall = hits / len(expected_urls) if expected_urls else 1.0
                print(f"Recall: {recall:.2f} ({hits}/{len(expected_urls)})")
                return recall
                
        except Exception as e:
            print(f"Error calling API on turn {i+1}: {e}")
            return 0.0
            
    return 0.0


def main():
    try:
        r = httpx.get(f"{BASE_URL}/health", timeout=5.0)
        assert r.json()["status"] == "ok"
    except Exception:
        print("Server not running. Start with: python -m src.app")
        sys.exit(1)

    trace_files = sorted(TRACES_DIR.glob("C*.md"), key=lambda x: int(re.search(r"C(\d+)", x.name).group(1)))
    
    if not trace_files:
        print(f"No trace files found in {TRACES_DIR}")
        sys.exit(1)

    print(f"Found {len(trace_files)} traces. Starting evaluation...")
    
    total_recall = 0.0
    t0 = time.time()
    
    for filepath in trace_files:
        user_queries, expected_urls = parse_trace(filepath)
        recall = evaluate_trace(filepath.name, user_queries, expected_urls)
        total_recall += recall
        
    avg_recall = total_recall / len(trace_files)
    print("\n=========================================")
    print(f"Evaluation Complete in {time.time()-t0:.1f}s")
    print(f"Average Recall@10: {avg_recall:.2f}")
    print("=========================================")


if __name__ == "__main__":
    main()
