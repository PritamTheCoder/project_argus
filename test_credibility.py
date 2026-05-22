import asyncio
import logging
import json
from src.graph.builder import build_graph

# Configure logging to just show verifier outputs clearly
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_source_awareness():
    print("Building argus graph...")
    graph = build_graph()
    
    # We ask a question that naturally returns a mix of academic papers and blogs
    query = "Comparison of solid-state vs lithium-ion battery degradation timelines 2024"
    
    initial_state = {
        "query": query,
        "research_plan": [],
        "search_queries": [],
        "scraped_data": [],
        "source_map": {},
        "structured_evidence": [],
        "verified_facts": [],
        "critique": "",
        "re_search_required": False,
        "report": "",
        "loop_count": 0,
        "active_node": "start",
        "error_log": []
    }

    print(f"\n--- Running Pipeline against: '{query}' ---")
    final_state = await graph.ainvoke(initial_state)
    
    print("\n\n================================================")
    print(" VERIFIER RESULTS & SOURCE CREDIBILITY REASONING")
    print("================================================\n")
    
    verified_facts = final_state.get("verified_facts", [])
    
    if not verified_facts:
         print("No facts were verified.")
         return
         
    for i, fact in enumerate(verified_facts):
         score = fact.get("credibility_score", "UNKNOWN")
         stype = fact.get("source_type", "UNKNOWN")
         conf = fact.get("confidence", "UNKNOWN")
         reason = fact.get("reasoning", "UNKNOWN")
         url = fact.get("source_url", "UNKNOWN")
         
         print(f"Fact {i+1}:")
         print(f"  Source URL: {url}")
         print(f"  Source Credibility: {score} ({stype})")
         print(f"  LLM Confidence: {conf}")
         print(f"  LLM Reasoning: {reason}")
         print("-" * 60)

    print("\n\n================================================")
    print(" GHOSTWRITER FINAL REPORT")
    print("================================================\n")
    report = final_state.get("report", "No report generated.")
    print(report)
    
    with open("report_out.txt", "w", encoding="utf-8") as f:
        f.write(report)

if __name__ == "__main__":
    asyncio.run(test_source_awareness())
