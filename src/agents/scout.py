"""
Project Argus - Scout Agent

Role: Research Execution
Responsibility: Iterates through the plan, calls the search/scrape tools,
and builds the raw data payload.
"""

import logging
import asyncio
from src.schema.state import AgentState
from src.tools.scout import run_scout

logger = logging.getLogger(__name__)


async def scout_node(state: AgentState) -> dict:
    """
    Execute the research plan by searching and scraping for each query.
    
    Args:
        state: AgentState containing `plan` (list of strings).
        
    Returns:
        dict: Updates `scraped_data` and `source_map`.
    """
    logger.info("Scout: Starting research execution...")
    
    queries = state["plan"]
    all_scraped_data = []
    source_map = state.get("source_map", {}).copy()
    
    # Determine the next available source ID (e.g. 1, 2, 3...)
    # We parse keys like "[1]" to find the max integer.
    current_ids = [
        int(k.strip("[]")) 
        for k in source_map.keys() 
        if k.startswith("[") and k.endswith("]") and k.strip("[]").isdigit()
    ]
    next_id = max(current_ids) + 1 if current_ids else 1
    
    # Execute searches concurrently for speed, but sequentially per query to avoid 
    # hitting rate limits too hard or overwhelming the scraper.
    # For robust production, we might want to parallelize this further, 
    # but let's go linear-by-query, parallel-within-query (run_scout does this) for now.
    
    for q in queries:
        logger.info(f"Scout: Processing query '{q}'")
        results = await run_scout(q)
        
        # Process successful results
        for res in results:
            if not res["success"] or not res["content"]:
                continue
                
            # Check if URL is already in our map to avoid duplicates
            existing_id = None
            for sid, sdata in source_map.items():
                if sdata["url"] == res["url"]:
                    existing_id = sid
                    break
            
            if existing_id:
                source_id = existing_id
            else:
                source_id = f"[{next_id}]"
                source_map[source_id] = {
                    "url": res["url"],
                    "snippet": res["content"][:200] + "..." # Preview for UI/Citations
                }
                next_id += 1
            
            # Append to scraped data payload
            all_scraped_data.append({
                "source_id": source_id,
                "url": res["url"],
                "content": res["content"],
                "query": q
            })
            
    logger.info(f"Scout: Collected {len(all_scraped_data)} valid pages.")
    
    return {
        "scraped_data": all_scraped_data,
        "source_map": source_map,
        "active_node": "scout"
    }
