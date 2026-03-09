"""
Project Argus - Scout Agent

Role: Research Execution
Responsibility: Iterates through the plan, calls the search/scrape tools,
and builds the raw data payload.
"""

import logging
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
        
        # We need to extract the search+scrape steps from run_scout so we can intercept 
        # the URLs before scraping to apply the Global Seen-Set.
        from src.tools.search import search_ddg
        from src.tools.scraper import scrape_urls
        
        # 1. Search
        urls = await search_ddg(q)
        if not urls:
            continue
            
        # 2. Apply Global Seen-Set Deduplication
        seen_urls = {v["url"] for v in source_map.values()}
        new_urls = [u for u in urls if u not in seen_urls]
        
        if not new_urls:
            logger.info(f"Scout: All {len(urls)} URLs for query '{q}' were already seen.")
            continue
            
        logger.info(f"Scout: Found {len(urls)} URLs, {len(new_urls)} are new. Scraping...")
        
        # 3. Scrape ONLY the new URLs
        results = await scrape_urls(new_urls, query=q)
        
        # Process successful results
        for res in results:
            if not res["success"] or not res["content"]:
                continue
                
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
