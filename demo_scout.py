import asyncio
import json
import logging
from src.tools.scout import run_scout

# log
logging.basicConfig(level=logging.INFO)

async def main():
    print("Running scout for query: 'AI Agent'...")
    
    # Run the scout tool
    results = await run_scout("AI Agent", max_results=2)
    
    # Save to JSON file
    output_file = "sample_output.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print(f"\nSuccess! Results saved to {output_file}")
    print(f"Found {len(results)} results.")

if __name__ == "__main__":
    asyncio.run(main())
