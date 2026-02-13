"""
Demo: LangExtract Refiner

Quick test to verify extract_facts() works end-to-end.
Requires GOOGLE_API_KEY in .env.

Run:  python demo_refiner.py
"""

import json
import logging
from src.tools.refiner import extract_facts

logging.basicConfig(level=logging.INFO)

# ── Sample text (mimics scraped Markdown) ────────────────────────────────────
SAMPLE_TEXT = """
# Solid-State Batteries in 2026

Solid-state batteries have reached a major milestone. QuantumScape announced
on January 15, 2026 that their prototype cell achieved an energy density of
500 Wh/kg, a 40% improvement over conventional lithium-ion cells.

Toyota plans to begin limited production by Q3 2026, targeting an initial
output of 10,000 units per month. The estimated cost per kWh has dropped to
$80, down from $120 in 2024.

A solid-state battery replaces the liquid electrolyte found in traditional
lithium-ion batteries with a solid material, reducing fire risk and enabling
faster charging times of approximately 15 minutes to 80% capacity.
"""

# ── Schema: what we want to extract ──────────────────────────────────────────
SCHEMA = {
    "dates": "date or time reference",
    "metrics": "numeric value or measurement",
    "definitions": "concept definition or explanation",
}


def main():
    print("=" * 60)
    print("  Project Argus — Refiner Demo (LangExtract)")
    print("=" * 60)
    print()

    result = extract_facts(SAMPLE_TEXT, SCHEMA)

    print(f"\n[OK] Extracted {len(result['facts'])} facts:\n")
    print(json.dumps(result["facts"], indent=2, ensure_ascii=False))

    # Show source_span verification for the first few facts
    print("\n── Source Span Verification ──")
    for fact in result["facts"][:5]:
        span = fact["source_span"]
        start, end = span["start"], span["end"]
        if start is not None and end is not None:
            snippet = SAMPLE_TEXT[start:end]
            print(f"  [{fact['class']}] \"{fact['text']}\"")
            print(f"    span [{start}:{end}] → \"{snippet}\"")
            print()


if __name__ == "__main__":
    main()
