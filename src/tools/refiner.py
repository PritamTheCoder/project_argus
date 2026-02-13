"""
Project Argus - LangExtract Refiner Tool

Extracts structured facts from raw Markdown using LangExtract.
Every extracted fact includes a source_span (character offset) so
we can trace it back to the original text.
"""

import logging
import langextract as lx
from src.config import GOOGLE_API_KEY, REFINER_MODEL

try:
    from absl import logging as absl_logging
except ImportError:
    absl_logging = None
    
logger = logging.getLogger(__name__)


def _build_prompt(schema: dict) -> str:
    """
    Turn a schema dict into a clear extraction prompt.

    Example schema: {"dates": "date", "metrics": "numeric value"}
    → "Extract the following entities ... dates, metrics ..."
    """
    parts = [f"{key} ({desc})" for key, desc in schema.items()]
    joined = ", ".join(parts)

    return (
        f"Extract the following entities from the text: {joined}. "
        "Use exact text for extractions. Do not paraphrase or overlap entities. "
        "Provide meaningful attributes for each entity to add context."
    )


def _build_examples(schema: dict) -> list:
    """
    Build a minimal few-shot example from the schema so LangExtract
    knows which extraction_class values and attribute shapes to expect.
    """
    # Create one dummy extraction per class so the model learns the schema
    sample_extractions = []
    
    for cls, desc in schema.items():
        
        sample_extractions.append(
            lx.data.Extraction(
                extraction_class=cls,
                extraction_text=f"<example {cls}>",
                attributes={"type": desc},
            )
        )

    return [
        lx.data.ExampleData(
            text="This is a placeholder example sentence with <example> entities.",
            extractions=sample_extractions,
        )
    ]


def extract_facts(text: str, schema: dict) -> dict:
    """
    Extract structured facts from raw text using LangExtract.

    Args:
        text:   Raw Markdown / plain text to extract from.
        schema: A dict mapping entity names to descriptions.
                Example: {"dates": "date", "metrics": "numeric value",
                          "definitions": "concept definition"}

    Returns:
        A dict with two keys:
        - "facts": list of extracted facts, each containing:
            - class (str):       the entity category
            - text (str):        the exact extracted text
            - source_span (dict): {"start": int, "end": int} character offsets
            - attributes (dict): extra context from the model
        - "raw_jsonl": the raw JSONL string from LangExtract (for archiving)
    """
    logger.info(f"Refiner: extracting facts for schema keys {list(schema.keys())}")

    prompt = _build_prompt(schema)
    examples = _build_examples(schema)

    # --- Call LangExtract ---------------------------------------------------
    result = lx.extract(
        text_or_documents=text,
        prompt_description=prompt,
        examples=examples,
        model_id=REFINER_MODEL,
        api_key=GOOGLE_API_KEY,
    )

    # --- Parse the AnnotatedDocument into a simple list of fact dicts --------
    facts = []

    # result is an AnnotatedDocument with a .extractions list
    extractions = result.extractions or []
    for ext in extractions:
        # Source span lives in ext.char_interval (CharInterval dataclass)
        span = ext.char_interval
        fact = {
            "class": ext.extraction_class,
            "text": ext.extraction_text,
            "source_span": {
                "start": span.start_pos if span else None,
                "end": span.end_pos if span else None,
            },
            "attributes": ext.attributes if ext.attributes else {},
        }
        facts.append(fact)

    logger.info(f"Refiner: extracted {len(facts)} facts")

    # Build a raw JSONL string for optional archiving
    raw_jsonl = ""
    try:
        import json
        import tempfile, os

        # langextract's save utility to get proper JSONL
        tmp_path = os.path.join(tempfile.gettempdir(), "argus_refiner_tmp.jsonl")
        lx.io.save_annotated_documents([result], output_name="argus_refiner_tmp.jsonl", output_dir=tempfile.gettempdir())
        if os.path.exists(tmp_path):
            with open(tmp_path, "r", encoding="utf-8") as f:
                raw_jsonl = f.read()
            os.remove(tmp_path)
    except Exception as e:
        logger.debug(f"Could not serialize raw JSONL: {e}")
        raw_jsonl = str(result)

    return {"facts": facts, "raw_jsonl": raw_jsonl}
