"""
Project Argus - Model Ping Tests

Sends a minimal "Hello" prompt to every configured agent model
to verify API keys and model availability are working.

Run with:
    python -m pytest tests/test_model_ping.py -v
"""

import pytest
import src.config  # Loads .env
from src.utils.llm_factory import get_llm
from src.config import (
    LIBRARIAN_MODEL, LIBRARIAN_PROVIDER,
    CRITIC_MODEL, CRITIC_PROVIDER,
    VERIFIER_MODEL, VERIFIER_PROVIDER,
    WRITER_MODEL, WRITER_PROVIDER,
    REFINER_MODEL, REFINER_PROVIDER,
)

# Every test here calls a real provider API, so the whole module is a live probe:
# skipped by default, run with `pytest --run-live` (see conftest.py).
pytestmark = pytest.mark.live

# Each tuple: (test_id, model_name, provider)
AGENT_MODELS = [
    ("librarian", LIBRARIAN_MODEL, LIBRARIAN_PROVIDER),
    ("critic", CRITIC_MODEL, CRITIC_PROVIDER),
    ("verifier", VERIFIER_MODEL, VERIFIER_PROVIDER),
    ("writer", WRITER_MODEL, WRITER_PROVIDER),
    ("refiner", REFINER_MODEL, REFINER_PROVIDER),
]


@pytest.mark.parametrize("agent_name,model,provider", AGENT_MODELS, ids=[m[0] for m in AGENT_MODELS])
def test_model_ping(agent_name, model, provider):
    """Send a trivial prompt to verify each model responds without errors."""
    llm = get_llm(model, provider, temperature=0)
    response = llm.invoke("Reply with only the word 'OK'.")

    # Ensure we got a non-empty response
    content = response.content if hasattr(response, "content") else str(response)
    assert content, f"{agent_name} ({model} via {provider}) returned an empty response"
    print(f"  ✓ {agent_name:12s} | {provider:6s} / {model:30s} → {content.strip()[:60]}")
