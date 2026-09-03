"""Project Argus - shared pytest configuration.

Defines the ``live`` marker for tests that hit real external services (LLM
providers, the open web). These depend on network, API keys, and third-party
availability — a provider returning 503 ("model overloaded") is an environment
condition, not a code regression — so they are SKIPPED by default and must be
opted into explicitly:

    pytest --run-live            # run everything including live probes
    RUN_LIVE_TESTS=1 pytest      # same, via env var (CI-friendly)
"""

import os
import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "live: test hits real external services (LLM APIs / web); skipped unless --run-live.",
    )


def pytest_addoption(parser):
    parser.addoption(
        "--run-live",
        action="store_true",
        default=False,
        help="Run tests marked 'live' that call real external services.",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-live") or os.getenv("RUN_LIVE_TESTS") == "1":
        return
    skip_live = pytest.mark.skip(reason="live test (needs network/keys); pass --run-live to run")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)
