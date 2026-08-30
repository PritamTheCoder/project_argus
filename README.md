# Project Argus: The Multi-Agent Research Pod

## Vision Statement 

**Project Argus** aims to solve the "Hallucination and Rabbit Hole" problem in AI research. While most agents simply summarize snippets, Argus utilizes a Pod Architecture to mimic a professional editorial room. It prioritizes Deep Scraping (reading the full text, not just the search snippet) and Interventionist UX, allowing humans to guide the agent before it commits to a deep-dive path.

By : Pritam Thapa | PritamTheCoder

## Setup

Requires Python >= 3.10.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
```

`playwright install chromium` downloads the browser binary Crawl4AI drives for
deep-scraping; it's a separate step from `pip install` and must be re-run on
any new machine or rebuilt venv, or the scraper tests/nodes fail with
`Executable doesn't exist at ...`.

Run tests: `python -m pytest -q` (5 tests are `@pytest.mark.live` and skip by
default — pass `--run-live` or set `RUN_LIVE_TESTS=1` to run them against real
network/API keys).