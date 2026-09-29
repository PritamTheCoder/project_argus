# Project Argus

Argus turns a research question into a cited report. A LangGraph pipeline plans the search, reads source pages, extracts facts, checks those facts against the text they came from, and writes the report. A final pass checks that each cited sentence is actually supported by the evidence behind its citation.

By Pritam Thapa

## Pipeline

```
librarian → plan_gate → scout → refiner → verifier → fact_checker
                              ▲                         │
                              │            ┌────────────┼─────────────┐
                              │            ▼            ▼             ▼
                              └────────── scout    reflector     consensus
                                       (broaden)       │             │
                                                       ▼             ▼
                                                     scout      ghostwriter
                                                  (fill a gap)       │
                                                                     ▼
                                                             citation_auditor
```

`fact_checker` is the Critic. After it, the graph either searches again or moves on. The loop stops at `MAX_RESEARCH_LOOPS` (default 2). `consensus` runs once, the Writer (`ghostwriter`) drafts the report, and `citation_auditor` checks the draft before the run ends.

| Node | What it does |
|---|---|
| Librarian | Splits the question into sub-queries, each tagged `TRUSTED_ONLY`, `TRUSTED_FIRST`, or `MIXED`. |
| Plan gate | Continues immediately, or pauses when the request sets `require_approval`. |
| Scout | Picks retrieval tools, fetches pages, and keeps the passages relevant to the question. |
| Refiner | Extracts structured facts from those pages. |
| Verifier | Grounds each fact to a verbatim quote and records support level and source credibility. |
| Critic | Looks for gaps and weak sources, and decides whether another search loop is warranted. |
| Reflector | Writes targeted follow-up queries when the Critic found specific gaps. |
| Consensus | Clusters agreeing claims and surfaces contradictions. |
| Writer | Drafts the markdown report. Claims cite sources as `[n]`. |
| Citation auditor | Judges each cited sentence against the evidence for those `[n]` ids. Verdicts are `SUPPORTED`, `PARTIALLY_SUPPORTED`, `UNSUPPORTED`, or `CONTRADICTED`. |

Scout can call `web_search` (Exa, then Brave, then DuckDuckGo), Semantic Scholar, arXiv, Crossref, Europe PMC, SEC EDGAR, a knowledge-graph lookup, and a calculator. Academic abstracts and Exa page text are used as returned. Snippet-only results are fetched with Crawl4AI, which drives a Playwright Chromium browser.

When the citation auditor finds cited sentences, it prepends a **Citation Integrity** line to the report and stores one verdict per sentence on `citation_audit`.

## Setup

Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
cp .env.example .env
```

`playwright install chromium` downloads the browser Crawl4AI uses. Re-run it on a new machine or a rebuilt virtualenv. Without it, scraping fails with `Executable doesn't exist`.

A run needs a [Groq](https://console.groq.com) key in `.env` (`GROQ_API_KEY`, or `GROQ_API_KEY_A` / `_B` / `_C`). The default model for every agent is `openai/gpt-oss-120b` on Groq. Set `GOOGLE_API_KEY` as well if you want Gemini (`gemini-3.1-flash-lite`) as the fallback when Groq fails.

Search works with no search key: `web_search` uses DuckDuckGo. `EXA_API_KEY` and `BRAVE_API_KEY` are optional earlier rungs. Semantic Scholar, arXiv, Crossref, Europe PMC, and SEC EDGAR do not require a key. SEC asks for a descriptive `EDGAR_USER_AGENT` with a real contact address before heavy use.

## Command line

```bash
python main.py "What are the latest breakthroughs in solid-state batteries?"
```

With no argument, `main.py` uses a built-in solid-state battery question. The process prints each node as it runs, then the report. Run state is stored in SQLite.

## HTTP API

```bash
uvicorn src.api.app:app --reload
```

Interactive docs are at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs). `GET /` lists the routes. `GET /health` needs no key.

Research routes require an API key. Create one (shown once):

```bash
python -m src.api.auth "your name"
```

Send it as `Authorization: Bearer <key>`.

Submit a question. The response is `202` with a `job_id`. Poll until `status` is `done` or `error`.

```bash
curl -s -X POST http://127.0.0.1:8000/research \
  -H "Authorization: Bearer $ARGUS_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the latest breakthroughs in solid-state batteries?"}'

curl -s http://127.0.0.1:8000/research/$JOB_ID \
  -H "Authorization: Bearer $ARGUS_API_KEY"
```

A job moves through `queued`, `running`, and then `done` or `error`. A finished job includes `report`, `source_map`, `quality_score`, `citation_audit`, and `usage`. `citation_audit` is a list of `{sentence, cited_ids, verdict, reason}`.

Set `"require_approval": true` on `POST /research` to pause after planning. Status becomes `awaiting_approval` and the job carries `pending_plan`. Resume with `POST /research/{job_id}/approve-plan` and the final sub-query list (`query` plus `mode`: `TRUSTED_ONLY`, `TRUSTED_FIRST`, or `MIXED`).

Other routes, all key-scoped to the jobs you created:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/research/` | List your jobs. |
| `GET` | `/research/{job_id}/graph` | Facts, sources, contradictions, consensus findings, and gaps. Fills in while the job is still running. |
| `GET` | `/research/{job_id}/facts/{fact_id}` | One fact: quote, source, credibility, contradictions. |
| `GET` | `/research/{job_id}/history` | Checkpoints you can branch from. |
| `POST` | `/research/{job_id}/branch` | New job from a checkpoint, with an extra sub-query. The source job stays as it was. |
| `POST` | `/research/{job_id}/dig-deeper` | New job that follows up on one fact, one gap, or a query you supply. |
| `POST` | `/research/{job_id}/facts/{fact_id}/flag` | Mark a fact disputed. By default this also starts a re-verification job. |

A missing job and a job owned by someone else both return `404`.

## Tests

```bash
python -m pytest -q
```

Nine tests are marked `live` (six model pings and three search/scout calls). They skip unless you pass `--run-live` or set `RUN_LIVE_TESTS=1`. Those need network access and API keys.
