# Implementation Plan: Phase 8 (Recursive Self-Improvement Loop)

## Overview
The goal of this phase is to move from a blunt re-search model to a targeted, recursive pipeline that intelligently fills knowledge gaps. 

## Tasks

| # | File | Change | Goal |
|---|------|--------|------|
| 1 | `src/schema/state.py` | Add `knowledge_gaps` and `gap_queries` state | Track what is missing and what to search next |
| 2 | `src/graph/kg.py` | Add `find_gaps(query, existing_facts)` method | Query the KG to determine topics with insufficient coverage |
| 3 | `src/agents/verifier.py` | Set `knowledge_gap_detected` flag on `NOT_SUPPORTED` facts | Trigger the new reflector loop when facts are weakly supported |
| 4 | `src/agents/reflector.py` | Create Reflector node | Targeted follow-up queries based on gap analysis |
| 5 | `src/graph/builder.py` | Add conditional routing | Wire Critic -> Reflector -> Scout loop for gaps |
