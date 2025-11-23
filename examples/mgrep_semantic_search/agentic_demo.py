#!/usr/bin/env python3
"""
Agentic mgrep demo - LLM agent using semantic code search.

Usage:
    uv run python examples/mgrep_semantic_search/agentic_demo.py "Find all FAISS thread safety code"

The agent will:
1. Understand your prompt
2. Decide to use mgrep_search tool
3. Execute semantic search
4. Return synthesized results
"""

# CRITICAL: Set thread limits BEFORE any imports to prevent N² thread explosion
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Prevent fork deadlock warnings

import asyncio
import sys

from examples.mgrep_semantic_search.mgrep_agent import run_mgrep_analysis


async def main() -> None:
    if len(sys.argv) < 2:
        print(
            "Usage: uv run python examples/mgrep_semantic_search/agentic_demo.py <prompt>"
        )
        print(
            "Example: uv run python examples/mgrep_semantic_search/agentic_demo.py 'Find FAISS thread safety patterns'"
        )
        sys.exit(1)

    prompt = sys.argv[1]
    print(f"🔎 Agent Prompt: {prompt}\n")

    try:
        result = await run_mgrep_analysis(prompt)
        print(result)
    except Exception as e:
        print(f"Error running agent: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
