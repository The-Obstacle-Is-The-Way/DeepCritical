# Mgrep Semantic Code Search Demo

End-to-end demonstration of semantic code search using FAISS vector store and sentence transformers.

## What This Does

Indexes a codebase and performs instant semantic search:

1.  **Indexing:** Watches directories and indexes code files
2.  **Embeddings:** Uses sentence-transformers for semantic understanding
3.  **Search:** FAISS vector similarity search (<10ms)
4.  **Agentic:** LLM agent that decides when to use mgrep

## Quick Start

### 1. Direct API Demo

```bash
uv run python examples/mgrep_semantic_search/simple_demo.py
```

Shows:

*   Server startup
*   Automatic indexing
*   Direct search API calls
*   Results with scores

### 2. Agentic Demo

```bash
export ANTHROPIC_API_KEY="your-key"
uv run python examples/mgrep_semantic_search/agentic_demo.py "Find all thread safety patterns in FAISS code"
```

The agent will:

1.  Understand your natural language prompt
2.  Decide to use `mgrep_search` tool
3.  Execute semantic search
4.  Synthesize and return results

### Example Prompts

*   "Find database connection code"
*   "Show me error handling patterns"
*   "Where is FAISS thread safety implemented?"
*   "Find all async/await usage"

## Architecture

### Direct API (`simple_demo.py`)

```python
server = MgrepServer.get_instance()
server.start()
results = await server.search("query", top_k=5)
```

### Agentic (`agentic_demo.py`)

```python
from DeepResearch.src.utils.config_loader import ModelConfigLoader

config = ModelConfigLoader()
agent = Agent(model=config.get_default_llm_model(), ...)

@agent.tool_plain
async def mgrep_search(query: str) -> str:
    # Tool that LLM can call
    return results
```

## Key Files

*   `simple_demo.py`: Direct API usage (formerly `mgrep_e2e_demo.py`)
*   `agentic_demo.py`: LLM agent demo
*   `mgrep_agent.py`: Agent definition
*   `mgrep_deps.py`: Agent dependencies

## Performance

*   **Indexing:** ~50 docs/second
*   **Search:** <10ms per query
*   **Memory:** Minimal with local embeddings
