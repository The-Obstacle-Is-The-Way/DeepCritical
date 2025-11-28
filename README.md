# DeepCritical Fork - ABANDONED

> **This fork is abandoned. Do not use this codebase.**

---

## Why This Project Is Abandoned

After extensive code analysis, I've concluded that **this codebase has fundamental architectural problems that make it unlikely to ever function as a working deep research agent**, even with significant engineering effort.

**Full technical post-mortem:** [WHY_THIS_FORK_IS_ABANDONED.md](./WHY_THIS_FORK_IS_ABANDONED.md)

---

## The Hard Numbers

| Metric | This Codebase | Industry Standard |
|--------|---------------|-------------------|
| **Test Coverage** | **35%** | 80%+ |
| **MCP Servers** | **28** (25,253 lines) | 3-5 focused |
| **Modules with 0% Coverage** | **20+ critical** | 0 |
| **NotImplementedError in production paths** | **5+** | 0 |
| **Lines NOT tested** | **18,912** | - |

---

## The "Too Many Tools" Problem

This codebase exposes **100-200+ tools** to the LLM. Research shows this fundamentally breaks agent performance:

> "You start feeling the decline in tool calling accuracy surprisingly early, after adding just a handful of tools."
> — [Writer Engineering](https://writer.com/engineering/rag-mcp/)

> "Having too many tools exposed can degrade the performances and the output of your LLM!"
> — [Stefano Demiliani](https://demiliani.com/2025/09/04/model-context-protocol-and-the-too-many-tools-problem/)

**Cursor IDE hard-limits MCP tools to 40.** This codebase has 5x that.

---

## Even 1-2 Million Token Context Won't Save It

Current LLM context windows (2025):

| Model | Context Window |
|-------|----------------|
| Gemini 2.5 Pro | 1,000,000 tokens |
| Gemini 1.5 Pro | 2,000,000 tokens |
| Claude 3.5 | 200,000 tokens |
| GPT-4 Turbo | 128,000 tokens |

**The problem isn't context size.** It's the ["Lost in the Middle" phenomenon](https://medium.com/@adityakamat007/understanding-llm-context-windows-why-400k-tokens-doesnt-mean-what-you-think-918704d04085):

> "LLMs perform best when important information is at the beginning or end of the context, not buried in the middle."

With 200+ tool definitions, critical tools get buried. Even with Gemini's 2M tokens, **the model spends its attention budget parsing tool schemas instead of reasoning about the task**.

The computational cost also scales **O(n²)** with context length. Doubling context quadruples inference cost.

---

## Zero Integration Testing on Critical Paths

These modules have **0% test coverage**:

| Module | Lines | Purpose |
|--------|-------|---------|
| `deep_agent_graph.py` | 215 | **Core workflow orchestration** |
| `vllm_agent.py` | 103 | Local LLM support |
| `workflow_pattern_agents.py` | 191 | Agent patterns |
| `bioinformatics_agent_implementations.py` | 108 | Bio agent prompts |

**Nobody has ever run these end-to-end and verified they work.**

---

## NotImplementedError Everywhere

These will crash at runtime:

```python
# 5 Bioinformatics MCP Servers
BWAServer:      raise NotImplementedError
TopHatServer:   raise NotImplementedError
HTSeqServer:    raise NotImplementedError
PicardServer:   raise NotImplementedError
HOMERServer:    raise NotImplementedError

# FAISS Vector Store
add_document_chunks():      raise NotImplementedError
add_document_text_chunks(): raise NotImplementedError

# vLLM Integration (3 methods)
raise NotImplementedError
```

---

## Horizontal Bloat vs Vertical Slices

**How state-of-the-art research agents are built** ([arXiv: Deep Research Agents](https://arxiv.org/abs/2506.18096)):

> "Dynamic Subagents are a key architectural update—the planner LLM can define subagents dynamically after analyzing the user's objective, instead of defining all MCP servers and subagents upfront."

**What this codebase does (the opposite):**

```
Layer 1: 28 MCP servers ──────────► None work together
Layer 2: 15+ agents ──────────────► 11-35% coverage
Layer 3: 6 state machines ────────► 0-26% coverage
Layer 4: Orchestration ───────────► TODOs everywhere
                ↓
        Nothing works end-to-end
```

---

## Probability It Would Ever Work

With 28 servers × 6 workflows × 15 agents = **2,520+ interaction paths**

At 35% coverage:
- P(single component works) ≈ 0.35
- P(5-component workflow) ≈ 0.35^5 ≈ **0.5%**

---

## What Good Looks Like

From [MCP-Agent best practices](https://thealliance.ai/blog/building-a-deep-research-agent-using-mcp-agent):

> "Build focused MCP servers from the beginning. Don't create a single MCP server for everything."

From [RAG-MCP research](https://writer.com/engineering/rag-mcp/):

> "The RAG-MCP approach more than triples tool-selection accuracy and reduces prompt tokens by over 50%."

---

## My Contributions (Before Abandoning)

I shipped these as **vertical slices with tests** before realizing the architectural problems:

| PR | What | Quality |
|----|------|---------|
| #174 | Fixed **204 type errors** | Production-grade |
| #175 | Test suite + bug fix | TDD approach |
| #179 | GunzipServer MCP | Full implementation |
| #183 | HaplotypeCaller MCP | Full implementation |
| #217 | Embeddings + FAISS | Full implementation |

They were absorbed into horizontal bloat. I regret contributing professional work to a project that doesn't maintain professional standards.

---

## References

### MCP Tool Bloat Problem
- [Model Context Protocol and the "too many tools" problem](https://demiliani.com/2025/09/04/model-context-protocol-and-the-too-many-tools-problem/)
- [The MCP Tool Trap](https://jentic.com/blog/the-mcp-tool-trap)
- [When too many tools become too much context](https://writer.com/engineering/rag-mcp/)
- [SEP-1576: Mitigating Token Bloat in MCP](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1576)

### Deep Research Agent Architecture
- [Deep Research Agents: A Systematic Examination (arXiv)](https://arxiv.org/abs/2506.18096)
- [MCP-Agent: Building Scalable Deep Research Agents](https://thealliance.ai/blog/building-a-deep-research-agent-using-mcp-agent)
- [State of AI Agents in 2025](https://carlrannaberg.medium.com/state-of-ai-agents-in-2025-5f11444a5c78)

### Context Window Research
- [Understanding LLM Context Windows](https://medium.com/@adityakamat007/understanding-llm-context-windows-why-400k-tokens-doesnt-mean-what-you-think-918704d04085)
- [Gemini Long Context Explained](https://ai.google.dev/gemini-api/docs/long-context)
- [Google DeepMind: Long Context Windows](https://blog.google/technology/ai/long-context-window-ai-models/)

---

## License

Original upstream code is GPL-3.0. My contributions remain under that license.
