# Memory & Context Management Brainstorm (Issue #31)

**Branch**: `brainstorm/memory-context-management-issue-31`
**Issue**: [#31 - Add Agent Memory and Workflow Context Management](https://github.com/DeepCritical/DeepCritical/issues/31)
**Created**: 2025-10-05
**Last Updated**: 2025-10-18
**Requestor**: @Josephrp (Tonic)
**Main Contributor**: @MarioAderman

---

## Problem Statement

> "Some agents need a memory store that selectively or comprehensively injects into their context window, it's a similar situation with workflows"

**Priority**: Critical (Blocking current work)

### Core Challenge

Mario appears to be struggling with:
1. **Architectural Decision Paralysis**: Multiple viable approaches (mem0, memgpt, zep, custom KV cache, in-memory store)
2. **Integration Complexity**: How to integrate memory into the existing Pydantic Graph + Pydantic AI architecture
3. **Selective vs Comprehensive Memory**: Which agents need what type of memory?
4. **Storage Backend**: How to leverage existing infrastructure (Neo4j, ChromaDB, Qdrant)?

---

## Mario's Proposed Solution

Mario has designed a **Ports & Adapters Pattern** architecture:

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         WORKFLOW LEVEL                                   │
│  BioinformaticsState, PRIMEState, etc. (session_id tracking)            │
│                              ▼                                           │
│                        ┌──────────┐                                      │
│                        │ BaseAgent│ (auto-retrieve/store)                │
│                        └─────┬────┘                                      │
└──────────────────────────────┼───────────────────────────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
         ┌────────┐      ┌────────┐      ┌────────┐
         │Profile │      │Profile │      │Profile │  ◄── Agent Filters
         │ BioInfo│      │ PRIME  │      │ Search │      (what matters)
         └────┬───┘      └────┬───┘      └────┬───┘
              └────────────────┼────────────────┘
                               ▼
                        ┌──────────┐
                        │ Factory  │  ◄── Config Router (YAML → Provider)
                        │ (router) │
                        └─────┬────┘
                              │
                ┌─────────────┴──────────────┐
                │  MemoryProvider Protocol   │  ◄── PORT (interface)
                └─────────────┬──────────────┘
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │ CustomProvider│  │BaselineProvider│ │LettaAdapter  │  ◄── ADAPTERS
    │ (hierarchical)│  │   (FIFO)      │  │(ext. wrapper)│      (swappable)
    └───────┬──────┘  └───────┬──────┘  └───────┬──────┘
            │                 │                  │
            └─────────────────┼──────────────────┘
                              ▼
              ┌───────────────────────────────────────┐
              │      STORAGE LAYER (pluggable)        │
              ├───────────────────────────────────────┤
              │  Chroma      │  Neo4j     │  Local    │
              │  (vectors)   │  (graph)   │  (disk)   │
              └───────────────────────────────────────┘
```

### Key Components

1. **Port (Interface)**: `MemoryProvider` protocol - defines what agents expect
2. **Adapters (Implementations)**:
   - `CustomProvider`: Hierarchical implementation
   - `BaselineProvider`: Simple FIFO
   - `LettaAdapter`: Future external API wrapper
3. **Factory (Config Router)**: Reads YAML → instantiates matching provider (for benchmark experiments)
4. **Profiles (Agent Filters)**: What each agent type cares about
   - **BioinformaticsAgent**: papers/genes > user prefs
   - **PRIMEAgent**: tool history > chat

---

## Proposed Alternatives

From the original issue:

1. **Vendor-in in-memory memory-store**
2. **Add `mem0`** - https://github.com/mem0ai/mem0
3. **Vendor-in `memgpt`** - https://github.com/cpacker/MemGPT
4. **Add `zep`** - https://github.com/getzep/zep
5. **Vendor-in KV cache**
6. **Wrap our own**

---

## Use Cases

> "Almost every agent needs this. Almost every workflow will need this to be part of a graph."

### Specific Use Cases (Inferred)

1. **BioinformaticsAgent**:
   - Store previous BLAST results
   - Remember failed tool attempts (adaptive re-planning)
   - Retain protein sequence analysis context
   - Cache research paper references

2. **PRIMEAgent**:
   - Tool execution history
   - Previous design iterations
   - Molecular constraints from earlier in the workflow

3. **DeepSearch Workflow**:
   - Previous search queries
   - Visited URLs
   - Extracted research snippets

4. **Code Execution Agent**:
   - Previous code executions
   - Error patterns
   - Successful debugging strategies

---

## Community Feedback

### @anabossler (2025-10-18)
> "Looks great! like the Port/Adapter abstraction.
> I'd just make sure the memory layer supports relevance-ranked retrieval (maybe we can use MMR?)
> Also +1 for considering a Neo4j adapter!"

---

## What Mario Might Be Struggling With

Based on the issue and architecture:

### 1. **Integration Points**
- **Where** in the codebase to add memory?
  - In `ResearchState` dataclass?
  - In `BaseNode` class?
  - In individual agent classes?
  - As a separate `MemoryManager` service?

### 2. **State Management**
- How does memory interact with `GraphRunContext[ResearchState]`?
- Should memory be part of the state or external to it?
- How to handle session IDs across workflow transitions?

### 3. **Pydantic AI Integration**
- How to inject memory into `Agent(deps_type=...)` dependencies?
- Should memory be a tool (`@agent.tool_plain`) or a dependency?
- How to handle memory in deferred tool execution (`@defer`)?

### 4. **Storage Backend Decision**
- **Neo4j**: Already in the stack, supports graph relationships + vector indexing
- **ChromaDB**: Already used, lightweight vector store
- **Qdrant**: Already configured, production-grade vector DB
- **All three**: Use Neo4j for graph, ChromaDB/Qdrant for vectors?

### 5. **Retrieval Strategy**
- **Selective**: Filter by agent profile (Mario's approach)
- **Comprehensive**: Dump everything into context
- **Ranked**: Use MMR or similarity search
- **Hierarchical**: Short-term vs long-term memory

### 6. **Configuration Design**
- Where in Hydra config hierarchy?
  - `configs/memory.yaml`?
  - `configs/agents/memory.yaml`?
  - Per-agent config in `configs/agents/*.yaml`?
- How to enable/disable per workflow?

### 7. **Testing Strategy**
- How to test memory persistence across workflow runs?
- How to benchmark different providers?
- How to validate selective memory injection?

---

## Architecture Analysis: Existing System

### Current State Management

**File**: `DeepResearch/app.py`

```python
@dataclass
class ResearchState:
    question: str
    config: OmegaConf
    notes: List[str] = field(default_factory=list)
    results: Dict[str, Any] = field(default_factory=dict)
    # ... more fields
```

**Nodes** modify state through `GraphRunContext`:
```python
async def run(self, ctx: GraphRunContext[ResearchState]) -> NextNode:
    ctx.state.notes.append("New note")
    return NextNode()
```

### Current Agent System

**File**: `DeepResearch/src/agents/`

Pydantic AI agents use `deps_type` for dependencies:
```python
agent = Agent(
    model="anthropic:claude-sonnet-4-0",
    deps_type=AgentDeps,
    result_type=ResultType
)
```

---

## Brainstorm: Integration Approaches

### Option 1: Memory as State Field
**Add memory to `ResearchState`:**

```python
@dataclass
class ResearchState:
    question: str
    config: OmegaConf
    memory: MemoryProvider  # <-- Add here
    notes: List[str] = field(default_factory=list)
    results: Dict[str, Any] = field(default_factory=dict)
```

**Pros**:
- Easy access in all nodes
- Part of workflow state lifecycle
- Simple to pass to agents

**Cons**:
- Memory might not be serializable
- State gets heavier
- Tight coupling

---

### Option 2: Memory as Dependency Injection
**Pass memory through agent dependencies:**

```python
@dataclass
class AgentDeps:
    config: OmegaConf
    memory: MemoryProvider  # <-- Add here
    session_id: str

agent = Agent(
    deps_type=AgentDeps,
    # ...
)
```

**Pros**:
- Clean separation of concerns
- Follows dependency injection pattern
- Easy to mock in tests

**Cons**:
- Requires modifying all agent signatures
- More boilerplate

---

### Option 3: Memory as Singleton Service
**Global memory manager:**

```python
# DeepResearch/src/services/memory_manager.py
class MemoryManager:
    _instance = None

    @classmethod
    def get_instance(cls) -> MemoryManager:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
```

**Pros**:
- Easy to access anywhere
- No state modification needed
- Lightweight

**Cons**:
- Global state (anti-pattern)
- Hard to test
- Session management complexity

---

### Option 4: Memory as Tool (Mario's Likely Approach)
**Make memory a Pydantic AI tool:**

```python
@agent.tool_plain
def retrieve_memory(
    ctx: RunContext[AgentDeps],
    query: str,
    agent_profile: str
) -> List[MemoryItem]:
    """Retrieve relevant memory for this agent"""
    memory = ctx.deps.memory
    return memory.retrieve(query, profile=agent_profile)

@agent.tool_plain
def store_memory(
    ctx: RunContext[AgentDeps],
    content: str,
    metadata: Dict[str, Any]
) -> None:
    """Store new memory"""
    ctx.deps.memory.store(content, metadata)
```

**Pros**:
- Agents explicitly control memory access
- Fits Pydantic AI tool paradigm
- Easy to audit what's being stored/retrieved

**Cons**:
- Requires agent to know when to use memory
- More token usage (LLM decides when to call)

---

## Recommendations for Mario

### 1. **Start with Option 2 + Option 4 Hybrid**
- Add `memory: MemoryProvider` to `AgentDeps`
- Provide memory as tools for explicit access
- This balances flexibility with control

### 2. **Use Neo4j as Primary Storage**
- Already in the stack
- Supports both graph relationships AND vector indexing
- Can model hierarchical memory (short-term → long-term)

### 3. **Implement Profiles as Pydantic Models**

```python
# DeepResearch/src/memory/profiles.py
from pydantic import BaseModel

class MemoryProfile(BaseModel):
    agent_type: str
    priorities: List[str]  # ["papers", "genes", "tool_history"]
    max_items: int = 10
    relevance_threshold: float = 0.7

BIOINFORMATICS_PROFILE = MemoryProfile(
    agent_type="bioinformatics",
    priorities=["papers", "genes", "blast_results"],
    max_items=15
)
```

### 4. **Configuration Structure**

```yaml
# configs/memory.yaml
defaults:
  - _self_

memory:
  provider: custom  # or baseline, letta
  storage:
    backend: neo4j  # or chroma, qdrant, local
    connection: ${db.neo4j}

  profiles:
    bioinformatics:
      priorities: [papers, genes, blast_results]
      max_items: 15
      relevance_threshold: 0.7

    prime:
      priorities: [tool_history, molecular_constraints]
      max_items: 10
      relevance_threshold: 0.8
```

### 5. **Implementation Roadmap**

1. **Phase 1: Protocol & Interface**
   - Define `MemoryProvider` protocol
   - Create `MemoryProfile` models
   - Add to `AgentDeps`

2. **Phase 2: Baseline Implementation**
   - `BaselineProvider` (simple FIFO)
   - Local storage only
   - No ranking, just append/retrieve

3. **Phase 3: Custom Hierarchical**
   - `CustomProvider` with Neo4j
   - Profile-based filtering
   - MMR retrieval

4. **Phase 4: External Adapters**
   - `LettaAdapter` (if needed)
   - `Mem0Adapter` (if needed)

---

## Questions for Mario

1. **Storage Backend**: Do you want to use Neo4j (already in stack) or add a new dependency?

2. **Retrieval Strategy**: How should agents decide what memory to retrieve?
   - Similarity search?
   - Recency-biased?
   - Profile-filtered then ranked?

3. **Session Management**: How do you want to handle `session_id`?
   - Generate per workflow run?
   - User-provided?
   - Stored in `ResearchState`?

4. **Memory Lifecycle**: When should memory be cleared?
   - Never (persistent across all runs)?
   - Per-session?
   - Configurable TTL?

5. **Integration Preference**: Where do you feel most comfortable adding memory?
   - `ResearchState`?
   - `AgentDeps`?
   - As tools?
   - Singleton service?

---

## Next Steps

1. **Document Feedback**: Get Mario's answers to the questions above
2. **Prototype**: Start with `BaselineProvider` + local storage
3. **Test Integration**: Add to one agent (e.g., BioinformaticsAgent)
4. **Iterate**: Gather feedback, refine approach
5. **Scale**: Roll out to other agents once proven

---

## Resources

- **Issue**: https://github.com/DeepCritical/DeepCritical/issues/31
- **Pydantic AI Docs**: https://ai.pydantic.dev/
- **Neo4j Vector Index**: https://neo4j.com/docs/cypher-manual/current/indexes-for-vector-search/
- **MMR (Maximal Marginal Relevance)**: https://python.langchain.com/docs/modules/model_io/prompts/example_selectors/mmr

---

**Let's help Mario ship this! 🚀**
