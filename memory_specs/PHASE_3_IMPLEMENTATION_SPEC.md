# Phase 3: Memory System Implementation Spec

**Status**: ✅ **IRON-CLAD** (Audited 2025-11-20)
**Date**: 2025-11-20 (Revised from first principles)
**Goal**: Define the concrete, minimal-complexity architecture for integrating long-term memory into DeepCritical.

**Philosophy**: Start simple (pure Mem0), ship fast, iterate based on real-world pain points. Avoid premature optimization.

---

## Section 0: Architecture Decision Record

### Decision: Start with Pure Mem0 (Option A)

**From Phase 2's Decision Tree**:
- ✅ **Option A**: Pure Mem0 OSS (CHOSEN for Phase 3/4)
- ⏸️ **Option B**: Pure Letta OSS (fallback if Mem0 fails)
- ⏸️ **Option C**: Hybrid (Mem0 + G-Memory patterns) - DEFERRED to Phase 5+
- ⏸️ **Option D**: Custom (H-MEM + Zep patterns) - DEFERRED to Phase 5+

### Why Pure Mem0?

**✅ Pros**:
1. **Production-ready**: Y Combinator-backed, 43,252 stars, SaaS fallback
2. **Neo4j compatible**: Already have Neo4j in Phase 1 baseline
3. **Hybrid storage**: Graph + Vector + KV (flexible)
4. **Token efficiency**: 90% savings (critical for ~28 bioinformatics modules)
5. **Fastest to ship**: No custom graph logic, use Mem0's APIs directly
6. **Low risk**: If Mem0 fails, pivot to Letta (Option B) in Phase 5

**❌ Cons**:
1. **Multi-agent gaps**: Designed for single-agent contexts (not hierarchical orchestration)
2. **Pydantic AI integration unclear**: No documented patterns (requires custom adapter)

**Decision Rationale**: Meta-Plan says "ship iterative vertical slices." Pure Mem0 is simplest path to working memory. We can add G-Memory patterns (Option C) LATER if Mem0 proves insufficient for multi-agent orchestration.

### When We'd Pivot

**To Letta (Option B)** if:
- Mem0's graph schema conflicts with our Agent/Workflow/Action structure (test in Phase 4A)
- Mem0's LLM-driven extraction adds unacceptable latency (>500ms per operation)
- Mem0's SaaS-first design limits self-hosted customization

**To Hybrid (Option C)** if:
- Mem0 works for conversational memory BUT
- We need G-Memory's hierarchical graphs (Insight/Query/Interaction) for multi-agent coordination
- Performance testing shows Mem0's retrieval is too slow for tool-heavy workflows

**Decision Authority**: User + Mario (discuss after Phase 4A prototype)

---

## 1. Executive Summary

We will implement a **Ports & Adapters Memory System** powered by **Mem0 OSS**:
1. **Mem0 (OSS)** for all memory operations (chat, execution traces, agent profiles)
2. **MemoryProvider Protocol** for vendor-agnostic interface (swap Mem0 for Letta/custom if needed)
3. **Neo4j Backend** (via Mem0's Neo4j adapter - already supported)
4. **Agent-Specific Configuration** for selective retrieval per agent role

This approach follows Phase 2's "Option A" recommendation and Meta-Plan's "ship simple, iterate" philosophy.

---

## 2. Architecture Overview

```mermaid
graph TB
    subgraph "Agent Layer (Pydantic AI)"
        Agent[Pydantic AI Agent]
        Deps[AgentDependencies]
    end

    subgraph "Port (Interface)"
        Provider["&lt;Protocol&gt;<br/>MemoryProvider"]
    end

    subgraph "Adapter (Implementation)"
        Mem0Adapter[Mem0Adapter<br/>(wraps Mem0 client)]
    end

    subgraph "Storage Layer"
        Mem0Client[Mem0 Client Library]
        Neo4j[(Neo4j Database<br/>Graph + Vector + KV)]
    end

    Agent --> Deps
    Deps --> Provider
    Provider --> Mem0Adapter
    Mem0Adapter --> Mem0Client
    Mem0Client --> Neo4j
```

**Key Design Principles**:
1. **Thin Adapter**: Mem0Adapter is a thin wrapper over Mem0's client (no custom graph logic)
2. **Protocol-First**: MemoryProvider is abstract - swap Mem0 for Letta in 1 line of config
3. **Leverage Mem0**: Use Mem0's Neo4j adapter (don't reinvent graph writes)

---

## 3. Data Schema (Neo4j via Mem0)

Mem0 manages the schema dynamically. We don't fight it - we **adapt our data to fit Mem0's model**.

### Mem0's Schema (What We Get)

**Nodes** (auto-created by Mem0):
- `Entity` (User, Topic, Concept) - extracted from conversations
- `Memory` - stored facts/events

**Edges** (auto-created by Mem0):
- `RELATED_TO` - connections between entities
- `MENTIONED_IN` - entities → memories

### Our Usage Pattern (How We Structure Data)

**For Conversational Memory** (unstructured):
```python
mem0.add(
    "User asked about P53 protein interactions",
    user_id="user_123",
    agent_id="bioinformatics_agent",
    metadata={"topic": "protein_analysis", "gene": "P53"}
)
```

**For Execution Traces** (structured):
```python
mem0.add(
    f"Agent {agent_id} executed tool {tool_name} in workflow {workflow_id}. Result: {result}",
    user_id="system",
    agent_id=agent_id,
    metadata={
        "type": "execution_trace",
        "workflow_id": workflow_id,
        "tool_name": tool_name,
        "status": "success|failure",
        "timestamp": datetime.now().isoformat()
    }
)
```

**Key Insight**: We use Mem0's `metadata` field to store structured info (workflow_id, tool_name, status). Mem0's graph will auto-extract entities from text, and we can filter by metadata for precise queries.

**Migration from ExecutionHistory** (Phase 1 component):
- `ExecutionHistory.record()` will call `memory.add_trace()` internally
- ExecutionHistory becomes a **compatibility shim** - agents don't change
- In Phase 5+, deprecate ExecutionHistory if memory system proves sufficient

---

## 4. Core Interfaces (The "Port")

**Location**: `DeepResearch/src/memory/core.py`

```python
from typing import Protocol, Any
from datetime import datetime
from pydantic import BaseModel

class MemoryItem(BaseModel):
    """Single memory result from search/retrieval."""
    content: str
    score: float
    metadata: dict[str, Any]
    timestamp: datetime
    memory_id: str

class MemoryProvider(Protocol):
    """Vendor-agnostic memory interface (Ports & Adapters pattern)."""

    async def add(
        self,
        content: str,
        user_id: str,
        agent_id: str,
        metadata: dict[str, Any] | None = None
    ) -> str:
        """Add memory (conversational or execution trace).

        Args:
            content: Natural language description of memory
            user_id: User identifier (use "system" for execution traces)
            agent_id: Agent identifier (e.g., "bioinformatics_agent")
            metadata: Structured data (type, workflow_id, tool_name, etc.)

        Returns:
            memory_id: Unique identifier for this memory
        """
        ...

    async def search(
        self,
        query: str,
        user_id: str,
        agent_id: str,
        filters: dict[str, Any] | None = None,
        limit: int = 5
    ) -> list[MemoryItem]:
        """Semantic search across memories.

        Args:
            query: Natural language search query
            user_id: User context for filtering
            agent_id: Agent context for filtering
            filters: Metadata filters (e.g., {"type": "execution_trace", "workflow_id": "xyz"})
            limit: Max results to return

        Returns:
            Ranked list of relevant memories
        """
        ...

    async def get_all(
        self,
        user_id: str,
        agent_id: str,
        filters: dict[str, Any] | None = None,
        limit: int = 10
    ) -> list[MemoryItem]:
        """Get recent memories without semantic search.

        Args:
            user_id: User context
            agent_id: Agent context
            filters: Metadata filters (e.g., {"type": "execution_trace"})
            limit: Max results

        Returns:
            Recent memories (sorted by timestamp)
        """
        ...
```

**Design Notes**:
1. **Unified `add()` method**: No separate `add_trace()` - use `metadata["type"]` to distinguish conversational vs. execution traces
2. **Filters over separate methods**: `search(filters={"type": "execution_trace"})` instead of `get_history()`
3. **Consistent with Mem0's API**: Matches Mem0's `add(messages, user_id, metadata)` pattern

---

## 5. Mem0 Adapter Implementation

**Location**: `DeepResearch/src/memory/adapters/mem0_adapter.py`

```python
from mem0 import MemoryClient
from ..core import MemoryProvider, MemoryItem

class Mem0Adapter:
    """Adapter wrapping Mem0 client to implement MemoryProvider protocol."""

    def __init__(self, config: dict):
        """Initialize Mem0 client with Neo4j backend.

        Args:
            config: {
                "api_key": "...",  # Optional: use Mem0 Cloud
                "host": "localhost",  # For self-hosted
                "config": {
                    "graph_store": {
                        "provider": "neo4j",
                        "config": {
                            "url": "bolt://localhost:7687",
                            "username": "neo4j",
                            "password": "..."
                        }
                    },
                    "vector_store": {
                        "provider": "neo4j",  # Use Neo4j's vector index
                        "config": {...}
                    }
                }
            }
        """
        self.client = MemoryClient(**config)

    async def add(
        self,
        content: str,
        user_id: str,
        agent_id: str,
        metadata: dict | None = None
    ) -> str:
        """Add memory via Mem0."""
        messages = [{"role": "user", "content": content}]
        result = self.client.add(
            messages=messages,
            user_id=f"{user_id}:{agent_id}",  # Composite key for agent-specific memory
            metadata=metadata or {}
        )
        return result["id"]

    async def search(
        self,
        query: str,
        user_id: str,
        agent_id: str,
        filters: dict | None = None,
        limit: int = 5
    ) -> list[MemoryItem]:
        """Search memories via Mem0."""
        results = self.client.search(
            query=query,
            user_id=f"{user_id}:{agent_id}",
            limit=limit,
            filters=filters
        )
        return [
            MemoryItem(
                content=r["memory"],
                score=r["score"],
                metadata=r.get("metadata", {}),
                timestamp=r["created_at"],
                memory_id=r["id"]
            )
            for r in results
        ]

    async def get_all(
        self,
        user_id: str,
        agent_id: str,
        filters: dict | None = None,
        limit: int = 10
    ) -> list[MemoryItem]:
        """Get all memories for user+agent."""
        results = self.client.get_all(
            user_id=f"{user_id}:{agent_id}",
            limit=limit
        )
        # Filter by metadata if needed
        if filters:
            results = [r for r in results if all(r.get("metadata", {}).get(k) == v for k, v in filters.items())]
        return [
            MemoryItem(
                content=r["memory"],
                score=1.0,  # No score for get_all
                metadata=r.get("metadata", {}),
                timestamp=r["created_at"],
                memory_id=r["id"]
            )
            for r in results[:limit]
        ]
```

**Key Design Choices**:
1. **Composite user_id**: Use `f"{user_id}:{agent_id}"` to namespace memories per agent (agent-specific memory)
2. **Leverage Mem0's API**: No custom graph writes - Mem0 handles Neo4j operations
3. **Thin wrapper**: ~50 lines of code - just maps our protocol to Mem0's API

---

## 6. Integration with Phase 1 Baseline

### 6.1 AgentDependencies (from Phase 1)

**Location**: `DeepResearch/src/datatypes/agents.py`

**Current State** (Phase 1):
```python
@dataclass
class AgentDependencies:
    """Dependencies for agent execution."""
    config: dict[str, Any] = field(default_factory=dict)
    tools: list[str] = field(default_factory=list)
    other_agents: list[str] = field(default_factory=list)
    data_sources: list[str] = field(default_factory=list)
```

**Phase 3 Addition**:
```python
@dataclass
class AgentDependencies:
    """Dependencies for agent execution."""
    config: dict[str, Any] = field(default_factory=dict)
    tools: list[str] = field(default_factory=list)
    other_agents: list[str] = field(default_factory=list)
    data_sources: list[str] = field(default_factory=list)
    memory: MemoryProvider | None = None  # ← NEW: Inject memory provider
```

**Injection Pattern** (Hydra config):
```python
# In agent initialization
from DeepResearch.src.memory.factory import create_memory_provider

deps = AgentDependencies(
    config=cfg.agent,
    tools=cfg.tools,
    memory=create_memory_provider(cfg.memory)  # Factory creates Mem0Adapter or LettaAdapter
)

agent = BioinformaticsAgent(deps=deps)
```

---

### 6.2 ResearchState (from Phase 1)

**Location**: `DeepResearch/app.py`

**Current State** (Phase 1):
```python
@dataclass
class ResearchState:
    question: str
    plan: list[str] | None = field(default_factory=list)
    full_plan: list[dict[str, Any]] | None = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    answers: list[str] = field(default_factory=list)
    # ... other fields ...
    config: DictConfig | None = None
```

**Phase 3 Addition**:
```python
@dataclass
class ResearchState:
    question: str
    plan: list[str] | None = field(default_factory=list)
    full_plan: list[dict[str, Any]] | None = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    answers: list[str] = field(default_factory=list)
    # ... other fields ...
    config: DictConfig | None = None
    memory_session_id: str | None = None  # ← NEW: Workflow-level memory context
    memory_provider: MemoryProvider | None = None  # ← NEW: Workflow-scoped memory client
```

**Usage in Graph Nodes**:
```python
@dataclass
class Plan(BaseNode[ResearchState]):
    async def run(self, ctx: GraphRunContext[ResearchState]) -> NextNode:
        # Retrieve past plans from memory
        if ctx.state.memory_provider:
            past_plans = await ctx.state.memory_provider.search(
                query=ctx.state.question,
                user_id=ctx.state.memory_session_id or "default_user",
                agent_id="planner",
                filters={"type": "plan"},
                limit=3
            )
            ctx.state.notes.append(f"Retrieved {len(past_plans)} similar past plans")

        # ... planning logic ...

        # Store this plan in memory
        if ctx.state.memory_provider:
            await ctx.state.memory_provider.add(
                content=f"Plan for '{ctx.state.question}': {ctx.state.plan}",
                user_id=ctx.state.memory_session_id or "default_user",
                agent_id="planner",
                metadata={"type": "plan", "workflow_id": ctx.state.memory_session_id}
            )

        return Execute()
```

---

### 6.3 ExecutionHistory Migration (from Phase 1)

**Location**: `DeepResearch/src/datatypes/agents.py`

**Current ExecutionHistory** (Phase 1):
```python
@dataclass
class ExecutionHistory:
    """History of agent executions."""
    items: list[dict[str, Any]] = field(default_factory=list)

    def record(self, agent_type: AgentType, result: AgentResult, **kwargs):
        """Record an execution result."""
        self.items.append({
            "timestamp": time.time(),
            "agent_type": agent_type.value,
            "success": result.success,
            "execution_time": result.execution_time,
            "error": result.error,
            **kwargs
        })
```

**Phase 3 Migration Path**:

**Option A: Augment ExecutionHistory (Recommended for Phase 4A)**
```python
@dataclass
class ExecutionHistory:
    """History of agent executions (now persists to memory)."""
    items: list[dict[str, Any]] = field(default_factory=list)
    memory_provider: MemoryProvider | None = None  # ← NEW

    def record(self, agent_type: AgentType, result: AgentResult, **kwargs):
        """Record an execution result (in-memory + persistent memory)."""
        execution = {
            "timestamp": time.time(),
            "agent_type": agent_type.value,
            "success": result.success,
            "execution_time": result.execution_time,
            "error": result.error,
            **kwargs
        }
        self.items.append(execution)  # Keep in-memory list for backward compat

        # Also persist to memory system
        if self.memory_provider:
            asyncio.create_task(self._persist_to_memory(execution))

    async def _persist_to_memory(self, execution: dict):
        """Persist execution to memory asynchronously."""
        content = f"Agent {execution['agent_type']} executed with status {execution['success']}"
        if execution.get('error'):
            content += f" (Error: {execution['error']})"

        await self.memory_provider.add(
            content=content,
            user_id="system",
            agent_id=execution["agent_type"],
            metadata={
                "type": "execution_trace",
                "success": execution["success"],
                "execution_time": execution["execution_time"],
                **{k: v for k, v in execution.items() if k not in ["timestamp", "agent_type", "success", "execution_time", "error"]}
            }
        )
```

**Why This Approach**:
- ✅ Zero breaking changes - existing code using ExecutionHistory.record() works as-is
- ✅ Gradual migration - memory persistence added without touching 100+ callsites
- ✅ Backward compatible - in-memory `items` list still exists for legacy code
- ⏸️ Deprecate in Phase 5+ - once memory system proven, remove in-memory list

---

## 7. Configuration (Hydra)

**Location**: `configs/memory/mem0.yaml`

```yaml
memory:
  enabled: true
  provider: "mem0"  # Options: "mem0", "letta", "custom"

  # Mem0-specific config
  mem0:
    # Option 1: Self-hosted (recommended)
    host: "localhost"
    config:
      graph_store:
        provider: "neo4j"
        config:
          url: "${db.neo4j.uri}"
          username: "${db.neo4j.username}"
          password: "${db.neo4j.password}"
      vector_store:
        provider: "neo4j"  # Use Neo4j's vector index
        config:
          url: "${db.neo4j.uri}"
          username: "${db.neo4j.username}"
          password: "${db.neo4j.password}"

    # Option 2: Mem0 Cloud (fallback for testing)
    # api_key: "${oc.env:MEM0_API_KEY}"

  # Agent-specific settings
  agent_configs:
    bioinformatics_agent:
      search_limit: 10
      retention_days: 30
    prime_agent:
      search_limit: 5
      retention_days: 7
```

**Usage in Code**:
```python
from hydra import compose, initialize
from DeepResearch.src.memory.factory import create_memory_provider

with initialize(config_path="configs"):
    cfg = compose(config_name="config", overrides=["memory=mem0"])

    memory = create_memory_provider(cfg.memory)

    # Use in agent
    deps = AgentDependencies(memory=memory)
    agent = BioinformaticsAgent(deps=deps)
```

---

## 8. Implementation Plan (Phase 4 Vertical Slices)

### Phase 4A: Foundation + BioinformaticsAgent Pilot (1 week)

**Goal**: Prove memory works with ONE agent before scaling.

**Acceptance Criteria**:
- ✅ BioinformaticsAgent stores tool executions in memory
- ✅ BioinformaticsAgent retrieves past executions when asked "What did I do with P53?"
- ✅ Neo4j contains Entity/Memory nodes viewable in Neo4j Browser
- ✅ ExecutionHistory.record() persists to memory automatically

**Tasks**:
1. **Day 1-2: Core Implementation**
   - Create `DeepResearch/src/memory/` package
   - Implement `core.py` (MemoryProvider protocol)
   - Implement `adapters/mem0_adapter.py` (Mem0Adapter class)
   - Implement `factory.py` (create_memory_provider factory)
   - Add `configs/memory/mem0.yaml`

2. **Day 3-4: Integration**
   - Modify `AgentDependencies` to add `memory` field
   - Modify `ExecutionHistory` to persist to memory
   - Update `BioinformaticsAgent` to receive memory in deps
   - Add memory calls to BioinformaticsAgent tool execution hooks

3. **Day 5: Testing**
   - Unit tests: Mock MemoryProvider, verify agent calls it correctly
   - Integration tests: Use testcontainers Neo4j, verify nodes/edges created
   - Manual test: Run BioinformaticsAgent on "Find P53 targets", check Neo4j, ask "What did I do?"

**Rollback Plan**: If Mem0 fails (schema conflicts, performance issues):
- Remove `memory` from AgentDependencies
- Disable `ExecutionHistory` memory persistence
- Document failure reasons
- Pivot to Letta (Option B) in Phase 4B

**Success Metrics**:
- Memory add latency < 500ms
- Memory search returns relevant results (manual inspection)
- No exceptions in agent execution

---

### Phase 4B: Multi-Agent Rollout (3-5 days)

**Prerequisites**: Phase 4A complete, Mem0 works for BioinformaticsAgent

**Goal**: Enable memory for PRIMEAgent, PlannerAgent, ExecutorAgent.

**Acceptance Criteria**:
- ✅ All agents persist executions to memory
- ✅ Agent-specific memory namespacing works (BioinformaticsAgent sees only its memories)
- ✅ Cross-agent queries work (e.g., "What tools did ExecutorAgent use in workflow X?")

**Tasks**:
1. Add `memory` to all agent initialization points
2. Test agent-specific namespacing (`user_id:agent_id` pattern)
3. Add memory search to inter-agent coordination (if applicable)

**Rollback Plan**: Disable memory for specific agents if issues arise (config: `memory.agent_configs.X.enabled: false`)

---

### Phase 4C: Workflow-Level Memory (3-5 days)

**Prerequisites**: Phase 4B complete

**Goal**: Enable ResearchState-level memory (workflow-scoped context).

**Acceptance Criteria**:
- ✅ Pydantic Graph nodes (Plan, Execute, Analyze) use memory
- ✅ Workflow retrieves past similar workflows
- ✅ Workflow stores final results for future reference

**Tasks**:
1. Modify `ResearchState` to add `memory_session_id`, `memory_provider`
2. Initialize memory in Pydantic Graph entry point
3. Add memory calls to Plan node (retrieve past plans)
4. Add memory calls to Synthesize node (store final results)

---

### Phase 4D: Performance Tuning + Observability (2-3 days)

**Prerequisites**: Phase 4C complete

**Goal**: Ensure memory doesn't slow down workflows.

**Acceptance Criteria**:
- ✅ Memory add/search latency < 500ms (p95)
- ✅ Memory doesn't block agent execution (async operations)
- ✅ Observability: Memory usage tracked in logs/metrics

**Tasks**:
1. Add latency tracking to MemoryProvider methods
2. Add async background memory writes (fire-and-forget for non-critical operations)
3. Add logging: memory hits/misses, search relevance scores
4. Load testing: Run 100 workflows, measure memory overhead

---

## 9. Testing Strategy

### Unit Tests

**Location**: `tests/test_memory/`

```python
# test_mem0_adapter.py
async def test_mem0_adapter_add():
    mock_client = MagicMock()
    mock_client.add.return_value = {"id": "mem_123"}

    adapter = Mem0Adapter(mock_client)
    mem_id = await adapter.add(
        content="Test memory",
        user_id="user1",
        agent_id="agent1",
        metadata={"type": "test"}
    )

    assert mem_id == "mem_123"
    mock_client.add.assert_called_once()

# test_agent_integration.py
async def test_agent_uses_memory():
    mock_memory = MagicMock(spec=MemoryProvider)
    deps = AgentDependencies(memory=mock_memory)

    agent = BioinformaticsAgent(deps=deps)
    result = await agent.run(task="Find P53 targets")

    # Verify agent called memory.add() for execution trace
    mock_memory.add.assert_called()
    call_args = mock_memory.add.call_args
    assert "P53" in call_args.kwargs["content"]
    assert call_args.kwargs["metadata"]["type"] == "execution_trace"
```

### Integration Tests

**Location**: `tests/test_memory_integration/`

```python
# test_neo4j_persistence.py
@pytest.mark.containerized
async def test_memory_persists_to_neo4j():
    # Spin up Neo4j testcontainer
    with Neo4jContainer("neo4j:5.13") as neo4j:
        config = {
            "host": "localhost",
            "config": {
                "graph_store": {
                    "provider": "neo4j",
                    "config": {
                        "url": neo4j.get_connection_url(),
                        "username": "neo4j",
                        "password": neo4j.password
                    }
                }
            }
        }

        adapter = Mem0Adapter(config)

        # Add memory
        mem_id = await adapter.add(
            content="Test execution trace",
            user_id="system",
            agent_id="test_agent",
            metadata={"type": "execution_trace"}
        )

        # Verify Neo4j contains nodes
        with neo4j.get_driver() as driver:
            with driver.session() as session:
                result = session.run("MATCH (n:Memory {id: $id}) RETURN n", id=mem_id)
                assert result.single() is not None
```

### Performance Tests

**Location**: `tests/test_memory_performance/`

```python
# test_latency.py
@pytest.mark.performance
async def test_memory_add_latency():
    adapter = Mem0Adapter(get_test_config())

    latencies = []
    for i in range(100):
        start = time.time()
        await adapter.add(
            content=f"Test memory {i}",
            user_id="perf_test",
            agent_id="test_agent"
        )
        latencies.append(time.time() - start)

    p95_latency = sorted(latencies)[94]
    assert p95_latency < 0.5, f"p95 latency {p95_latency}s exceeds 500ms target"
```

---

## 10. Rollback Plan

### If Mem0 Fails in Phase 4A

**Symptoms**:
- Schema conflicts (Mem0 can't handle Agent/Workflow/Action structure)
- Unacceptable latency (>500ms for add/search)
- API limitations (Mem0's API doesn't support our use cases)

**Rollback Steps**:
1. Disable memory in config: `memory.enabled: false`
2. Document failure reasons in `AUDIT_PHASE2_PHASE3.md`
3. Decision point: Pivot to Letta (Option B) or defer memory to Phase 5

**Pivot to Letta** (Option B):
- Implement `adapters/letta_adapter.py` (similar to Mem0Adapter)
- Letta uses filesystem + embeddings (different architecture)
- Update `factory.py` to support `provider: "letta"`
- Re-run Phase 4A tests with LettaAdapter

**Defer to Phase 5**: If both Mem0 and Letta fail, defer memory system and focus on other priorities (multi-agent coordination, tool improvements, etc.)

---

## 11. Future Enhancements (Phase 5+)

### When to Add G-Memory Patterns (Option C)

**Trigger**: Phase 4D performance testing shows Mem0's retrieval is too slow OR multi-agent coordination requires hierarchical memory.

**Implementation**:
- Add `adapters/hybrid_adapter.py` that:
  - Uses Mem0 for conversational memory
  - Uses custom Neo4j logic for G-Memory hierarchy (Insight/Query/Interaction graphs)
- Route high-level goals → Insight graphs
- Route task-specific plans → Query graphs
- Route tool executions → Interaction graphs

**Effort Estimate**: 1-2 weeks (requires custom Cypher queries, graph traversal logic)

### When to Add Zep Patterns (Temporal Reasoning)

**Trigger**: Research domain requires tracking contradictions, retroactive updates (e.g., "Paper X was retracted, update all related memories").

**Implementation**:
- Add bi-temporal tracking (event_time vs. ingestion_time)
- Implement edge invalidation logic (newer facts override older facts)

**Effort Estimate**: 2-3 weeks (complex graph logic)

---

## 12. Success Criteria (Phase 3 → Phase 4 Readiness)

✅ **Phase 3 is ready to break down into Phase 4 vertical slices if**:
1. Architecture decision justified (pure Mem0 vs. hybrid) ✅
2. All Phase 1 integration points addressed (AgentDependencies, ResearchState, ExecutionHistory) ✅
3. MemoryProvider protocol defined (vendor-agnostic) ✅
4. Mem0Adapter implementation sketched (concrete, not vague) ✅
5. Configuration structure specified (Hydra YAML) ✅
6. Phase 4 vertical slices defined (4A/B/C/D with acceptance criteria) ✅
7. Testing strategy defined (unit, integration, performance) ✅
8. Rollback plan documented (if Mem0 fails, pivot to Letta) ✅

**This Phase 3 spec is IRON-CLAD and ready for Phase 4 breakdown.** ✅

---

## 13. Alignment with Meta-Plan

**Meta-Plan Philosophy**: "Ship iterative vertical slices instead of planning forever. Build, test, learn, iterate."

**How Phase 3 Aligns**:
- ✅ **Iterative**: Starts with simplest option (pure Mem0), defers complexity (hybrid) to Phase 5+
- ✅ **Testable**: Each Phase 4 slice has acceptance criteria + rollback plan
- ✅ **Low risk**: If Mem0 fails, pivot to Letta (Option B) without rewriting agents
- ✅ **Production-ready**: Mem0 is Y Combinator-backed, battle-tested
- ✅ **Shippable**: Phase 4A delivers working memory in 1 week (BioinformaticsAgent pilot)

**Phase 3 is the "marriage" of Phase 1 (baseline integration points) + Phase 2 (Mem0 research) + Meta-Plan (iterative shipping).** ✅

---

**Status**: ✅ **IRON-CLAD - Ready for Phase 4 Breakdown**
**Next Step**: Break Phase 4A/B/C/D into implementation tickets (TDD, SOLID, DRY, YAGNI, GOF principles)
**Sign-off**: User + Mario review → Proceed to Phase 4A implementation

---

**Related Documentation**:
- `PHASE_1_BASELINE_FOUNDATION.md` - Codebase integration points
- `PHASE_2_MEMORY_RESEARCH.md` - Memory system research (Mem0, Letta, Zep, G-Memory)
- `META_PLAN.MD` - 4-phase approach overview
- `AUDIT_PHASE2_PHASE3.md` - Audit findings (all issues resolved)
