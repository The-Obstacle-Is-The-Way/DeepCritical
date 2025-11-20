# Phase 4C: Agent Wiring & Middleware Spec

**Status**: 📝 Planned
**Dependency**: Phase 4B (Mem0 Adapter)
**Goal**: "Surgically" inject the memory system into the DeepResearch agent architecture (`Pydantic AI` + `Pydantic Graph`) and the Executor context (`WorkflowDAG`).

---

## 1. Objectives
- **Double Wiring**: Inject `MemoryProvider` into:
    1. `AgentDependencies` (for Pydantic AI Agents).
    2. `ExecutionContext` (for PRIME/Bioinformatics Workflow Executors).
- **ResearchState Update**: Add `memory_context` to persist relevant memories across graph nodes.
- **Memory Tool**: Create a tool that Agents can use to actively recall information.
- **Backward Compatibility**: Ensure all agents work if `memory=None`.

---

## 2. Codebase Modifications

### A. Agent Dependencies (CRITICAL FIX)
**File**: `DeepResearch/src/datatypes/agents.py`
**Change**:
```python
@dataclass
class AgentDependencies:
    config: dict[str, Any] = field(default_factory=dict)
    tools: list[str] = field(default_factory=list)
    other_agents: list[str] = field(default_factory=list)
    data_sources: list[str] = field(default_factory=list)
    memory: Any | None = None  # Type is Any to avoid circular imports, MemoryProvider at runtime
    user_id: str | None = None  # CRITICAL: Required for namespace isolation
    agent_id: str | None = None  # CRITICAL: Required for namespace isolation
```

**Why This Is Critical**:
Without `user_id` and `agent_id`, the adapter cannot enforce namespace isolation (`user_id:agent_id`). All agents would share the same memory pool, breaking the architecture.

### B. Execution Context (Crucial Fix)
**File**: `DeepResearch/src/datatypes/execution.py`
**Change**:
```python
@dataclass
class ExecutionContext:
    workflow: WorkflowDAG
    history: ExecutionHistory
    data_bag: dict[str, Any] = field(default_factory=dict)
    current_step: int = 0
    max_retries: int = 3
    manual_confirmation: bool = False
    adaptive_replanning: bool = True
    memory: Any | None = None  # MemoryProvider for automated tracing
    workflow_id: str | None = None  # CRITICAL: Required for trace namespacing
    agent_id: str | None = None  # CRITICAL: Required for trace namespacing
```

**Why This Is Critical**:
ExecutionHistory needs `workflow_id` and `agent_id` to call `add_trace()`. Without these, traces cannot be properly attributed.

### C. Research State
**File**: `DeepResearch/DeepResearch/app.py`
**Change**:
```python
@dataclass
class ResearchState:
    # ... existing ...
    memory_context: list[dict] = field(default_factory=list) # Snapshot of memories
```

### D. Agent Orchestrator / Factory
**File**: `DeepResearch/src/agents/agent_orchestrator.py`

**Approach**: Dependency Injection (recommended for testability)

**Logic**:
1. Add `memory_provider: MemoryProvider | None = None` field to `AgentOrchestrator` dataclass.
2. In agent creation methods, inject memory into `AgentDependencies`:
```python
deps = AgentDependencies(
    config=...,
    memory=self.memory_provider  # Inject here
)
```
3. In app initialization (e.g., `DeepResearch/app.py`):
```python
from DeepResearch.src.memory.factory import get_memory_provider

memory_provider = get_memory_provider(cfg.memory) if cfg.memory.enabled else None
orchestrator = AgentOrchestrator(
    config=...,
    memory_provider=memory_provider
)
```

**Alternative**: Global singleton (less testable, not recommended for Phase 4):
```python
# In each agent creation:
from DeepResearch.src.memory.factory import get_memory_provider
deps = AgentDependencies(memory=get_memory_provider(cfg))
```

### E. Memory Tools
**File**: `DeepResearch/src/tools/memory_tools.py`

**Complete Implementation**:
```python
from pydantic_ai import Agent, RunContext

from DeepResearch.src.datatypes.agents import AgentDependencies


def register_memory_tools(agent: Agent[AgentDependencies, str]) -> None:
    """Register memory tools on a Pydantic AI agent."""

    @agent.tool_plain
    async def recall_memory(
        ctx: RunContext[AgentDependencies], query: str, limit: int = 5
    ) -> dict[str, object]:
        """Search long-term memory for relevant information.

        Args:
            query: Search query string
            limit: Maximum number of results

        Returns:
            {"results": [...]} or {"results": []} if memory unavailable
        """
        if ctx.deps.memory is None or ctx.deps.user_id is None or ctx.deps.agent_id is None:
            return {"results": []}

        hits = await ctx.deps.memory.search(
            query, user_id=ctx.deps.user_id, agent_id=ctx.deps.agent_id, limit=limit
        )
        return {"results": [hit.model_dump() for hit in hits]}

    @agent.tool_plain
    async def save_note(
        ctx: RunContext[AgentDependencies],
        content: str,
        metadata: dict[str, object] | None = None,
    ) -> dict[str, str]:
        """Save a note to long-term memory.

        Args:
            content: Note content
            metadata: Optional metadata tags

        Returns:
            {"id": "mem_123"} or {"id": ""} if memory unavailable
        """
        if ctx.deps.memory is None or ctx.deps.user_id is None or ctx.deps.agent_id is None:
            return {"id": ""}

        memory_id = await ctx.deps.memory.add(
            content=content,
            user_id=ctx.deps.user_id,
            agent_id=ctx.deps.agent_id,
            metadata=metadata,
        )
        return {"id": memory_id}
```

**Key Features**:
- Safe fallbacks when memory disabled (`memory=None`)
- Uses agent's `user_id` and `agent_id` for namespace isolation
- Proper async signatures
- Clear docstrings for agent's tool use

---

## 3. TDD Strategy

1.  **Wiring Test**:
    - **File**: `DeepResearch/tests/memory/test_wiring.py`
    - Instantiate `AgentDependencies` with `MockMemoryAdapter`.
    - Instantiate `ExecutionContext` with `MockMemoryAdapter`.
    - Verify fields are accessible.

2.  **Tool Test**:
    - Create a dummy `pydantic_ai.Agent` with `memory_tools`.
    - Run it: "Recall P53 info".
    - Assert `MockAdapter.search` was called.

3.  **Orchestrator Test**:
    - Mock the Agent Factory.
    - Verify that created agents receive the memory provider if config enabled.

---

## 4. Implementation Steps

1.  **Update Datatypes**: `agents.py` and `execution.py`.
2.  **Update App State**: `app.py`.
3.  **Create Tool**: `src/tools/memory_tools.py`.
4.  **Update Factory**: Modify `src/agents/agent_orchestrator.py` to inject memory.
5.  **Tests**: Verify wiring.

---

## 5. Acceptance Criteria
- [ ] `AgentDependencies` has `memory` field.
- [ ] `ExecutionContext` has `memory` field.
- [ ] `ResearchState` has `memory_context`.
- [ ] Agents can successfully call `recall_memory`.
- [ ] Existing agents run without crashing (optional memory).

---

**Next Phase**: 4D (Pilot Execution Tracing)