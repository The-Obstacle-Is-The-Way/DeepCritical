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

### A. Agent Dependencies
**File**: `DeepResearch/src/datatypes/agents.py`
**Change**:
```python
@dataclass
class AgentDependencies:
    # ... existing ...
    memory: Any | None = None # Type is Any to avoid circular imports, strictly MemoryProvider at runtime
```

### B. Execution Context (Crucial Fix)
**File**: `DeepResearch/src/datatypes/execution.py`
**Change**:
```python
@dataclass
class ExecutionContext:
    # ... existing ...
    memory: Any | None = None # Inject memory here for automated tracing
```

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

### E. Memory Tool
**File**: `DeepResearch/src/tools/memory_tools.py`
**Logic**:
- `recall_memory(query: str)`: Uses `ctx.deps.memory.search()`.
- `save_note(content: str)`: Uses `ctx.deps.memory.add()`.

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