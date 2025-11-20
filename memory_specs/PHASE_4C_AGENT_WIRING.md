# Phase 4C: Agent Wiring & Middleware Spec

**Status**: 📝 Planned
**Dependency**: Phase 4B (Mem0 Adapter)
**Goal**: "Surgically" inject the memory system into the DeepResearch agent architecture (`Pydantic AI` + `Pydantic Graph`) without breaking existing flows.

---

## 1. Objectives
- Modify `AgentDependencies` to carry the `MemoryProvider` instance.
- Update `ResearchState` to support holding memory context (snapshots).
- Create `MemoryMiddleware` to automate memory interactions (optional auto-save/retrieval hooks).
- Ensure backward compatibility: Agents must function even if memory provider is `None` or `Mock`.

---

## 2. Codebase Modifications

### A. Agent Dependencies
**File**: `DeepResearch/src/datatypes/agents.py`
**Change**:
```python
@dataclass
class AgentDependencies:
    # ... existing fields ...
    memory: Optional[MemoryProvider] = None # New field
```

### B. Research State
**File**: `DeepResearch/app.py` (or wherever `ResearchState` is defined)
**Change**:
```python
@dataclass
class ResearchState:
    # ... existing fields ...
    memory_context: list[MemoryItem] = field(default_factory=list) # Snapshot of relevant memories
```

### C. Agent Factory Update
**File**: `DeepResearch/src/agents/agent_orchestrator.py` (and others)
**Change**:
- Update the logic that instantiates agents to retrieve the `MemoryProvider` from the Factory (created in 4A) and pass it into `AgentDependencies`.

### D. Memory Tool (The "Hook")
**File**: `DeepResearch/src/tools/memory_tools.py` (New)
**Responsibility**: Expose memory as a Pydantic AI tool.
**Functions**:
- `recall_memory(query: str, filter_type: str = None)`: Wraps `provider.search()`.
- `save_note(content: str)`: Wraps `provider.add()`.

---

## 3. TDD Strategy

1.  **Dependency Injection Test**:
    - Create a test agent with `AgentDependencies`.
    - Inject `MockMemoryAdapter` (from 4A).
    - Verify the agent can access `ctx.deps.memory`.

2.  **Tool Execution Test**:
    - Attach `recall_memory` tool to a simple `pydantic_ai.Agent`.
    - Run the agent with a prompt: "Recall what I said about P53."
    - Verify the tool calls `provider.search()` on the mock adapter.

3.  **State Persistence Test**:
    - Initialize `ResearchState`.
    - Manually populate `memory_context`.
    - Verify `Pydantic Graph` nodes can access this context.

---

## 4. Implementation Steps

1.  **Update Datatypes**:
    - Modify `DeepResearch/src/datatypes/agents.py` to add the memory field.

2.  **Create Tools**:
    - Write `DeepResearch/src/tools/memory_tools.py`.

3.  **Update Agent Initialization**:
    - Modify the orchestrator/server logic to fetch memory from the factory and inject it.
    - **Crucial**: Wrap this in a `try/except` or config check to allow disabling memory easily.

4.  **Verify**:
    - Run existing agent tests to ensure no regressions (pass `memory=None` where needed).
    - Run new wiring tests.

---

## 5. Acceptance Criteria
- [ ] `AgentDependencies` includes optional `MemoryProvider`.
- [ ] `ResearchState` includes `memory_context`.
- [ ] Agents can successfully call `recall_memory` tool when enabled.
- [ ] Existing tests pass (backward compatibility verified).
- [ ] No changes to the core logic of existing tools (bioinformatics modules).

---

**Next Phase**: 4D (Pilot Execution Tracing)
