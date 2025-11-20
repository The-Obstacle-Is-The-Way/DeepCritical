# Phase 4D: Pilot Execution & ExecutionHistory Tracing Spec

**Status**: 📝 Planned
**Dependency**: Phase 4C (Agent Wiring)
**Goal**: Prove the system's value by enabling the **Bioinformatics Agent** to automatically persist its execution history (`ExecutionHistory` Complex Object) into long-term memory.

---

## 1. Objectives
- **Target the Right History**: Modify `DeepResearch/src/utils/execution_history.py` (The Complex History), *not* the simple list in `agents.py`.
- **Interceptor Pattern**: Add a `memory_provider` field to `ExecutionHistory`. If present, `add_item()` triggers a background write to memory.
- **Live Pilot**: Run `BioinformaticsAgent`. Verify that its tool calls (e.g., `run_blast`, `fetch_sequence`) appear in Neo4j.
- **G-Memory Structure**: Ensure traces use `type="trace"` metadata.

---

## 2. Codebase Modifications

### A. Execution History Interceptor
**File**: `DeepResearch/src/utils/execution_history.py`

**Validation**: ExecutionHistory already implements persistence (lines 146-179):
- ✅ `to_dict()` - Serialization (line 130)
- ✅ `from_dict()` - Deserialization (line 138)
- ✅ `save_to_file()` - JSON persistence (line 146)
- ✅ `load_from_file()` - JSON restore (line 158)

**Our Enhancement**: Add memory_provider hook to ALSO persist to Mem0 (backward compatible).

**Changes**:
1.  Update `__init__`: Accept optional `memory_provider` and `use_file_lock`.
2.  Update `add_item(item: ExecutionItem)`:
    - If `self.memory_provider` is set:
        - Serialize `item` to dict.
        - Call `self.memory_provider.add_trace(...)` (Fire & Forget / Async).
        - Optional: Use FileLock for concurrent access safety.

### B. Executor Wiring
**File**: `DeepResearch/src/agents/prime_executor.py` (or whichever executor uses this history)
**Changes**:
- When initializing `ExecutionHistory`, pass the `ctx.memory` from `ExecutionContext` (added in 4C).

---

## 3. The "Live Fire" Test Plan

**Test File**: `DeepResearch/tests/memory/test_end_to_end_pilot.py`

1.  **Setup**:
    - Instantiate `MockMemoryAdapter` (for unit test speed) OR `Mem0Adapter` (for integration).
    - Create `ExecutionContext` with this memory.
    - Initialize `ExecutionHistory` with this memory.

2.  **Simulate Execution**:
    - `history.add_item(ExecutionItem(tool="blast", status="success", result={"e_value": 0.0}))`

3.  **Verification**:
    - `memories = await memory.search("blast", ...)`
    - Assert `len(memories) == 1`.
    - Assert `memories[0].metadata["type"] == "trace"`.
    - Assert `memories[0].metadata["tool"] == "blast"`.

---

## 4. Implementation Steps

1.  **Modify ExecutionHistory**: Add the optional provider and the hook in `add_item`.
2.  **Modify Executor**: Ensure the provider flows from Context -> History.
3.  **Write Pilot Test**: `test_end_to_end_pilot.py`.
4.  **Run Pilot**: Execute the test.

---

## 5. Acceptance Criteria
- [ ] `ExecutionHistory` in `src/utils` accepts a memory provider.
- [ ] Adding an item to history triggers a memory write.
- [ ] Trace data is searchable.
- [ ] Original functionality of `ExecutionHistory` (metrics, file save) remains untouched.
- [ ] **End-to-End**: A simulated tool run results in a verifiable memory entry.

---

## 6. Optional Enhancement: Thread Safety with FileLock

**Source**: Existing pattern in `DeepResearch/src/utils/analytics.py:83`

For concurrent workflows writing to same ExecutionHistory:

```python
from filelock import FileLock
from pathlib import Path
import asyncio

@dataclass
class ExecutionHistory:
    items: list[ExecutionItem] = field(default_factory=list)
    memory_provider: MemoryProvider | None = None
    use_file_lock: bool = False  # Enable for concurrent workflows
    lock_file: Path | None = None

    def add_item(self, item: ExecutionItem) -> None:
        """Add item with optional thread-safe memory persistence."""
        self.items.append(item)

        if self.memory_provider:
            asyncio.create_task(self._persist_to_memory(item))

    async def _persist_to_memory(self, item: ExecutionItem) -> None:
        """Persist to memory with optional file lock."""
        execution_dict = {
            "step_name": item.step_name,
            "tool": item.tool,
            "status": item.status.value,
            "result": item.result,
            "error": item.error,
            "timestamp": item.timestamp,
            "parameters": item.parameters,
            "duration": item.duration,
            "retry_count": item.retry_count
        }

        # Optional: FileLock for concurrent access
        if self.use_file_lock and self.lock_file:
            with FileLock(str(self.lock_file)):
                await self.memory_provider.add_trace(
                    agent_id=self._agent_id,
                    workflow_id=self._workflow_id,
                    trace_data=execution_dict
                )
        else:
            await self.memory_provider.add_trace(
                agent_id=self._agent_id,
                workflow_id=self._workflow_id,
                trace_data=execution_dict
            )
```

**Configuration**:
```python
# In executor initialization
history = ExecutionHistory(
    memory_provider=ctx.memory,
    use_file_lock=cfg.memory.persistence.use_file_lock,  # From config
    lock_file=Path(".execution_history.lock")
)
```

**When to Use**:
- **Phase 4 Pilot**: NOT NEEDED (single BioinformaticsAgent)
- **Phase 5 Multi-Agent**: RECOMMENDED (concurrent workflows)

**Reference**: `DeepResearch/src/utils/analytics.py:78-97` for proven FileLock pattern

---

**Status**: Ready to begin implementation of Phase 4A.