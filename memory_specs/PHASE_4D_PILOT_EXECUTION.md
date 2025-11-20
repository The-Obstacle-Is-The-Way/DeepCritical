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
**Changes**:
1.  Update `__init__`: Accept optional `memory_provider`.
2.  Update `add_item(item: ExecutionItem)`:
    - If `self.memory_provider` is set:
        - Serialize `item` to dict.
        - Call `self.memory_provider.add_trace(...)` (Fire & Forget / Async).

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

**Status**: Ready to begin implementation of Phase 4A.