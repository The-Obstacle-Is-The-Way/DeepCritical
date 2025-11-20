# Phase 4D: Pilot Execution & ExecutionHistory Tracing Spec

**Status**: 📝 Planned
**Dependency**: Phase 4C (Agent Wiring)
**Goal**: Prove the system's value by enabling the **Bioinformatics Agent** to persist its execution history (tool usage, results) into the long-term memory graph.

---

## 1. Objectives
- Bridge the `ExecutionHistory` datatype (from Phase 1) to the `MemoryProvider` (from Phase 4).
- Implement an "Interceptor" or "Hook" that automatically calls `memory.add()` whenever a tool execution completes.
- Run a "Live Fire" test: Use `BioinformaticsAgent` to run a real (or simulated) tool flow and verify the trace exists in memory.
- **G-Memory Pattern**: Structure these memories with metadata `type="trace"` and `workflow_id="..."`.

---

## 2. Codebase Modifications

### A. Execution Trace Interceptor
**File**: `DeepResearch/src/utils/execution_history.py` (or wherever the history recording logic lives)
**Change**:
- When `add_execution_log` is called:
    1. Check if `memory_provider` is available.
    2. If yes, fire-and-forget (async) a call to `memory.add()`.
    3. **Format**:
        - Content: `f"Executed {tool_name}: {status}"`
        - Metadata: `{ "type": "trace", "tool": tool_name, "result_summary": ..., "timestamp": ... }`

### B. Workflow Integration (Bioinformatics)
**File**: `DeepResearch/configs/bioinformatics/agents.yaml` (or similar)
**Change**:
- Ensure `memory_enabled: true` for this specific agent profile.

---

## 3. The "Live Fire" Test Plan

**Test File**: `DeepResearch/tests/memory/test_end_to_end_pilot.py`

1.  **Setup**:
    - Use `Mem0Adapter` (Mock or Real depending on CI environment).
    - Initialize `BioinformaticsAgent`.

2.  **Execution**:
    - Prompt: "Analyze the BRCA1 gene using the dummy tool."
    - Agent runs -> Calls Tool -> Returns Result.

3.  **Verification**:
    - Immediate Search: `memory.search("BRCA1 trace")`.
    - **Expectation**: Find a memory item corresponding to the tool execution.
    - **Metadata Check**: Verify `type="trace"` is present.

---

## 4. Implementation Steps

1.  **Locate Hook Point**:
    - Audit `DeepResearch/src/utils/execution_history.py` to find the exact write method.

2.  **Implement Bridge**:
    - Add the logic to serialize `ExecutionHistory` items into a format suitable for `memory.add()`.

3.  **Run Pilot**:
    - Create a script `scripts/test_memory_pilot.py` that sets up the environment and runs the agent.

4.  **Analyze Results**:
    - Inspect the memory store.
    - Confirm the trace is retrievable.

---

## 5. Acceptance Criteria
- [ ] `ExecutionHistory` events are automatically mirrored to the memory system.
- [ ] Metadata correctly tags these as `type: trace`.
- [ ] `BioinformaticsAgent` runs successfully with memory enabled.
- [ ] End-to-end test proves data round-trip (Execution -> Memory -> Retrieval).
- [ ] **Phase 4 Complete**: The system is now live and capable of self-recording its actions.

---

**Status**: Ready to begin implementation of Phase 4A.
