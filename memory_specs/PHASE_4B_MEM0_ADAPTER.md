# Phase 4B: Mem0 Adapter & Neo4j Configuration Spec

**Status**: 📝 Planned
**Dependency**: Phase 4A (Core Interface)
**Goal**: Implement the `Mem0Adapter` that connects the `MemoryProvider` interface to the real Mem0 SDK and our existing Neo4j infrastructure.

---

## 1. Objectives
- Implement `Mem0Adapter` using the official `mem0ai` SDK.
- **Reuse Existing Config**: Extract Neo4j credentials/URI directly from `DeepResearch/src/vector_stores/neo4j_config.py` structure or Hydra config to avoid duplication.
- **Handle Dual Modes**: Support `oss` (Neo4j) and `cloud` (Mem0 Platform).
- **Safe Normalization**: Convert Mem0's variable response formats (list vs dict) into strict `MemoryItem` objects.
- **Integration Testing**: Verify with `testcontainers-neo4j`.

---

## 2. New Files & Locations

### A. Mem0 Adapter
**File**: `DeepResearch/src/memory/adapters/mem0_adapter.py`
**Responsibility**: Real implementation of `MemoryProvider`.
**Key Logic**:
- **Init**:
    - Load config.
    - If `mode == "oss"`:
        - Construct `mem0_config` dictionary.
        - Map `graph_store`: `provider: "neo4j"`, config from Hydra `db.neo4j`.
        - Map `vector_store`: `provider: "neo4j"`, config from Hydra `db.neo4j`.
        - Instantiate `self.client = Memory.from_config(mem0_config)`.
    - If `mode == "cloud"`:
        - Instantiate `self.client = MemoryClient(api_key=..., ...)`
- **Methods**:
    - `add()`: Wraps `self.client.add()`.
    - `add_trace()`: JSON dumps the trace data into `content`, sets `metadata={"type": "trace", ...}`.
    - `search()`: Wraps `self.client.search()`. Calls `_normalize_response()`.
    - `get_all()`: Wraps `self.client.get_all()`.
    - `delete()`: Wraps `self.client.delete()`.
- **Normalization Helper**:
    - Handles `{"results": [...]}` vs `[...]`.
    - Handles missing keys gracefully.

### B. Integration Tests
**File**: `DeepResearch/tests/memory/test_mem0_integration.py`
**Responsibility**: Prove it works with real Neo4j.
**Tools**: `testcontainers` (specifically `neo4j` container).
**Critical Check**:
- Verify that `add_trace()` creates a node in Neo4j.
- Verify that `search()` can find that node.

---

## 3. Configuration Updates (Hydra)

### A. Memory Config File
**File**: `DeepResearch/configs/memory/default.yaml` (New)
**Content**:
```yaml
memory:
  enabled: true
  provider: "mem0"
  mode: "oss" # or "cloud"

  # OSS Configuration (Maps to existing db.neo4j)
  oss:
    graph_store:
      provider: "neo4j"
      config:
        url: "${db.neo4j.uri}"
        username: "${db.neo4j.username}"
        password: "${db.neo4j.password}"
    vector_store:
      provider: "neo4j"
      config:
        url: "${db.neo4j.uri}"
        username: "${db.neo4j.username}"
        password: "${db.neo4j.password}"
        embedding_model_dims: 1536 # Match existing
  
  # Cloud Configuration
  cloud:
    api_key: "${oc.env:MEM0_API_KEY}"
```

### B. Wire into Main Config
**File**: `DeepResearch/configs/config.yaml`
**Change**: Add `memory: default` to the `defaults:` list:
```yaml
defaults:
  - challenge: default
  - workflow_orchestration: default
  - db: neo4j
  - memory: default  # ← ADD THIS LINE
  - statemachines/flows: prime
  - _self_
```

This allows memory config to be composed via Hydra and overridden at CLI:
```bash
# Enable memory with OSS mode
uv run deepresearch memory.enabled=true memory.mode=oss

# Override provider
uv run deepresearch memory.provider=mock
```

---

## 4. TDD Strategy

1.  **Mocking Mem0 (Unit Test)**:
    - **File**: `DeepResearch/tests/memory/test_mem0_adapter_unit.py`
    - Mock `mem0.Memory` class.
    - Test config construction logic (ensure it pulls from Hydra correctly).
    - Test `add_trace` serialization.

2.  **Live Neo4j (Integration Test)**:
    - **File**: `DeepResearch/tests/memory/test_mem0_integration.py`
    - Spin up Neo4j container.
    - Instantiate `Mem0Adapter`.
    - `await adapter.add("test memory", "u1", "a1")`.
    - `results = await adapter.search("test", "u1", "a1")`.
    - Assert `len(results) > 0`.

---

## 5. Implementation Steps

1.  **Dependencies**: `uv add mem0ai neo4j`.
2.  **Config**: Create `DeepResearch/configs/memory/default.yaml`.
3.  **Code**: Implement `DeepResearch/src/memory/adapters/mem0_adapter.py`.
4.  **Factory**: Update `DeepResearch/src/memory/factory.py` to support `provider="mem0"`.
5.  **Tests**: Write and run unit & integration tests.

---

## 6. Acceptance Criteria
- [ ] `Mem0Adapter` implemented.
- [ ] Adapter correctly normalizes Mem0 responses.
- [ ] `add_trace` successfully stores structured data as searchable text + metadata.
- [ ] Integration test passes with real Neo4j container.
- [ ] No hardcoded credentials.

---

**Next Phase**: 4C (Agent Wiring & Middleware)