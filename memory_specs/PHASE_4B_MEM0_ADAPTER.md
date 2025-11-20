# Phase 4B: Mem0 Integration & Neo4j Wiring Spec

**Status**: 📝 Planned
**Dependency**: Phase 4A (Core Interface)
**Goal**: Implement the `Mem0Adapter` that connects the `MemoryProvider` interface to the real Mem0 SDK and our existing Neo4j infrastructure.

---

## 1. Objectives
- Implement `Mem0Adapter` using the official `mem0ai` SDK.
- Integrate with the **existing** Neo4j instance defined in `DeepResearch/src/vector_stores/neo4j_vector_store.py` logic (reusing credentials/config).
- Support both **OSS Mode** (local Neo4j via `Memory.from_config`) and **Cloud Mode** (SaaS via `MemoryClient`).
- Normalize Mem0's variable response formats (list vs dict) to match our strict `MemoryItem` model.
- Verify connection using `testcontainers` (Dockerized Neo4j) to ensure CI/CD compatibility.

---

## 2. New Files & Locations

### A. Mem0 Adapter
**File**: `DeepResearch/src/memory/adapters/mem0_adapter.py`
**Responsibility**: Real implementation of `MemoryProvider`.
**Key Logic**:
- **Init**: Accepts Hydra config. Determines `oss` vs `cloud` mode.
- **OSS Mode**: Constructs Mem0 config dict using `db.neo4j` values from our existing config.
- **Wrappers**:
    - `add()`: Calls `mem0.add()`.
    - `search()`: Calls `mem0.search()`, normalizes response.
    - `get_all()`: Calls `mem0.get_all()`.
- **Normalization**: Handles `{results: [...]}` vs `[...]` discrepancy.

### B. Integration Tests
**File**: `DeepResearch/tests/memory/test_mem0_integration.py`
**Responsibility**: Prove it works with real Neo4j.
**Tools**: `testcontainers` (specifically `neo4j` container).

---

## 3. Configuration Updates (Hydra)

**File**: `DeepResearch/configs/memory/default.yaml` (New)
**Content**:
```yaml
memory:
  enabled: true
  provider: "mem0"
  mode: "oss" # or "cloud"
  
  # Link to existing db config
  neo4j_ref:
    uri: "${db.neo4j.uri}"
    username: "${db.neo4j.username}"
    password: "${db.neo4j.password}"
```

---

## 4. TDD Strategy

1.  **Mocking Mem0 (Unit Test)**:
    - Test `Mem0Adapter` logic *without* a real database by mocking the `mem0.Memory` class.
    - Verify it correctly constructs the config dict from our Hydra config.
    - Verify response normalization logic handles edge cases.

2.  **Live Neo4j (Integration Test)**:
    - Spin up Neo4j container.
    - Instantiate `Mem0Adapter` pointing to container.
    - Perform Add -> Search -> Get All loop.
    - **Crucial**: Verify data actually persists in Neo4j (using a direct Neo4j driver to inspect nodes).

---

## 5. Implementation Steps

1.  **Dependencies**:
    - `uv add mem0ai neo4j` (if not present).

2.  **Write Configuration**:
    - Add `DeepResearch/configs/memory/default.yaml`.

3.  **Write Adapter**:
    - Implement `DeepResearch/src/memory/adapters/mem0_adapter.py`.
    - Ensure strict mapping of `MemoryItem` fields (handling missing/null values safely).

4.  **Update Factory**:
    - Update `DeepResearch/src/memory/factory.py` to handle `provider: "mem0"`.

5.  **Write Integration Test**:
    - Create `DeepResearch/tests/memory/test_mem0_integration.py`.
    - Use `testcontainers` to verify end-to-end flow.

---

## 6. Acceptance Criteria
- [ ] `Mem0Adapter` implemented and adheres to `MemoryProvider` protocol.
- [ ] Hydra configuration correctly maps existing `db.neo4j` vars to Mem0 config.
- [ ] Integration tests pass with a real Dockerized Neo4j instance.
- [ ] Adapter gracefully handles Mem0 response variations.
- [ ] No hardcoded credentials; strictly uses Hydra config.

---

**Next Phase**: 4C (Agent Wiring & Middleware)
