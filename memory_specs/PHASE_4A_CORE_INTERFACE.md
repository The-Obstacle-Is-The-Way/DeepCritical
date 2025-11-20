# Phase 4A: Core Interface & Mock Adapter Implementation Spec

**Status**: 📝 Planned
**Dependency**: Phase 3 Implementation Spec
**Goal**: Establish the vendor-agnostic `MemoryProvider` protocol and a robust `MockAdapter` to enable safe, test-driven development of the memory system without external dependencies.

---

## 1. Objectives
- Define the **SSOT** (Single Source of Truth) interfaces for all memory operations.
- Implement strictly typed data models (`MemoryItem`, `MemoryFilter`) ensuring no leakage of Mem0/Neo4j implementation details into the agent layer.
- Create a `MockMemoryAdapter` that passes all protocol tests, serving as the foundation for unit testing agents in later phases.
- **Zero External Dependencies**: This slice must run without Neo4j, Docker, or API keys.

---

## 2. New Files & Locations

### A. Core Interfaces
**File**: `DeepResearch/src/memory/core.py`
**Responsibility**: Defines the `Protocol` and Pydantic models.

**Required Imports**:
```python
from typing import Protocol, Optional, Any
from datetime import datetime
from pydantic import BaseModel
```

**Key Components**:
- `class MemoryItem(BaseModel)`: Standardized return shape.
    - `id`: str
    - `content`: str
    - `score`: Optional[float]
    - `metadata`: dict[str, Any]
    - `created_at`: Optional[datetime]
    - `agent_id`: Optional[str]
    - `user_id`: Optional[str]
- `class MemoryProvider(Protocol)`: The interface ensuring strict adherence to the Ports & Adapters pattern.
    - `async def add(self, content: str, user_id: str, agent_id: str, metadata: Optional[dict] = None) -> str`: Unstructured add.
    - `async def add_trace(self, agent_id: str, workflow_id: str, trace_data: dict) -> str`: Structured trace logging.
    - `async def search(self, query: str, user_id: str, agent_id: str, limit: int = 5, filters: Optional[dict] = None) -> list[MemoryItem]`: Hybrid search.
    - `async def get_all(self, user_id: str, agent_id: str, limit: int = 10, filters: Optional[dict] = None) -> list[MemoryItem]`: Retrieval.
    - `async def delete(self, memory_id: str) -> bool`: Cleanup.
    - `async def reset(self) -> bool`: Full wipe (for testing).

### B. Mock Implementation
**File**: `DeepResearch/src/memory/adapters/mock_adapter.py`
**Responsibility**: In-memory implementation of `MemoryProvider` using a simple `list[dict]`.
**Key Features**:
- Simulates "user_id:agent_id" namespacing.
- Basic substring matching for `search()` to verify retrieval logic.
- Metadata filtering logic to mirror real database queries (e.g. `type="trace"`).
- `add_trace` implementation that wraps the `trace_data` into a standardized JSON string content for "searchability" while keeping raw data in metadata.

### C. Factory/Registry
**File**: `DeepResearch/src/memory/factory.py`
**Responsibility**: Central entry point to get a memory provider.
**Logic**:
- `get_memory_provider(config: DictConfig) -> MemoryProvider`
- Currently only supports `provider: "mock"`.
- Raises `NotImplementedError` for "mem0" (until Phase 4B).

---

## 3. TDD Strategy (Test First)

**Test File**: `DeepResearch/tests/memory/test_core_interface.py`

### Test Cases:
1.  **Protocol Compliance**:
    - Verify `MockMemoryAdapter` explicitly implements `MemoryProvider`.
    - Static type check (ty) verification.

2.  **CRUD Operations**:
    - `test_add_memory()`: Add an item, verify it exists in the internal list.
    - `test_get_all()`: Retrieve all items for a specific `user_id:agent_id`. Verify isolation (Agent A shouldn't see Agent B's memories).
    - `test_delete()`: Remove an item, verify it's gone.

3.  **Filtering & Search**:
    - `test_search_substring()`: Add "Bioinformatics P53", search "P53", expect result.
    - `test_metadata_filtering()`: Add items with `{type: "trace"}` and `{type: "chat"}`. Filter by `type="trace"`.
    
4.  **Tracing**:
    - `test_add_trace()`: Verify `add_trace` correctly formats the memory item and sets `type="trace"` in metadata.

---

## 4. Implementation Steps

1.  **Scaffold Directory**:
    - `mkdir -p DeepResearch/src/memory/adapters`
    - `touch DeepResearch/src/memory/__init__.py`

2.  **Write Tests**:
    - Create `DeepResearch/tests/memory/test_core_interface.py` with failing tests (Red).

3.  **Implement Core**:
    - Write `DeepResearch/src/memory/core.py` defining the Protocol and Models.

4.  **Implement Mock**:
    - Write `DeepResearch/src/memory/adapters/mock_adapter.py` to pass the tests (Green).

5.  **Implement Factory**:
    - Write `DeepResearch/src/memory/factory.py`.

6.  **Verify**:
    - Run `pytest DeepResearch/tests/memory/test_core_interface.py`.
    - Run `uvx ty check DeepResearch/src/memory`.

---

## 5. Acceptance Criteria
- [ ] `DeepResearch/src/memory/core.py` exists with `MemoryProvider` Protocol.
- [ ] `MockMemoryAdapter` is implemented and passes strict type checking.
- [ ] Unit tests provide 100% coverage for the Mock adapter.
- [ ] No import errors or circular dependencies with existing code.
- [ ] **NO** Mem0 or Neo4j code is introduced in this phase.

---

**Next Phase**: 4B (Mem0 Adapter Implementation)