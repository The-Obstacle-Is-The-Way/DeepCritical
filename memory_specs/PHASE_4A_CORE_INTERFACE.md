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

**Complete Implementation**:
```python
from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class MemoryFilter(BaseModel):
    """Optional filter container for metadata-based queries."""

    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryItem(BaseModel):
    """Normalized memory record returned by providers."""

    model_config = ConfigDict(extra="allow")

    id: str
    content: str
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None
    agent_id: str | None = None
    user_id: str | None = None


@runtime_checkable
class MemoryProvider(Protocol):
    """Vendor-agnostic memory interface (Ports & Adapters pattern)."""

    async def add(
        self,
        content: str,
        user_id: str,
        agent_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Add unstructured memory. Returns memory_id."""
        ...

    async def add_trace(
        self,
        agent_id: str,
        workflow_id: str,
        trace_data: dict[str, Any],
        user_id: str = "system",
    ) -> str:
        """Add structured execution trace. Returns memory_id."""
        ...

    async def search(
        self,
        query: str,
        user_id: str,
        agent_id: str,
        limit: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        """Hybrid search with namespace isolation."""
        ...

    async def get_all(
        self,
        user_id: str,
        agent_id: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        """Retrieve all memories for agent with optional filters."""
        ...

    async def delete(self, memory_id: str) -> bool:
        """Delete memory by ID. Returns True if deleted."""
        ...

    async def reset(self) -> bool:
        """Full wipe (testing only). Returns True if successful."""
        ...
```

**Key Design Decisions**:
- `@runtime_checkable`: Enables `isinstance(adapter, MemoryProvider)` checks
- `MemoryItem.model_config`: Allows extra fields from different providers
- `metadata filters`: Strict equality only (partial matches in Phase 5)
- `search ordering`: Newest first (created_at desc) for deterministic tests
- Empty query behavior: Returns all memories within namespace

### B. Mock Implementation
**File**: `DeepResearch/src/memory/adapters/mock_adapter.py`
**Responsibility**: In-memory implementation of `MemoryProvider` for testing and offline runs.

**Complete Implementation**:
```python
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any

from DeepResearch.src.memory.core import MemoryItem, MemoryProvider


class MockMemoryAdapter(MemoryProvider):
    """In-memory MemoryProvider for tests and offline runs."""

    def __init__(self) -> None:
        self._memories: dict[str, dict[str, Any]] = {}
        self._counter = 0
        self._lock = asyncio.Lock()

    def _namespace(self, user_id: str, agent_id: str) -> str:
        """Generate namespace key for agent isolation."""
        return f"{user_id}:{agent_id}"

    async def add(
        self,
        content: str,
        user_id: str,
        agent_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        async with self._lock:
            memory_id = f"mem_{self._counter}"
            self._counter += 1
        record = {
            "id": memory_id,
            "content": content,
            "user_id": user_id,
            "agent_id": agent_id,
            "metadata": metadata or {},
            "created_at": datetime.now(timezone.utc),
            "namespace": self._namespace(user_id, agent_id),
        }
        self._memories[memory_id] = record
        return memory_id

    async def add_trace(
        self,
        agent_id: str,
        workflow_id: str,
        trace_data: dict[str, Any],
        user_id: str = "system",
    ) -> str:
        """Add execution trace with structured metadata."""
        payload = json.dumps(trace_data, default=str)
        metadata = {"type": "trace", "workflow_id": workflow_id, **trace_data.get("metadata", {})}
        return await self.add(
            content=f"[trace] workflow={workflow_id} agent={agent_id} data={payload}",
            user_id=user_id,
            agent_id=agent_id,
            metadata=metadata,
        )

    async def search(
        self,
        query: str,
        user_id: str,
        agent_id: str,
        limit: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        """Search with namespace isolation and metadata filtering."""
        namespace = self._namespace(user_id, agent_id)
        results = []
        query_l = (query or "").lower()
        for record in self._memories.values():
            # 1. Namespace isolation
            if record["namespace"] != namespace:
                continue
            # 2. Metadata filters (strict equality)
            if filters:
                if any(record["metadata"].get(k) != v for k, v in filters.items()):
                    continue
            # 3. Substring search (case-insensitive)
            if query_l and query_l not in record["content"].lower():
                continue
            results.append(record)
        # Sort newest first for deterministic tests
        results.sort(key=lambda r: r["created_at"], reverse=True)
        return [
            MemoryItem(
                id=r["id"],
                content=r["content"],
                score=1.0,  # Mock: all results have score 1.0
                metadata=r["metadata"],
                created_at=r["created_at"],
                agent_id=r["agent_id"],
                user_id=r["user_id"],
            )
            for r in results[:limit]
        ]

    async def get_all(
        self,
        user_id: str,
        agent_id: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        """Retrieve all memories (delegates to search with empty query)."""
        return await self.search("", user_id, agent_id, limit=limit, filters=filters)

    async def delete(self, memory_id: str) -> bool:
        """Delete memory by ID."""
        return self._memories.pop(memory_id, None) is not None

    async def reset(self) -> bool:
        """Full wipe for testing."""
        self._memories.clear()
        self._counter = 0
        return True
```

**Key Features**:
- **Namespace Isolation**: `user_id:agent_id` prevents cross-agent memory leaks
- **Async Lock**: Thread-safe counter for concurrent access
- **Substring Matching**: Case-insensitive search in content
- **Metadata Filtering**: Strict equality for trace filtering (`type="trace"`)
- **Deterministic Ordering**: Newest first for reproducible tests
- **Trace Encoding**: JSON payload in content, structured metadata for filtering

### C. Factory/Registry
**File**: `DeepResearch/src/memory/factory.py`
**Responsibility**: Central entry point to get a memory provider.

**Complete Implementation**:
```python
from __future__ import annotations

from typing import Mapping

from omegaconf import DictConfig

from DeepResearch.src.memory.adapters.mock_adapter import MockMemoryAdapter
from DeepResearch.src.memory.core import MemoryProvider


def get_memory_provider(cfg: DictConfig | Mapping[str, object] | None) -> MemoryProvider:
    """
    Factory for memory providers.

    Args:
        cfg: Hydra config or dict with 'provider' key

    Returns:
        MemoryProvider instance

    Raises:
        ValueError: If provider is unknown
        NotImplementedError: If provider not yet implemented
    """
    provider = (cfg or {}).get("provider", "mock") if isinstance(cfg, Mapping) else getattr(cfg, "provider", "mock")

    if provider == "mock":
        return MockMemoryAdapter()
    if provider == "mem0":
        raise NotImplementedError("Mem0 adapter arrives in Phase 4B.")

    raise ValueError(f"Unsupported memory provider: {provider}")
```

**Key Features**:
- Handles both `DictConfig` and plain `dict`
- Default to "mock" if config missing
- Clear error messages for invalid providers

---

## 3. TDD Strategy (Test First)

**Test File**: `DeepResearch/tests/memory/test_core_interface.py`

**Complete Test Suite**:
```python
import pytest

from DeepResearch.src.memory.adapters.mock_adapter import MockMemoryAdapter
from DeepResearch.src.memory.core import MemoryProvider


@pytest.mark.asyncio
async def test_mock_add_and_search_namespaced():
    """Test namespace isolation between agents."""
    adapter = MockMemoryAdapter()
    await adapter.add("bio p53", user_id="u1", agent_id="bio")
    await adapter.add("prime task", user_id="u1", agent_id="prime")
    hits = await adapter.search("p53", user_id="u1", agent_id="bio")
    assert len(hits) == 1
    assert hits[0].content == "bio p53"


@pytest.mark.asyncio
async def test_metadata_filtering():
    """Test filtering by metadata (e.g. type='trace')."""
    adapter = MockMemoryAdapter()
    await adapter.add("chat msg", user_id="u1", agent_id="a1", metadata={"type": "chat"})
    await adapter.add_trace("a1", "wf1", {"tool": "blast"}, user_id="u1")
    traces = await adapter.search("", user_id="u1", agent_id="a1", filters={"type": "trace"})
    assert len(traces) == 1
    assert traces[0].metadata["type"] == "trace"


@pytest.mark.asyncio
async def test_add_trace_structure():
    """Test add_trace creates properly structured memory."""
    adapter = MockMemoryAdapter()
    mem_id = await adapter.add_trace(
        agent_id="bio",
        workflow_id="wf1",
        trace_data={"tool": "blast", "result": {"hits": 3}},
        user_id="system",
    )
    hits = await adapter.search("blast", user_id="system", agent_id="bio")
    assert len(hits) == 1
    assert hits[0].id == mem_id
    assert hits[0].metadata["type"] == "trace"
    assert hits[0].metadata["workflow_id"] == "wf1"


@pytest.mark.asyncio
async def test_delete_and_reset():
    """Test delete and reset operations."""
    adapter = MockMemoryAdapter()
    mem_id = await adapter.add("test", user_id="u1", agent_id="a1")
    assert await adapter.delete(mem_id) is True
    assert await adapter.delete(mem_id) is False  # Already deleted
    await adapter.add("test2", user_id="u1", agent_id="a1")
    assert await adapter.reset() is True
    hits = await adapter.get_all(user_id="u1", agent_id="a1")
    assert len(hits) == 0


def test_protocol_type_check():
    """Test runtime protocol compliance."""
    adapter = MockMemoryAdapter()
    assert isinstance(adapter, MemoryProvider)
```

**Key Test Coverage**:
- ✅ Namespace isolation (bio vs prime agents)
- ✅ Metadata filtering (`type="trace"`)
- ✅ Trace structure validation
- ✅ CRUD operations (add, delete, reset)
- ✅ Protocol compliance (`isinstance` check)
- ✅ Async markers (`@pytest.mark.asyncio`)

**Run Tests**:
```bash
pytest DeepResearch/tests/memory/test_core_interface.py -v
uvx ty check DeepResearch/src/memory
```

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