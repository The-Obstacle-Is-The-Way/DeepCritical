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
**Responsibility**: Real implementation of `MemoryProvider` with Mem0 SDK.

**Complete Implementation**:
```python
from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any, Callable, Sequence, TypeVar

from mem0 import Memory
from mem0.client import MemoryClient
from omegaconf import DictConfig, OmegaConf

from DeepResearch.src.memory.core import MemoryItem, MemoryProvider

T = TypeVar("T")


class Mem0Adapter(MemoryProvider):
    """Mem0-backed MemoryProvider (OSS Neo4j or Cloud)."""

    def __init__(self, cfg: DictConfig):
        self.mode = getattr(cfg, "mode", "oss")
        self._client = self._init_client(cfg)

    def _init_client(self, cfg: DictConfig):
        """Initialize Mem0 client based on mode (OSS or Cloud)."""
        if self.mode == "oss":
            neo_cfg = cfg.get("oss", {})
            graph_cfg = OmegaConf.to_object(neo_cfg.get("graph_store", {})) or {}
            vector_cfg = OmegaConf.to_object(neo_cfg.get("vector_store", {})) or {}
            if not graph_cfg or not vector_cfg:
                raise ValueError("OSS mode requires graph_store and vector_store configs.")
            mem0_config = {
                "graph_store": graph_cfg,
                "vector_store": vector_cfg,
                "llm": neo_cfg.get("llm"),
                "embedding": neo_cfg.get("embedding"),
            }
            return Memory.from_config(mem0_config)
        if self.mode == "cloud":
            api_key = cfg.get("cloud", {}).get("api_key")
            base_url = cfg.get("cloud", {}).get("base_url")
            if not api_key:
                raise ValueError("Mem0 cloud mode requires cloud.api_key")
            return MemoryClient(api_key=api_key, base_url=base_url)
        raise ValueError(f"Unsupported Mem0 mode: {self.mode}")

    async def _run(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Run sync Mem0 SDK call in executor to avoid blocking."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))

    async def add(
        self,
        content: str,
        user_id: str,
        agent_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        resp = await self._run(
            self._client.add,
            content,
            user_id=user_id,
            agent_id=agent_id,
            metadata=metadata or {},
        )
        return resp.get("id") or resp.get("memory_id") or str(resp)

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
        raw = await self._run(
            self._client.search,
            query=query,
            user_id=user_id,
            agent_id=agent_id,
            limit=limit,
            filters=filters or {},
        )
        return self._normalize_response(raw, agent_id=agent_id, user_id=user_id, limit=limit)

    async def get_all(
        self,
        user_id: str,
        agent_id: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        raw = await self._run(
            self._client.get_all,
            user_id=user_id,
            agent_id=agent_id,
            filters=filters or {},
            limit=limit,
        )
        return self._normalize_response(raw, agent_id=agent_id, user_id=user_id, limit=limit)

    async def delete(self, memory_id: str) -> bool:
        return bool(await self._run(self._client.delete, memory_id))

    async def reset(self) -> bool:
        if hasattr(self._client, "reset"):
            await self._run(self._client.reset)
        return True

    def _normalize_response(
        self, raw: Any, agent_id: str, user_id: str, limit: int
    ) -> list[MemoryItem]:
        """Normalize Mem0's variable response formats into MemoryItem list."""
        entries: Sequence[Any]
        if isinstance(raw, dict) and "results" in raw:
            entries = raw["results"]
        elif isinstance(raw, Sequence):
            entries = raw
        else:
            return []
        items: list[MemoryItem] = []
        for entry in entries[:limit]:
            payload = entry.get("memory", entry)
            created = payload.get("created_at")
            created_dt = (
                datetime.fromisoformat(created) if isinstance(created, str) else None
            )
            items.append(
                MemoryItem(
                    id=str(payload.get("id") or payload.get("_id")),
                    content=payload.get("content", ""),
                    score=payload.get("score"),
                    metadata=payload.get("metadata") or {},
                    created_at=created_dt,
                    agent_id=payload.get("agent_id", agent_id),
                    user_id=payload.get("user_id", user_id),
                )
            )
        return items
```

**Key Features**:
- **Sync-to-Async Bridge**: `run_in_executor` for non-blocking Mem0 SDK calls
- **Dual Mode Support**: OSS (Neo4j) and Cloud (Mem0 Platform)
- **Response Normalization**: Handles `{"results": [...]}` vs `[...]` formats
- **Config Validation**: Raises clear errors for missing credentials
- **Trace Encoding**: Same format as MockAdapter for consistency
- **Version Pin Recommendation**: Use a tested Mem0 SDK (e.g., `uv add mem0ai==<tested_version> neo4j==5.* testcontainers[neo4j]==4.*`) and keep the sync→async bridge (`run_in_executor`) because the SDK is synchronous.

### B. Factory Update
**File**: `DeepResearch/src/memory/factory.py`

```python
from DeepResearch.src.memory.adapters.mem0_adapter import Mem0Adapter

if provider == "mem0":
    return Mem0Adapter(cfg)
```

### C. Integration Tests
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

**Content (aligned to existing Hydra structure in `configs/db/neo4j.yaml`: `db.uri/username/password`)**:
```yaml
memory:
  enabled: true
  provider: mem0
  mode: oss  # or cloud

  # OSS Configuration (Maps to existing db.* keys)
  oss:
    graph_store:
      provider: neo4j
      config:
        url: ${db.uri}
        username: ${db.username}
        password: ${db.password}
    vector_store:
      provider: neo4j
      config:
        url: ${db.uri}
        username: ${db.username}
        password: ${db.password}
        embedding_model_dims: 1536  # Match existing

  # Cloud Configuration
  cloud:
    api_key: ${oc.env:MEM0_API_KEY,}
    base_url: ${oc.env:MEM0_BASE_URL,}
```

### B. Wire into Main Config
**File**: `DeepResearch/configs/config.yaml`
**Change**: Add `memory: default` to the `defaults:` list (alongside existing `db: neo4j` and `neo4j: orchestrator` entries):
```yaml
defaults:
  - challenge: default
  - workflow_orchestration: default
  - db: neo4j
  - neo4j: orchestrator
  - memory: default  # ← ADD
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
    - Concrete example:
    ```python
    import pytest
    from neo4j import GraphDatabase
    from omegaconf import OmegaConf
    from testcontainers.neo4j import Neo4jContainer

    from DeepResearch.src.memory.adapters.mem0_adapter import Mem0Adapter


    @pytest.mark.asyncio
    @pytest.mark.containerized  # Skip in CI when Docker unavailable
    async def test_mem0_add_and_search_with_neo4j():
        with Neo4jContainer("neo4j:5.22") as neo4j:
            cfg = OmegaConf.create(
                {
                    "provider": "mem0",
                    "mode": "oss",
                    "oss": {
                        "graph_store": {
                            "provider": "neo4j",
                            "config": {
                                "url": neo4j.get_connection_url(),
                                "username": "neo4j",
                                "password": neo4j.NEO4J_ADMIN_PASSWORD,
                            },
                        },
                        "vector_store": {
                            "provider": "neo4j",
                            "config": {
                                "url": neo4j.get_connection_url(),
                                "username": "neo4j",
                                "password": neo4j.NEO4J_ADMIN_PASSWORD,
                            },
                        },
                    },
                }
            )
            adapter = Mem0Adapter(cfg)
            mem_id = await adapter.add("test memory", user_id="u1", agent_id="a1")
            results = await adapter.search("test", user_id="u1", agent_id="a1")
            assert any(r.id == mem_id for r in results)

            driver = GraphDatabase.driver(
                neo4j.get_connection_url().replace("bolt://", "neo4j://"),
                auth=("neo4j", neo4j.NEO4J_ADMIN_PASSWORD),
            )
            with driver.session(database="neo4j") as session:
                count = session.run(
                    "MATCH (m:Memory {id:$id}) RETURN count(m) AS c", {"id": mem_id}
                ).single()["c"]
                assert count == 1
    ```

---

## 5. Implementation Steps

1.  **Dependencies**: `uv add mem0ai neo4j`.
2.  **Config**: Create `DeepResearch/configs/memory/default.yaml`.
3.  **Code**: Implement `DeepResearch/src/memory/adapters/mem0_adapter.py`.
4.  **Factory**: Update `DeepResearch/src/memory/factory.py` to support `provider="mem0"` (return `Mem0Adapter(cfg)`).
5.  **Tests**: Write and run unit & integration tests.

---

## 6. Acceptance Criteria
- [ ] `Mem0Adapter` implemented.
- [ ] Adapter correctly normalizes Mem0 responses.
- [ ] `add_trace` successfully stores structured data as searchable text + metadata.
- [ ] Integration test passes with real Neo4j container.
- [ ] No hardcoded credentials.

---

## 7. Optional Enhancement: Local Cache with FileLock

**Source**: Existing pattern in `DeepResearch/src/utils/analytics.py:78-97`

If implementing local cache for offline operation or performance, use proven FileLock pattern:

```python
from filelock import FileLock
from pathlib import Path
import json

class Mem0Adapter:
    def __init__(self, config: dict):
        self.client = Memory.from_config(config)
        self.cache_file = Path(config.get("cache_file", ".mem0_cache.json"))
        self.use_cache = config.get("use_local_cache", False)

    async def search(self, query: str, user_id: str, agent_id: str, ...) -> list[MemoryItem]:
        """Search with optional local cache."""
        if self.use_cache:
            # Try cache first
            cached = self._load_cache(query)
            if cached:
                return cached

        # Mem0 search
        results = self.client.search(...)

        if self.use_cache:
            self._save_cache(query, results)

        return results

    def _save_cache(self, query: str, results: list) -> None:
        """Save cache atomically using FileLock pattern."""
        lock_file = str(self.cache_file) + ".lock"
        with FileLock(lock_file):
            cache_data = self._load_cache_unlocked() or {}
            cache_data[query] = {
                "results": [r.model_dump() for r in results],
                "timestamp": time.time()
            }
            with self.cache_file.open('w') as f:
                json.dump(cache_data, f)
```

**Configuration**:
```yaml
# configs/memory/default.yaml
memory:
  oss:
    use_local_cache: false  # Enable for offline operation
    cache_file: ".mem0_cache.json"
    cache_ttl: 300  # 5 minutes
```

**Priority**: Optional (defer to Phase 5 if not needed for pilot)

---

**Next Phase**: 4C (Agent Wiring & Middleware)
