# Phase 4 Validation Report

## Executive Summary
- Total gaps found: 32 across 4A–4D; many block implementation without extra guidance.
- Critical gaps: 20 (missing complete code, undefined wiring paths, async/sync mismatches).
- Minor gaps: 12 (clarifications on config defaults, filters, acceptance metrics).
- Recommendation: NEEDS REVISION before implementation starts.

## Phase 4A: Core Interface + Mock
### Gaps Found
- No `runtime_checkable` protocol or `MemoryFilter` model; metadata typing, default values, and allowed extras not defined.
- Missing full `MemoryProvider` signature details (return types, default user/agent semantics, ordering guarantees, error handling for None/empty queries).
- Mock adapter lacks implementation details: storage structure, id generation, timezone handling, namespace isolation, substring search rules, metadata filter semantics, delete/reset behavior, and trace serialization.
- `add_trace` lacks required metadata contract (`type="trace"`, workflow/agent/user defaults) and how trace content is encoded for search.
- Factory pattern undefined: how config is read (DictConfig vs dict), default provider selection, and failure mode when provider key is absent or unknown.
- Tests underspecified: no async markers, no fixtures for namespacing/filters, no coverage target, no ty/static protocol conformance step.

### Recommended Additions
- Add concrete core definitions (`DeepResearch/src/memory/core.py`) with strict typing and runtime validation:
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
    async def add(
        self,
        content: str,
        user_id: str,
        agent_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> str: ...

    async def add_trace(
        self,
        agent_id: str,
        workflow_id: str,
        trace_data: dict[str, Any],
        user_id: str = "system",
    ) -> str: ...

    async def search(
        self,
        query: str,
        user_id: str,
        agent_id: str,
        limit: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[MemoryItem]: ...

    async def get_all(
        self,
        user_id: str,
        agent_id: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[MemoryItem]: ...

    async def delete(self, memory_id: str) -> bool: ...
    async def reset(self) -> bool: ...
```
- Provide full mock adapter implementation (`DeepResearch/src/memory/adapters/mock_adapter.py`) that enforces namespace isolation and filtering:
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
        namespace = self._namespace(user_id, agent_id)
        results = []
        query_l = (query or "").lower()
        for record in self._memories.values():
            if record["namespace"] != namespace:
                continue
            if filters:
                if any(record["metadata"].get(k) != v for k, v in filters.items()):
                    continue
            if query_l and query_l not in record["content"].lower():
                continue
            results.append(record)
        # Newest first for deterministic tests
        results.sort(key=lambda r: r["created_at"], reverse=True)
        return [
            MemoryItem(
                id=r["id"],
                content=r["content"],
                score=1.0,
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
        return await self.search("", user_id, agent_id, limit=limit, filters=filters)

    async def delete(self, memory_id: str) -> bool:
        return self._memories.pop(memory_id, None) is not None

    async def reset(self) -> bool:
        self._memories.clear()
        self._counter = 0
        return True
```
- Clarify factory behavior (`DeepResearch/src/memory/factory.py`):
```python
from __future__ import annotations

from typing import Mapping

from omegaconf import DictConfig

from DeepResearch.src.memory.adapters.mock_adapter import MockMemoryAdapter
from DeepResearch.src.memory.core import MemoryProvider


def get_memory_provider(cfg: DictConfig | Mapping[str, object] | None) -> MemoryProvider:
    provider = (cfg or {}).get("provider", "mock") if isinstance(cfg, Mapping) else getattr(cfg, "provider", "mock")
    if provider == "mock":
        return MockMemoryAdapter()
    if provider == "mem0":
        raise NotImplementedError("Mem0 adapter arrives in Phase 4B.")
    raise ValueError(f"Unsupported memory provider: {provider}")
```
- Expand tests (`DeepResearch/tests/memory/test_core_interface.py`) with explicit async markers, namespace isolation, metadata filter coverage, and ty protocol conformance:
```python
import pytest
from DeepResearch.src.memory.adapters.mock_adapter import MockMemoryAdapter
from DeepResearch.src.memory.core import MemoryProvider


@pytest.mark.asyncio
async def test_mock_add_and_search_namespaced():
    adapter = MockMemoryAdapter()
    await adapter.add("bio p53", user_id="u1", agent_id="bio")
    await adapter.add("prime task", user_id="u1", agent_id="prime")
    hits = await adapter.search("p53", user_id="u1", agent_id="bio")
    assert [h.content for h in hits] == ["bio p53"]


def test_protocol_type_check():
    adapter = MockMemoryAdapter()
    assert isinstance(adapter, MemoryProvider)
```

### Ambiguities
- Should metadata filters support partial matches/lists or strict equality only?
- Desired ordering for `search`/`get_all` (created_at desc vs score desc) is not specified.
- Expected behavior when `query` is empty or None (return all vs raise).
- Default `user_id` for `add_trace` when not provided is unclear.

## Phase 4B: Mem0 Adapter
### Gaps Found
- Missing concrete imports and sync-to-async bridging; Mem0 SDK calls are synchronous while the protocol is async.
- mem0_config construction from Hydra unclear: which config group to read (`db` vs `neo4j`), how to validate presence of `uri/username/password`, and how to surface errors.
- Response normalization unspecified (Mem0 returns dict vs list; nested `memory` payloads, scores, timestamps).
- `add_trace` serialization contract undefined (content string shape, metadata keys, default user id).
- Mode handling incomplete: no base URL/timeout for cloud, no embedding options for OSS, no fallback when mode invalid.
- No dependency pinning/version guidance (`mem0ai`, `neo4j`, `testcontainers`), no import guard when SDK missing.
- Integration test lacks container lifecycle, Cypher assertion, and async handling; no strategy for running in CI without Docker.
- Hydra defaults change not reconciled with current `configs/config.yaml` (`neo4j: orchestrator` already present); override order and conflict resolution undefined.

### Recommended Additions
- Specify full adapter implementation (`DeepResearch/src/memory/adapters/mem0_adapter.py`) with sync-to-async wrapper and normalization:
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
- Clarify Hydra config mapping with explicit example (`DeepResearch/configs/memory/default.yaml`) aligned to existing `configs/db/neo4j.yaml` and `configs/config.yaml` defaults:
```yaml
defaults:
  - challenge: default
  - workflow_orchestration: default
  - db: neo4j
  - neo4j: orchestrator
  - memory: default  # new
  - _self_
  - override hydra/job_logging: default
  - override hydra/hydra_logging: default

memory:
  enabled: true
  provider: mem0
  mode: oss
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
  cloud:
    api_key: ${oc.env:MEM0_API_KEY,}
    base_url: ${oc.env:MEM0_BASE_URL,}
```
- Provide concrete integration test using `testcontainers` and Cypher validation (`DeepResearch/tests/memory/test_mem0_integration.py`):
```python
import pytest
from neo4j import GraphDatabase
from testcontainers.neo4j import Neo4jContainer

from DeepResearch.src.memory.adapters.mem0_adapter import Mem0Adapter
from omegaconf import OmegaConf


@pytest.mark.asyncio
@pytest.mark.containerized
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
- Pin dependency guidance in spec: `uv add mem0ai==<tested> neo4j==5.* testcontainers[neo4j]==4.*` and require import guards to raise a clear error when SDK missing.

### Ambiguities
- Actual Mem0 SDK API surface (sync vs async, return keys) is not documented; need to cite exact version to implement normalization safely.
- How to pass embedding model/LLM settings when `embedding_model_dims` differs from Mem0 defaults is undefined.
- Whether to store execution traces under `user_id="system"` or workflow owner is unclear.
- How to run integration tests in CI when Docker is unavailable (skip marker? fallback to mock?) needs definition.

## Phase 4C: Agent Wiring
### Gaps Found
- No explicit path from `hydra.main` (`DeepResearch/app.py`) to instantiate and pass `MemoryProvider` into orchestrators, nodes, or agents.
- `AgentDependencies` addition is typeless (`Any`) with no agent/user identifiers, so namespacing in adapters cannot work.
- ExecutionContext wiring missing: `PrimeExecute` currently builds `ExecutionContext` without memory or IDs; spec does not reference this creation site.
- `ResearchState.memory_context` shape and when it is populated are unspecified, risking unused field.
- Memory tool implementation absent (no code showing `ctx.deps.memory` usage or error handling when None).
- Orchestrator wiring lacks detail: how to store the provider on `AgentOrchestrator`, how to ensure all agents get the same instance, and when to fall back to mock if disabled.

### Recommended Additions
- Define dependency fields with identifiers to preserve namespaces (`DeepResearch/src/datatypes/agents.py`):
```python
@dataclass
class AgentDependencies:
    config: dict[str, Any] = field(default_factory=dict)
    tools: list[str] = field(default_factory=list)
    other_agents: list[str] = field(default_factory=list)
    data_sources: list[str] = field(default_factory=list)
    memory: MemoryProvider | None = None
    user_id: str | None = None
    agent_id: str | None = None
```
- Wire provider at app entry (`DeepResearch/app.py`) so every node can access it:
```python
from DeepResearch.src.memory.factory import get_memory_provider

@hydra.main(...)
def main(cfg: DictConfig) -> None:
    memory_provider = None
    if getattr(cfg, "memory", None) and getattr(cfg.memory, "enabled", False):
        memory_provider = get_memory_provider(cfg.memory)
    state = ResearchState(question=question, config=cfg)
    state.agent_orchestrator = AgentOrchestrator(
        config=AgentOrchestratorConfig(),  # existing config usage
        memory_provider=memory_provider,
    )
    run_graph(question, cfg, memory_provider)
```
- Thread provider through graph execution by updating `run_graph` signature and node creation sites (`PrimeExecute` etc.) to accept and propagate memory:
```python
def run_graph(question: str, cfg: DictConfig, memory_provider: MemoryProvider | None):
    state = ResearchState(question=question, config=cfg)
    state.memory_context = []
    # when building PrimeExecute
    history = PrimeExecutionHistory(memory_provider=memory_provider)
    context = ExecutionContext(
        workflow=ctx.state.workflow_dag,
        history=history,
        manual_confirmation=...,
        adaptive_replanning=...,
        memory=memory_provider,
        workflow_id=cfg.get("workflow_id", "prime"),  # new field
        agent_id="prime_executor",
    )
```
- Add memory tools (`DeepResearch/src/tools/memory_tools.py`) with safe fallbacks:
```python
from pydantic_ai import Agent, RunContext
from DeepResearch.src.datatypes.agents import AgentDependencies

def register_memory_tools(agent: Agent[AgentDependencies, str]) -> None:
    @agent.tool_plain
    async def recall_memory(
        ctx: RunContext[AgentDependencies], query: str, limit: int = 5
    ) -> dict[str, object]:
        if ctx.deps.memory is None or ctx.deps.user_id is None or ctx.deps.agent_id is None:
            return {"results": []}
        hits = await ctx.deps.memory.search(
            query, user_id=ctx.deps.user_id, agent_id=ctx.deps.agent_id, limit=limit
        )
        return {"results": [hit.model_dump() for hit in hits]}

    @agent.tool_plain
    async def save_note(
        ctx: RunContext[AgentDependencies], content: str, metadata: dict[str, object] | None = None
    ) -> dict[str, str]:
        if ctx.deps.memory is None or ctx.deps.user_id is None or ctx.deps.agent_id is None:
            return {"id": ""}
        memory_id = await ctx.deps.memory.add(
            content=content,
            user_id=ctx.deps.user_id,
            agent_id=ctx.deps.agent_id,
            metadata=metadata,
        )
        return {"id": memory_id}
```
- Update `ExecutionContext` to carry memory and identifiers (`DeepResearch/src/datatypes/execution.py`):
```python
@dataclass
class ExecutionContext:
    workflow: WorkflowDAG
    history: ExecutionHistory
    data_bag: dict[str, Any] = field(default_factory=dict)
    current_step: int = 0
    max_retries: int = 3
    manual_confirmation: bool = False
    adaptive_replanning: bool = True
    memory: MemoryProvider | None = None
    workflow_id: str | None = None
    agent_id: str | None = None
```
- Add wiring tests (`DeepResearch/tests/memory/test_wiring.py`) to assert memory is propagated from app → orchestrator → AgentDependencies and ExecutionContext.

### Ambiguities
- How to derive `user_id`/`agent_id` values (per Hydra config? runtime inputs?) is unspecified.
- Whether all agents share one provider instance or separate ones per flow is unclear.
- `ResearchState.memory_context` purpose (cache vs audit log) is undefined.
- Prime/Bioinformatics flows are synchronous today; need guidance on whether to refactor to async for memory calls or to keep sync and rely on background tasks.

## Phase 4D: Pilot Execution
### Gaps Found
- `ExecutionHistory.add_item` behavior with async provider undefined; current history is synchronous and will not run `asyncio.create_task` safely when no loop is running.
- Mapping from `ExecutionItem` fields to trace metadata/content is unspecified (which fields are required, how timestamps are encoded, how None values are handled).
- Source of `agent_id`/`workflow_id` for traces is not defined; `ExecutionContext` currently lacks these fields.
- No error handling/backpressure strategy when memory writes fail (log? ignore? retry?).
- Executor wiring not detailed: the exact place in `PrimeExecute`/Bioinformatics flow to pass memory provider and IDs is missing.
- Pilot test lacks concrete code (fixtures, markers, assertions on metadata/type, use of mock vs Mem0).
- Concurrency concerns unaddressed: fire-and-forget tasks may be dropped when `ToolExecutor.execute_workflow` runs synchronously and no running loop exists.

### Recommended Additions
- Extend `ExecutionHistory` to accept provider and persist deterministically, guarding against missing event loop (`DeepResearch/src/utils/execution_history.py`):
```python
from DeepResearch.src.memory.core import MemoryProvider

@dataclass
class ExecutionHistory:
    items: list[ExecutionItem] = field(default_factory=list)
    start_time: float = field(default_factory=lambda: datetime.now(timezone.utc).timestamp())
    end_time: float | None = None
    memory_provider: MemoryProvider | None = None
    workflow_id: str | None = None
    agent_id: str | None = None

    def add_item(self, item: ExecutionItem) -> None:
        self.items.append(item)
        if self.memory_provider:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._persist_to_memory(item))
            except RuntimeError:
                # No running loop; run synchronously to avoid dropping data
                asyncio.run(self._persist_to_memory(item))

    async def _persist_to_memory(self, item: ExecutionItem) -> None:
        payload = {
            "step_name": item.step_name,
            "tool": item.tool,
            "status": item.status.value,
            "result": item.result,
            "error": item.error,
            "timestamp": item.timestamp,
            "parameters": item.parameters,
            "duration": item.duration,
            "retry_count": item.retry_count,
        }
        await self.memory_provider.add_trace(
            agent_id=self.agent_id or "unknown_agent",
            workflow_id=self.workflow_id or "unknown_workflow",
            trace_data=payload,
            user_id="system",
        )
```
- Populate history/context with IDs at creation sites (e.g., `PrimeExecute` in `DeepResearch/app.py`):
```python
history = PrimeExecutionHistory(
    memory_provider=memory_provider,
    workflow_id="prime",
    agent_id="prime_executor",
)
context = ExecutionContext(
    workflow=ctx.state.workflow_dag,
    history=history,
    manual_confirmation=...,
    adaptive_replanning=...,
    memory=memory_provider,
    workflow_id="prime",
    agent_id="prime_executor",
)
```
- Add end-to-end pilot test (`DeepResearch/tests/memory/test_end_to_end_pilot.py`) using the mock adapter to avoid external deps:
```python
import pytest
from DeepResearch.src.memory.adapters.mock_adapter import MockMemoryAdapter
from DeepResearch.src.utils.execution_history import ExecutionHistory, ExecutionItem
from DeepResearch.src.utils.execution_status import ExecutionStatus


@pytest.mark.asyncio
async def test_history_add_item_persists_trace():
    memory = MockMemoryAdapter()
    history = ExecutionHistory(memory_provider=memory, workflow_id="wf1", agent_id="bio_agent")
    item = ExecutionItem(step_name="run_blast", tool="blast", status=ExecutionStatus.SUCCESS, result={"hits": 3})
    history.add_item(item)
    hits = await memory.search("blast", user_id="system", agent_id="bio_agent", filters={"type": "trace"})
    assert len(hits) == 1
    assert hits[0].metadata["workflow_id"] == "wf1"
```
- Document fallback strategy when no loop exists (synchronous write vs queued task) and expected performance impact.

### Ambiguities
- Whether synchronous `ToolExecutor.execute_workflow` should be refactored to async to avoid `asyncio.run` in `add_item` is undecided.
- Should failed memory writes fail the workflow or only log warnings?
- Are traces written for every retry attempt or only final outcomes?
- How to handle sensitive data in trace content/metadata (redaction policy) is not specified.

## Overall Assessment
- Implementation readiness: ~35% (core concepts present, but code-level guidance incomplete).
- Confidence level: MEDIUM (paths are plausible, but async/sync and config gaps risk rework).
- Estimated revision time: 6–8 hours to fill code samples, clarify wiring, and lock tests/configs.
