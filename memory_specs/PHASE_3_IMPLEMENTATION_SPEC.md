# Phase 3: Memory System Implementation Spec

**Status**: 🚧 Proposed
**Date**: 2025-11-19
**Goal**: Define the concrete architecture, schema, and implementation plan for integrating long-term memory into DeepCritical.

---

## 1. Executive Summary

We will implement a **Hybrid "Ports & Adapters" Memory System** that unifies:
1.  **Mem0 (OSS)** for handling unstructured user conversations (Dynamic Graph + Vectors).
2.  **Native Neo4j** for structured system events (Strict Schema: G-Memory Hierarchy).
3.  **Agent Profiles** for selective retrieval (O-Mem pattern).

This approach avoids "fighting" Mem0's dynamic schema for chat while ensuring we have a rigorous, queryable structure for agent execution traces (Tools, Plans, Workflows).

---

## 2. Architecture Overview

```mermaid
graph TB
    subgraph "Agent Layer (Pydantic AI)"
        Agent[BaseAgent]
        Deps[AgentDependencies]
        Middleware[MemoryMiddleware]
    end

    subgraph "Port (Interface)"
        Provider[< Protocol >\nMemoryProvider]
    end

    subgraph "Adapter (Implementation)"
        HybridAdapter[HybridMemoryAdapter]
    end

    subgraph "Storage Layer (Neo4j)"
        Mem0Logic[Mem0 Logic\n(Unstructured Chat)]
        SystemLogic[Native System Logic\n(Structured Traces)]
        
        Neo4j[(Neo4j Database)]
    end

    Agent --> Deps
    Deps --> Provider
    Middleware --> Provider
    Provider --> HybridAdapter
    
    HybridAdapter -->|Chat Logs| Mem0Logic
    HybridAdapter -->|Tool/Plan Events| SystemLogic
    
    Mem0Logic --> Neo4j
    SystemLogic --> Neo4j
```

---

## 3. Data Schema (Neo4j)

We will use a **Dual-Schema** approach within the same Neo4j database.

### A. Dynamic Schema (Managed by Mem0)
*   **Nodes**: `Entity` (User, Topic, Concept) extracted dynamically from chat.
*   **Edges**: `RELATIONSHIP` (dynamic types like "LIKES", "DISCUSSED").
*   **Use Case**: User personalization, conversational context.

### B. Strict Schema (Managed by Us - G-Memory Pattern)
*   **Nodes**:
    *   `Agent` (e.g., "bioinformatics_agent")
    *   `Workflow` (e.g., "PRIME_Run_123")
    *   `Goal` (Insight Layer)
    *   `Plan` (Query Layer)
    *   `Action` (Interaction Layer - Tool Calls)
    *   `Observation` (Tool Results)
*   **Edges**:
    *   `(:Agent)-[:EXECUTED]->(:Action)`
    *   `(:Action)-[:PRODUCED]->(:Observation)`
    *   `(:Plan)-[:HAS_STEP]->(:Action)`
    *   `(:Workflow)-[:ACHIEVED]->(:Goal)`
    *   `(:Action)-[:NEXT]->(:Action)` (Temporal Chain)

---

## 4. Core Interfaces (The "Port")

We will define a vendor-agnostic protocol in `DeepResearch/src/memory/core.py`.

```python
from typing import Protocol, Any, Optional
from datetime import datetime
from pydantic import BaseModel

class MemoryItem(BaseModel):
    content: str
    score: float
    metadata: dict[str, Any]
    timestamp: datetime

class MemoryProvider(Protocol):
    async def add(
        self, 
        content: str, 
        user_id: str, 
        agent_id: str, 
        metadata: Optional[dict] = None
    ) -> str:
        """Add unstructured memory (chat/notes)."""
        ...

    async def add_trace(
        self,
        agent_id: str,
        workflow_id: str,
        action: str,
        result: str,
        metadata: Optional[dict] = None
    ) -> str:
        """Add structured execution trace (tool usage)."""
        ...

    async def search(
        self, 
        query: str, 
        user_id: str, 
        agent_id: str, 
        limit: int = 5
    ) -> list[MemoryItem]:
        """Hybrid search (semantic + graph)."""
        ...

    async def get_history(
        self,
        agent_id: str,
        workflow_id: str,
        limit: int = 10
    ) -> list[MemoryItem]:
        """Retrieve recent execution history."""
        ...
```

---

## 5. Implementation Plan (Phase 4 Vertical Slices)

### Slice 1: The "Brain" Backbone (Foundation)
*   **Goal**: Set up the `MemoryProvider` interface and a basic `Neo4jAdapter`.
*   **Tasks**:
    1.  Create `DeepResearch/src/memory/` package.
    2.  Define `core.py` (interfaces).
    3.  Implement `neo4j_adapter.py` (using existing `Neo4jVectorStore` logic + adding direct graph writes).
    4.  Config: Add `configs/memory/default.yaml`.

### Slice 2: Integration Hooks (Wiring)
*   **Goal**: Connect the "Brain" to the "Body" (Agents).
*   **Tasks**:
    1.  Modify `ResearchState` in `app.py` (add `memory_client`).
    2.  Modify `AgentDependencies` in `datatypes/agents.py`.
    3.  Create `MemoryMiddleware` in `tools/deep_agent_middleware.py`.
    4.  Register `memory` tool in `BaseAgent`.

### Slice 3: The "Bioinformatics" Pilot (Proof of Concept)
*   **Goal**: Prove value with a real agent.
*   **Tasks**:
    1.  Enable memory for `BioinformaticsAgent`.
    2.  Run a workflow: "Find P53 targets".
    3.  Verify Neo4j: Check if `(:Agent)-[:EXECUTED]->(:Action)` nodes appear.
    4.  Verify Recall: Ask "What did I just do?" and check if agent retrieves the tool trace.

### Slice 4: Mem0 Integration (Unstructured Upgrade)
*   **Goal**: Add the "Chat" layer.
*   **Tasks**:
    1.  Install `mem0ai`.
    2.  Update `HybridAdapter` to route string-based `.add()` calls to Mem0.
    3.  Keep `.add_trace()` routing to our custom Neo4j logic.

---

## 6. Testing Strategy

*   **Unit Tests**: Mock `MemoryProvider` to ensure Agents call it correctly.
*   **Integration Tests**: Use `testcontainers` to spin up a disposable Neo4j instance. Verify nodes/edges are created.
*   **Performance**: Measure latency overhead of `MemoryMiddleware`. Target < 500ms added latency per step.

---

## 7. Configuration (Hydra)

**`configs/memory/neo4j.yaml`**
```yaml
memory:
  enabled: true
  provider: "hybrid_neo4j"
  neo4j:
    uri: "${db.neo4j.uri}"
    username: "${db.neo4j.username}"
    password: "${db.neo4j.password}"
  profiles:
    bioinformatics_agent:
      search_limit: 10
      retention_window: "7d"
```

---

**Status**: Ready for Slice 1 Implementation.
