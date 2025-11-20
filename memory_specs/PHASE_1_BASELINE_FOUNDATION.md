# Phase 1: Baseline Codebase Foundation Spec
## Memory System Integration Points - DeepCritical/DeepResearch

**Status**: 🚧 In Review - Seeking Senior Approval
**Date**: 2025-11-19
**Purpose**: Comprehensive mapping of ALL integration points for memory system implementation
**Related Issue**: [#31 - Add Agent Memory and Workflow Context Management](https://github.com/DeepCritical/DeepCritical/issues/31)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Current Architecture Overview](#current-architecture-overview)
3. [State Management Patterns](#state-management-patterns)
4. [Agent Architecture (Pydantic AI)](#agent-architecture-pydantic-ai)
5. [Workflow Patterns (Pydantic Graph)](#workflow-patterns-pydantic-graph)
6. [Configuration Structure (Hydra)](#configuration-structure-hydra)
7. [Existing Storage Backends](#existing-storage-backends)
8. [Tool Ecosystem](#tool-ecosystem)
9. [Testing Infrastructure](#testing-infrastructure)
10. [Memory Integration Points Map](#memory-integration-points-map)
11. [Recommendations](#recommendations)

---

## Executive Summary

DeepCritical/DeepResearch is a **Hydra + Pydantic Graph + Pydantic AI multi-agent system** with:
- **~379 Python files** across the repo (including tests)
- **Dozens of tools**, including **18 MCP bioinformatics servers**
- **Vector stores implemented**: Neo4j and FAISS (other backends are referenced in docs/configs but not implemented here)
- **Multiple workflows**: default/search, challenge, PRIME, Bioinformatics, RAG, DeepSearch, Enhanced/Primary REACT orchestration, plus workflow-pattern statemachines
- **Many specialized agents** (e.g., Parser, Planner, Executor, Bioinformatics, DeepSearch, Orchestrator, RAG, DeepAgent variants)

**Key Finding**: The architecture already has **natural integration points** for memory at every layer:
- **State Layer**: `ResearchState` flows through graph nodes
- **Agent Layer**: `AgentDependencies` for dependency injection
- **Tool Layer**: `ExecutionHistory` tracks all tool executions
- **Storage Layer**: Unified `VectorStore` abstract interface
- **Config Layer**: Hierarchical Hydra composition

**Memory System Can Integrate Without Refactoring** - All hooks already exist.

---

## Current Architecture Overview

### High-Level System Architecture

```mermaid
graph TB
    subgraph "User Interface"
        CLI[CLI Entry Point<br/>deepresearch]
    end

    subgraph "Configuration Layer"
        Hydra[Hydra Config Manager<br/>configs/]
        ConfigYAML[config.yaml]
        FlowConfigs[Flow Configs<br/>prime, bio, rag, etc.]
    end

    subgraph "Workflow Engine"
        PGraph[Pydantic Graph<br/>app.py]
        ResearchState[ResearchState<br/>Central State Object]
        Nodes[Graph Nodes<br/>Plan, Route, Execute, etc.]
    end

    subgraph "Agent Orchestration"
        MultiAgent[MultiAgentOrchestrator<br/>agents.py]
        BaseAgent[BaseAgent<br/>Abstract Base]
        SpecializedAgents[Specialized Agents<br/>Parser, Planner, Executor, etc.]
        AgentDeps[AgentDependencies<br/>Dependency Injection]
    end

    subgraph "Tool Ecosystem"
        ToolRegistry[ToolRegistry<br/>tool registry]
        MCPServers[MCP Servers<br/>18 bioinformatics tools]
        ExecHistory[ExecutionHistory<br/>Tracks all executions]
    end

    subgraph "Storage Layer"
        Neo4j[Neo4j Vector Store<br/>Graph + Vectors]
        FAISS[FAISS Vector Store<br/>Local In-Memory]
        VectorStoreABC[VectorStore ABC<br/>Unified Interface]
    end

    subgraph "External Services"
        LLM[LLM Providers<br/>Anthropic, OpenAI]
        WebAPIs[Web APIs<br/>PubMed, UniProt, etc.]
    end

    CLI --> Hydra
    Hydra --> PGraph
    PGraph --> ResearchState
    ResearchState --> Nodes
    Nodes --> MultiAgent
    MultiAgent --> BaseAgent
    BaseAgent --> SpecializedAgents
    SpecializedAgents --> AgentDeps
    AgentDeps --> ToolRegistry
    ToolRegistry --> MCPServers
    ToolRegistry --> ExecHistory
    SpecializedAgents --> LLM
    ToolRegistry --> WebAPIs
    Nodes --> Neo4j
    Nodes --> FAISS
    Neo4j --> VectorStoreABC
    FAISS --> VectorStoreABC

    style ResearchState fill:#ffeb3b,stroke:#f57c00,stroke-width:3px
    style AgentDeps fill:#4caf50,stroke:#2e7d32,stroke-width:3px
    style ExecHistory fill:#2196f3,stroke:#1565c0,stroke-width:3px
    style VectorStoreABC fill:#9c27b0,stroke:#6a1b9a,stroke-width:3px
```

### Data Flow Through System

```mermaid
sequenceDiagram
    participant User
    participant Hydra
    participant PGraph as Pydantic Graph
    participant State as ResearchState
    participant Node as Graph Node
    participant Agent as Pydantic AI Agent
    participant Tool as Tool Execution
    participant Storage as Vector Store

    User->>Hydra: CLI: deepresearch question="..."
    Hydra->>PGraph: Initialize with config
    PGraph->>State: Create ResearchState
    PGraph->>Node: Execute Plan Node
    Node->>State: Read question, config
    Node->>Agent: Invoke PlannerAgent
    Agent->>Tool: Execute tool via registry
    Tool->>Storage: Store execution result
    Storage-->>Tool: Result stored
    Tool-->>Agent: Execution result
    Agent-->>Node: Agent response
    Node->>State: Update state (notes, results)
    Node->>PGraph: Return NextNode
    PGraph->>Node: Execute NextNode
    Note over PGraph,Storage: Repeat until terminal node
    PGraph-->>User: Final synthesized result
```

---

## State Management Patterns

### ResearchState - The Central State Object

**Location**: `DeepResearch/app.py:45-96`

```python
@dataclass
class ResearchState:
    """Central state object for research workflow execution."""

    # ===== CORE FIELDS =====
    question: str
    plan: list[str] | None = None
    full_plan: list[dict[str, Any]] | None = None
    notes: list[str] = field(default_factory=list)
    answers: list[str] = field(default_factory=list)

    # ===== PRIME-SPECIFIC =====
    structured_problem: StructuredProblem | None = None
    workflow_dag: WorkflowDAG | None = None
    execution_results: dict[str, Any] = field(default_factory=dict)

    # ===== ORCHESTRATION STATE =====
    config: DictConfig | None = None
    orchestration_config: WorkflowOrchestrationConfig | None = None
    orchestration_state: OrchestrationState | None = None
    spawned_workflows: list[str] = field(default_factory=list)
    multi_agent_results: dict[str, Any] = field(default_factory=dict)

    # ===== ENHANCED REACT ARCHITECTURE =====
    app_configuration: AppConfiguration | None = None
    agent_orchestrator: AgentOrchestrator | None = None
    nested_loops: dict[str, Any] = field(default_factory=dict)
    active_subgraphs: dict[str, Any] = field(default_factory=dict)
    break_conditions_met: list[str] = field(default_factory=list)
    loss_function_values: dict[str, float] = field(default_factory=dict)
    current_mode: AppMode | None = None
```

### Memory Integration Point #1: Add Memory Fields to ResearchState

```python
# PROPOSED ADDITIONS (types to be defined)
@dataclass
class ResearchState:
    # ... existing fields ...

    memory_session_id: str | None = None
    memory_client: MemorySystemClient | None = None  # proposed injectable handle
    memory_config: MemoryConfig | None = None        # proposed per-run settings
```

**Rationale**:
- `memory_session_id`: Unique identifier for this workflow run (enables cross-run retrieval)
- `memory_client`/`memory_config`: Hook for workflow-scoped memory access

---

### Alternative State Classes

#### DeepAgentState
**Location**: `DeepResearch/src/datatypes/deep_agent_state.py:190-240`

```python
@dataclass
class DeepAgentState:
    session_id: str
    conversation_history: list[Message]
    shared_state: dict[str, Any]
    todos: list[Todo]
    files: dict[str, FileInfo]
    execution_history: ExecutionHistory
```

**Memory Integration Opportunity**: `conversation_history` is perfect for multi-turn memory.

#### RAGState
**Location**: `DeepResearch/src/statemachines/rag_workflow.py:59-100`

```python
@dataclass
class RAGState:
    question: str
    rag_config: RAGConfig | None = None
    documents: list[Document] = field(default_factory=list)
    rag_response: RAGResponse | None = None
    processing_steps: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    config: DictConfig | None = None
    execution_status: ExecutionStatus = ExecutionStatus.PENDING
```

**Memory Integration Opportunity**: `documents` field already stores retrievable data.

---

### State Flow Through Pydantic Graph Nodes

```mermaid
graph LR
    subgraph "Node Execution Pattern"
        A[Node.run called] --> B[Read ctx.state]
        B --> C[Perform Work]
        C --> D[Modify ctx.state]
        D --> E[Return NextNode]
    end

    subgraph "Memory Integration"
        M1[Retrieve Memories<br/>Before Work]
        M2[Store Decisions<br/>After Work]
    end

    B --> M1
    M1 --> C
    D --> M2
    M2 --> E

    style M1 fill:#4caf50,stroke:#2e7d32,stroke-width:2px
    style M2 fill:#2196f3,stroke:#1565c0,stroke-width:2px
```

**Example Node Pattern**:
```python
@dataclass
class PlanNode(BaseNode[ResearchState]):
    async def run(self, ctx: GraphRunContext[ResearchState]) -> NextNode:
        # 1. Read state
        question = ctx.state.question
        config = ctx.state.config

        # 2. MEMORY INTEGRATION: Retrieve relevant context
        if ctx.state.execution_context:
            memories = await ctx.state.execution_context.memory_client.retrieve(
                query=question,
                filters={"agent_type": "planner"},
                top_k=3
            )
            context = "\n".join([m.content for m in memories])

        # 3. Perform work
        planner_agent = PlannerAgent(...)
        plan = await planner_agent.run(question, context=context)

        # 4. Modify state
        ctx.state.plan = plan
        ctx.state.notes.append(f"Plan created with {len(plan)} steps")

        # 5. MEMORY INTEGRATION: Store decision
        if ctx.state.execution_context:
            await ctx.state.execution_context.memory_client.store(
                content=f"Plan: {plan}",
                metadata={
                    "node": "PlanNode",
                    "question": question,
                    "timestamp": datetime.now().isoformat()
                }
            )

        # 6. Return next node
        return ExecuteNode()
```

---

## Agent Architecture (Pydantic AI)

### BaseAgent Pattern

**Location**: `DeepResearch/agents.py:47-110`

```python
class BaseAgent(ABC):
    def __init__(
        self,
        agent_type: AgentType,
        model_name: str = "anthropic:claude-sonnet-4-0",
        dependencies: AgentDependencies | None = None,
        system_prompt: str | None = None,
        instructions: str | None = None,
    ):
        self.agent_type = agent_type
        self._agent: Agent[AgentDependencies, str] | None = None
        self._initialize_agent(system_prompt, instructions)

    def _initialize_agent(self, system_prompt: str | None, instructions: str | None):
        self._agent = Agent[AgentDependencies, str](
            self.model_name,
            deps_type=AgentDependencies,
            system_prompt=system_prompt or self._get_default_system_prompt(),
            instructions=instructions or self._get_default_instructions(),
        )
        self._register_tools()

    @abstractmethod
    def _register_tools(self):
        """Register tools with the agent."""
        pass
```

### Memory Integration Point #2: AgentDependencies

**Location**: `DeepResearch/src/datatypes/agents.py` (dataclass near top)

**Current Implementation**:
```python
@dataclass
class AgentDependencies:
    """Dependencies injected into Pydantic AI agents."""
    config: dict[str, Any] = field(default_factory=dict)
    tools: list[str] = field(default_factory=list)
    other_agents: list[str] = field(default_factory=list)
    data_sources: list[str] = field(default_factory=list)
```

**Proposed Enhancement**:
```python
@dataclass
class AgentDependencies:
    """Dependencies injected into Pydantic AI agents."""
    # keep existing fields
    config: dict[str, Any] = field(default_factory=dict)
    tools: list[str] = field(default_factory=list)
    other_agents: list[str] = field(default_factory=list)
    data_sources: list[str] = field(default_factory=list)

    # ===== MEMORY SYSTEM FIELDS (NEW - PROPOSED) =====
    memory_client: MemorySystemClient | None = None
    memory_config: MemoryConfig | None = None
    session_id: str | None = None
    agent_profile: AgentProfile | None = None
```

**Rationale**:
- `memory_client`: Access to memory system (retrieve, store)
- `memory_config`: Agent-specific memory settings (top_k, filters, etc.)
- `session_id`: Link memories to workflow session
- `agent_profile`: Mario's Profile concept (BioinformaticsAgent priorities: papers > genes)

---

### Specialized Agent Types

**Agent classes found (examples)**:
1. **ParserAgent** - Parses user questions
2. **PlannerAgent** - Creates execution plans
3. **ExecutorAgent** - Executes tool-based plans
4. **SearchAgent** - Web search coordination
5. **RAGAgent** - Retrieval-augmented generation
6. **BioinformaticsAgent** - Multi-source bio data fusion
7. **DeepSearchAgent** - Deep web research
8. **EvaluatorAgent** - Result evaluation
9. **DeepAgentVariants** (5 types): Basic, Enhanced, Comprehensive, Experimental, Production

**Memory Needs Per Agent Type**:

| Agent Type | Memory Priorities | Top-K | Retrieval Strategy |
|------------|------------------|-------|-------------------|
| BioinformaticsAgent | papers, genes, blast_results | 15 | Hybrid (semantic + metadata) |
| PRIMEAgent | tool_history, molecular_constraints | 10 | Recency-biased |
| PlannerAgent | previous_plans, execution_outcomes | 5 | Success-weighted |
| ExecutorAgent | tool_failures, adaptive_replans | 8 | Failure-pattern matching |
| DeepSearchAgent | visited_urls, search_queries | 12 | Deduplication-aware |

---

### Tool Registration Pattern

**Location**: `DeepResearch/agents.py:149-169`

```python
@abstractmethod
def _register_tools(self):
    """Register tools with the agent."""
    @self._agent.tool
    def tool_name(ctx: RunContext[AgentDependencies], param: str) -> str:
        # Tool implementation
        return result
```

### Memory Integration Point #3: Memory as Pydantic AI Tools

**Proposed Memory Tools**:
```python
def _register_memory_tools(self):
    """Register memory system tools."""

    @self._agent.tool
    async def retrieve_memory(
        ctx: RunContext[AgentDependencies],
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Retrieve relevant memories for the current task.

        Args:
            query: Semantic search query
            top_k: Number of memories to retrieve
            filters: Optional metadata filters (e.g., {"agent_type": "planner"})

        Returns:
            List of memory records with content and metadata
        """
        if not ctx.deps.memory_client:
            return []

        memories = await ctx.deps.memory_client.retrieve(
            query=query,
            top_k=top_k,
            filters=filters or {},
            session_id=ctx.deps.session_id
        )

        return [
            {
                "content": m.content,
                "metadata": m.metadata,
                "similarity": m.similarity_score,
                "timestamp": m.timestamp
            }
            for m in memories
        ]

    @self._agent.tool
    async def store_memory(
        ctx: RunContext[AgentDependencies],
        content: str,
        metadata: dict[str, Any] | None = None,
        importance: float = 0.5
    ) -> dict[str, Any]:
        """Store a new memory for future retrieval.

        Args:
            content: Text content to store
            metadata: Optional metadata (tags, agent type, etc.)
            importance: Importance score (0.0 to 1.0)

        Returns:
            Storage confirmation with memory ID
        """
        if not ctx.deps.memory_client:
            return {"success": False, "reason": "Memory client not available"}

        memory_id = await ctx.deps.memory_client.store(
            content=content,
            metadata={
                **(metadata or {}),
                "agent_type": ctx.deps.agent_profile.agent_type if ctx.deps.agent_profile else "unknown",
                "session_id": ctx.deps.session_id,
                "importance": importance
            }
        )

        return {
            "success": True,
            "memory_id": memory_id,
            "timestamp": datetime.now().isoformat()
        }
```

**Rationale**: Agents explicitly control memory access (no magic background retrieval).

---

## Workflow Patterns (Pydantic Graph)

### Workflow types in `app.py` (current)

**Location**: `DeepResearch/app.py` (Lines 189-1072)

1. **PrimaryREACTWorkflow** (Lines 189-454)
   - Spawns subworkflows
   - Tracks execution state
   - Generates comprehensive output

2. **EnhancedREACTWorkflow** (Lines 457-706)
   - Supports 4 modes: SINGLE_REACT, MULTI_LEVEL_REACT, NESTED_ORCHESTRATION, LOSS_DRIVEN
   - Break conditions and loss functions

3. **Default Workflow** (Lines 710-749)
   - Search → Analyze → Synthesize

4. **Challenge Flow** (Lines 751-779)
   - PrepareChallenge → RunChallenge → EvaluateChallenge

5. **DeepSearch Flow** (Lines 781-841)
   - DSPlan → DSExecute → DSAnalyze → DSSynthesize

6. **PRIME Flow** (Lines 843-996)
   - PrimeParse → PrimePlan → PrimeExecute → PrimeEvaluate

7. **Bioinformatics Flow** (Lines 998-1035)
   - BioinformaticsParse → BioinformaticsFuse

8. **RAG Flow** (Lines 1037-1072)
   - RAGParse → RAGExecute

### Workflow Execution Pattern

```mermaid
stateDiagram-v2
    [*] --> Plan
    Plan --> Route
    Route --> Execute_PRIME: flows.prime.enabled
    Route --> Execute_Bio: flows.bioinformatics.enabled
    Route --> Execute_RAG: flows.rag.enabled
    Route --> Execute_DeepSearch: flows.deepsearch.enabled
    Route --> Execute_Default: No specific flow

    Execute_PRIME --> Synthesize
    Execute_Bio --> Synthesize
    Execute_RAG --> Synthesize
    Execute_DeepSearch --> Synthesize
    Execute_Default --> Synthesize

    Synthesize --> [*]

    note right of Plan
        MEMORY: Retrieve previous plans
    end note

    note right of Execute_PRIME
        MEMORY: Store tool executions
    end note

    note right of Synthesize
        MEMORY: Store final result
    end note
```

---

### Nested Loop & Subgraph Spawning

**Location**: `DeepResearch/src/agents/agent_orchestrator.py:37-150`

```python
@dataclass
class AgentOrchestrator:
    config: AgentOrchestratorConfig
    nested_loops: dict[str, NestedReactConfig] = field(default_factory=dict)
    subgraphs: dict[str, SubgraphConfig] = field(default_factory=dict)
    active_loops: dict[str, Any] = field(default_factory=dict)
    execution_history: list[dict[str, Any]] = field(default_factory=list)

    def _register_orchestrator_tools(self):
        @self.orchestrator_agent.tool
        def spawn_nested_loop(
            ctx: RunContext[OrchestratorDependencies],
            loop_id: str,
            state_machine_mode: str,
            max_iterations: int = 10
        ) -> dict[str, Any]:
            nested_config = NestedReactConfig(...)
            self.nested_loops[loop_id] = nested_config
            return loop_result
```

### Memory Integration Point #4: Nested Loop Memory Context

**Proposed Enhancement**:
```python
def spawn_nested_loop(...) -> dict[str, Any]:
    # Create nested memory context (inherits from parent)
    nested_memory_context = ctx.deps.memory_client.create_nested_context(
        parent_session_id=ctx.deps.session_id,
        loop_id=loop_id
    )

    # Execute nested loop with its own memory context
    nested_config = NestedReactConfig(
        ...,
        memory_session_id=nested_memory_context.session_id
    )

    # After execution, merge memories back to parent
    await ctx.deps.memory_client.merge_nested_context(nested_memory_context)
```

**Rationale**: Nested loops get isolated memory contexts but can share upwards.

---

## Configuration Structure (Hydra)

### Hydra Configuration Hierarchy

**Location**: `configs/`

```
configs/
├── config.yaml (main entry, 118 lines)
├── app_modes/ (4 configs: loss_driven, multi_level_react, etc.)
├── bioinformatics/ (agents, data_sources, tools, workflow, variants/)
├── challenge/ (default.yaml)
├── db/ (datasets, neo4j, postgres, sqlite3)
├── deep_agent/ (basic, comprehensive, default)
├── deepsearch/ (default.yaml)
├── llm/ (llamacpp_local, tgi_local, vllm_pydantic)
├── neo4j/ (orchestrator, operations/)
├── prompts/ (30+ prompt configs)
└── statemachines/flows/ (16+ flow configs)
```

### Main Configuration File

**Location**: `configs/config.yaml:1-118`

```yaml
defaults:
  - challenge: default
  - workflow_orchestration: default
  - db: neo4j
  - neo4j: orchestrator
  - _self_

question: "What is machine learning..."
retries: 3

# Workflow orchestration
workflow_orchestration:
  enabled: true
  primary_workflow:
    workflow_type: primary_react
    max_retries: 3
    timeout: 300.0
    parameters:
      max_iterations: 10
      enable_reflection: true
      enable_self_correction: true

# Legacy flows
flows:
  prime: {enabled: false}
  bioinformatics: {enabled: false}
  rag: {enabled: false}
  deepsearch: {enabled: false}

# Performance
performance:
  enable_parallel_execution: true
  enable_result_caching: true
  cache_ttl: 3600
  enable_workflow_optimization: true
```

---

### Memory Integration Point #5: Memory Configuration

**Proposed New Directory**: `configs/memory/`

```
configs/memory/
├── default.yaml        # Standard memory config
├── neo4j.yaml          # Neo4j backend settings
├── faiss.yaml          # FAISS local backend
├── postgres.yaml       # PostgreSQL backend
└── redis.yaml          # Redis caching layer
```

**configs/memory/default.yaml**:
```yaml
memory:
  # Core settings
  enabled: true
  backend: "neo4j"  # or "faiss", "postgres", "redis", "in_memory"
  session_ttl: 86400  # 1 day in seconds

  # Compression
  compression:
    enabled: true
    algorithm: "lz4"  # or "gzip", "zstd"
    min_size_bytes: 1024  # Only compress if content > 1KB

  # Retrieval settings
  retrieval:
    top_k: 5
    min_similarity: 0.7
    max_age_seconds: 604800  # 7 days
    enable_reranking: true
    reranking_model: "cross-encoder/ms-marco-MiniLM-L-6-v2"

  # Storage settings
  storage:
    batch_size: 32
    flush_interval: 300  # seconds
    enable_async_writes: true

  # Vector store settings
  vector_store:
    backend: "neo4j"  # or "faiss", "chroma", "qdrant"
    embedding_model: "sentence-transformers/all-mpnet-base-v2"
    embedding_dimension: 768
    index_type: "cosine"  # or "euclidean", "dot_product"

  # Agent-specific profiles
  profiles:
    bioinformatics:
      priorities: [papers, genes, blast_results, protein_sequences]
      max_items: 15
      relevance_threshold: 0.7
      enable_cross_reference_expansion: true

    prime:
      priorities: [tool_history, molecular_constraints, design_iterations]
      max_items: 10
      relevance_threshold: 0.8
      enable_temporal_weighting: true

    planner:
      priorities: [previous_plans, execution_outcomes, failure_patterns]
      max_items: 5
      relevance_threshold: 0.75
      enable_success_weighting: true

    executor:
      priorities: [tool_failures, adaptive_replans, parameter_tunings]
      max_items: 8
      relevance_threshold: 0.65
      enable_failure_pattern_matching: true

    deepsearch:
      priorities: [visited_urls, search_queries, extracted_snippets]
      max_items: 12
      relevance_threshold: 0.7
      enable_deduplication: true
```

**configs/memory/neo4j.yaml**:
```yaml
defaults:
  - default
  - _self_

memory:
  backend: "neo4j"

  neo4j:
    # Inherits from db.neo4j config
    use_existing_connection: true

    # Vector index settings
    vector_index:
      index_name: "memory_embeddings"
      node_label: "Memory"
      vector_property: "embedding"
      dimensions: 768
      metric: "cosine"

    # Graph settings
    graph:
      enable_relationships: true
      relationship_types:
        - "FOLLOWS"        # Sequential memory
        - "REFERENCES"     # Cross-references
        - "SIMILAR_TO"     # Semantic similarity
        - "CAUSED_BY"      # Causal relationships
        - "PART_OF"        # Hierarchical (nested loops)

    # Performance
    batch_size: 100
    connection_pool_size: 10
```

**Update configs/config.yaml**:
```yaml
defaults:
  - memory: default  # NEW
  - challenge: default
  - workflow_orchestration: default
  - db: neo4j
  - neo4j: orchestrator
  - _self_
```

**CLI Override Examples**:
```bash
# Enable memory with Neo4j
uv run deepresearch memory.enabled=true memory.backend=neo4j question="..."

# Use FAISS for local development
uv run deepresearch memory.backend=faiss question="..."

# Disable memory TTL (indefinite retention)
uv run deepresearch memory.session_ttl=null question="..."

# Adjust retrieval settings
uv run deepresearch memory.retrieval.top_k=10 memory.retrieval.min_similarity=0.8 question="..."

# Use specific agent profile
uv run deepresearch flows.prime.enabled=true memory.profiles.prime.max_items=20 question="..."
```

---

## Existing Storage Backends

### Neo4j Vector Store

**Location**: `DeepResearch/src/vector_stores/neo4j_vector_store.py`

**Capabilities**:
- ✅ Native vector indexing (Neo4j 5+)
- ✅ Graph relationships (FOLLOWS, REFERENCES, SIMILAR_TO)
- ✅ Connection pooling
- ✅ Batch operations
- ✅ Async/sync driver support

**Configuration**:
```python
class Neo4jConnectionConfig:
    uri: str  # e.g., "neo4j://localhost:7687"
    username: str
    password: str
    database: str
    auth_type: Neo4jAuthType
    encrypted: bool

class VectorIndexConfig:
    index_name: str
    node_label: str
    vector_property: str
    dimensions: int
    metric: VectorIndexMetric  # COSINE, EUCLIDEAN
```

**Interface**:
```python
async def add_documents(self, documents: list[Document]) -> list[str]
async def search(
    self,
    query: str,
    search_type: SearchType,
    retrieval_query: str | None = None
) -> list[SearchResult]
async def delete_documents(self, document_ids: list[str]) -> bool
```

**Memory Usage**:
- Store `MemoryDocument` (extends `Document`)
- Leverage graph relationships for temporal/causal memory
- Use vector index for semantic retrieval

---

### FAISS Vector Store

**Location**: `DeepResearch/src/vector_stores/faiss_vector_store.py`

**Capabilities**:
- ✅ In-memory indexing (IndexFlatL2 with IndexIDMap)
- ✅ Disk persistence (pickle + FAISS native)
- ✅ No network overhead (local-only)
- ✅ Fast exact nearest-neighbor search

**Configuration**:
```python
class FAISSVectorStoreConfig(VectorStoreConfig):
    index_path: str
    data_path: str
```

**Interface**:
```python
def _load() -> None  # Load from disk
def _save() -> None  # Persist to disk
async def add_documents(documents: list[Document]) -> list[str]
async def search(query: str, search_type: SearchType) -> list[SearchResult]
```

**Memory Usage**:
- Suitable for **local development** and **testing**
- No external dependencies (Docker, etc.)
- Limited to L2 distance (Euclidean)

---

### VectorStore Abstract Base

**Location**: `DeepResearch/src/datatypes/rag.py` (inherited)

```python
class VectorStore(ABC):
    @abstractmethod
    async def add_documents(self, documents: list[Document]) -> list[str]:
        """Add documents to the vector store."""
        pass

    @abstractmethod
    async def search(
        self, query: str, search_type: SearchType
    ) -> list[SearchResult]:
        """Search for relevant documents."""
        pass

    @abstractmethod
    async def delete_documents(self, document_ids: list[str]) -> bool:
        """Delete documents by ID."""
        pass
```

**Implementations in repo**:
1. `Neo4jVectorStore` (graph + vectors)
2. `FAISSVectorStore` (local, in-memory/disk)

**Referenced elsewhere (not implemented here)**:
- ChromaDB, Qdrant, Pinecone, Weaviate, Postgres, Elasticsearch, Milvus

**Memory Integration**: **No changes needed** - memory system can use `VectorStore` interface directly.

---

### Document Structure (Perfect for Memory)

**Location**: `DeepResearch/src/datatypes/rag.py:76-200`

```python
class Document(BaseModel):
    id: str
    content: str
    chunks: list[Chunk] = []
    metadata: dict[str, Any]
    embedding: list[float] | Any | None
    created_at: datetime
    updated_at: datetime | None

    # Bioinformatics-specific (reusable for memory)
    bioinformatics_type: str | None
    source_database: str | None
    cross_references: dict[str, list[str]]
    quality_score: float | None
```

**Memory Integration**: Create `MemoryDocument` subclass:

```python
class MemoryDocument(Document):
    """Document specialized for memory storage."""

    # Override bioinformatics fields for memory-specific fields
    memory_type: str  # "execution_trace", "decision", "tool_result", etc.
    agent_type: str | None  # "planner", "executor", "bioinformatics", etc.
    session_id: str
    importance_score: float  # 0.0 to 1.0
    temporal_weight: float  # Decay over time

    # Relationships
    follows: str | None  # ID of previous memory
    caused_by: list[str] = []  # IDs of causal memories
    references: list[str] = []  # IDs of related memories
```

---

## Tool Ecosystem

### ToolRegistry & Execution Pattern

**Location**: `DeepResearch/src/tools/base.py:10-64`

```python
@dataclass
class ToolSpec:
    name: str
    description: str
    inputs: dict[str, str]  # param: type
    outputs: dict[str, str]  # key: type

@dataclass
class ExecutionResult:
    success: bool
    data: dict[str, Any]
    metrics: dict[str, Any]
    error: str | None

class ToolRunner:
    spec: ToolSpec

    def validate(self, params: dict[str, Any]) -> tuple[bool, str | None]:
        """Validate input parameters."""
        pass

    def run(self, params: dict[str, Any]) -> ExecutionResult:
        """Execute the tool."""
        pass

class ToolRegistry:
    def register(self, name: str, factory: Callable[[], ToolRunner]):
        """Register a tool factory."""
        pass

    def make(self, name: str) -> ToolRunner:
        """Instantiate a tool."""
        pass

    def list(self) -> list[str]:
        """List all registered tools."""
        pass
```

**Tools in registry (examples, not exhaustive)**:
- **Web Search**: ChunkedSearchTool, WebSearchTool
- **Bioinformatics**: GOAnnotationTool, PubMedRetrievalTool, 18 MCP servers
- **DeepSearch**: QueryRewriterTool, URLVisitTool, ReflectionTool
- **RAG**: RAGSearchTool, IntegratedSearchTool
- **Workflow**: EvaluatorTool, ErrorAnalyzerTool

---

### Memory Integration Point #6: Tool Execution Tracking

**Proposed Enhancement to ToolRunner**:

```python
class ToolRunner:
    spec: ToolSpec
    memory_client: MemorySystemClient | None = None  # NEW

    def run(self, params: dict[str, Any]) -> ExecutionResult:
        start_time = time.time()

        # Validate
        valid, error = self.validate(params)
        if not valid:
            result = ExecutionResult(success=False, error=error)
            self._store_execution_memory(params, result, time.time() - start_time)
            return result

        # Execute
        try:
            result = self._execute(params)
        except Exception as e:
            result = ExecutionResult(success=False, error=str(e))

        # MEMORY INTEGRATION: Store execution trace
        duration = time.time() - start_time
        self._store_execution_memory(params, result, duration)

        return result

    def _store_execution_memory(
        self,
        params: dict[str, Any],
        result: ExecutionResult,
        duration: float
    ) -> None:
        """Store tool execution in memory system."""
        if not self.memory_client:
            return

        content = f"Tool {self.spec.name} executed with params: {params}"
        if result.success:
            content += f"\nResult: {result.data}"
        else:
            content += f"\nError: {result.error}"

        self.memory_client.store(
            content=content,
            metadata={
                "memory_type": "tool_execution",
                "tool_name": self.spec.name,
                "success": result.success,
                "duration": duration,
                "parameters": params,
                "result_summary": result.data if result.success else None,
                "error": result.error,
                "timestamp": datetime.now().isoformat()
            }
        )
```

**Rationale**: Every tool execution becomes a memory trace automatically.

---

### MCP Bioinformatics Servers (18 Total)

**Location**: `DeepResearch/src/tools/bioinformatics/`

**Servers Available**:
1. `bwa_server.py` - BWA alignment
2. `bowtie2_server.py` - Bowtie2 alignment
3. `hisat2_server.py` - HISAT2 RNA-seq alignment
4. `star_server.py` - STAR fast alignment
5. `salmon_server.py` - Salmon transcript quantification
6. `kallisto_server.py` - Kallisto quantification
7. `bcftools_server.py` - VCF processing
8. `haplotypecaller_server.py` - GATK variant calling
9. `freebayes_server.py` - FreeBayes variant calling
10. `fastp_server.py` - Quality control
11. `fastqc_server.py` - FastQC quality reports
12. `multiqc_server.py` - MultiQC aggregation
13. `qualimap_server.py` - Qualimap quality assessment
14. `trimgalore_server.py` - Adapter trimming
15. `stringtie_server.py` - Transcript assembly
16. `featurecounts_server.py` - Read counting
17. `gunzip_server.py` - Decompression
18. Plus more...

**FastMCP Pattern**:
```python
from fastmcp import FastMCP

server = FastMCP("tool_name_server")

@server.tool()
def operation(param1: str, param2: int) -> dict[str, Any]:
    """Structured input/output for bioinformatics operations."""
    return {"result": data}
```

**Memory Integration**: MCP servers can record execution via context injection.

---

### ExecutionHistory - Already Tracks Everything!

**Location**: `DeepResearch/src/utils/execution_history.py:40-150`

```python
@dataclass
class ExecutionItem:
    step_name: str
    tool: str
    status: ExecutionStatus
    result: dict[str, Any] | None
    error: str | None
    timestamp: float
    parameters: dict[str, Any] | None
    duration: float | None
    retry_count: int

@dataclass
class ExecutionHistory:
    items: list[ExecutionItem]
    start_time: float
    end_time: float | None

    def add_item(self, item: ExecutionItem) -> None:
        """Add execution item."""
        pass

    def get_successful_steps(self) -> list[ExecutionItem]:
        """Get all successful executions."""
        pass

    def get_failed_steps(self) -> list[ExecutionItem]:
        """Get all failed executions."""
        pass

    def get_failure_patterns(self) -> dict[str, int]:
        """Analyze failure patterns."""
        pass

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        pass

    def save_to_file(self, filepath: str) -> None:
        """Save to JSON file."""
        pass
```

**Memory Integration Point #7: ExecutionHistory → Memory Pipeline**

```python
class ExecutionHistoryMemoryBridge:
    """Bridge between ExecutionHistory and Memory System."""

    def __init__(self, memory_client: MemorySystemClient):
        self.memory_client = memory_client

    async def ingest_execution_history(
        self,
        history: ExecutionHistory,
        session_id: str
    ) -> int:
        """Ingest all execution items into memory system.

        Returns:
            Number of memories stored
        """
        memories_stored = 0

        for item in history.items:
            # Create memory document
            content = f"Tool {item.tool} - Step {item.step_name}"
            if item.status == ExecutionStatus.SUCCESS:
                content += f"\nResult: {item.result}"
            else:
                content += f"\nError: {item.error}"

            # Store with rich metadata
            await self.memory_client.store(
                content=content,
                metadata={
                    "memory_type": "execution_trace",
                    "tool": item.tool,
                    "step_name": item.step_name,
                    "status": item.status.value,
                    "timestamp": item.timestamp,
                    "duration": item.duration,
                    "retry_count": item.retry_count,
                    "session_id": session_id,
                    "success": item.status == ExecutionStatus.SUCCESS
                }
            )
            memories_stored += 1

        # Also store failure patterns as a separate memory
        failure_patterns = history.get_failure_patterns()
        if failure_patterns:
            await self.memory_client.store(
                content=f"Failure patterns detected: {failure_patterns}",
                metadata={
                    "memory_type": "failure_analysis",
                    "patterns": failure_patterns,
                    "session_id": session_id,
                    "importance_score": 0.8  # High importance
                }
            )
            memories_stored += 1

        return memories_stored
```

**Rationale**: ExecutionHistory is a **goldmine** for memory - it already tracks everything we need!

---

## Testing Infrastructure

### Pytest Configuration

**Location**: `tests/conftest.py:28-70`

**Test Markers**:
```python
config.addinivalue_line("markers", "unit: Unit tests")
config.addinivalue_line("markers", "integration: Integration tests")
config.addinivalue_line("markers", "performance: Performance tests")
config.addinivalue_line("markers", "containerized: Tests requiring containers")
config.addinivalue_line("markers", "slow: Slow-running tests")
config.addinivalue_line("markers", "bioinformatics: Bioinformatics-specific tests")
config.addinivalue_line("markers", "llm: LLM framework tests")
config.addinivalue_line("markers", "pydantic_ai: Pydantic AI agent tests")
```

**Environment Variables**:
- `CI` - Running in CI/CD
- `DOCKER_TESTS` - Enable containerized tests
- `PERFORMANCE_TESTS` - Enable performance benchmarks
- `INTEGRATION_TESTS` - Enable integration tests

---

### Test Categories Found

**Test Files**:
- `tests/vector_stores/test_faiss_vector_store.py`
- `tests/test_basic.py`
- `tests/test_pubmed_retrieval.py`
- `tests/test_refactoring_verification.py`
- `tests/test_matrix_functionality.py`
- `tests/test_examples/` (genomics_agent, MCP tests)
- `tests/test_pydantic_ai/` (agent and tool integration)
- `tests/test_llm_framework/` (llamacpp, vllm containerized)

---

### Memory Testing Strategy

**Proposed Test Structure**: `tests/test_memory/`

```
tests/test_memory/
├── test_memory_client.py           # Unit tests for memory client
├── test_memory_backends.py         # Backend-specific tests (Neo4j, FAISS)
├── test_memory_retrieval.py        # Retrieval strategies (MMR, similarity)
├── test_memory_profiles.py         # Agent profile filtering
├── test_memory_integration.py      # Integration with agents/workflows
├── test_execution_history_bridge.py # ExecutionHistory → Memory
└── test_memory_performance.py      # Performance benchmarks
```

**Example Test**:
```python
# tests/test_memory/test_memory_client.py
import pytest
from DeepResearch.src.memory.client import MemorySystemClient
from DeepResearch.src.memory.config import MemoryConfig

@pytest.fixture
def memory_client():
    """Create in-memory client for testing."""
    config = MemoryConfig(backend="in_memory", session_ttl=3600)
    return MemorySystemClient(config)

@pytest.mark.asyncio
async def test_store_and_retrieve(memory_client):
    """Test basic store and retrieve operations."""
    # Store a memory
    memory_id = await memory_client.store(
        content="Test execution of BWA tool",
        metadata={"tool": "bwa", "status": "success"}
    )
    assert memory_id is not None

    # Retrieve it
    memories = await memory_client.retrieve(
        query="BWA tool execution",
        top_k=5
    )
    assert len(memories) > 0
    assert memories[0].content == "Test execution of BWA tool"
    assert memories[0].metadata["tool"] == "bwa"

@pytest.mark.asyncio
async def test_profile_filtering(memory_client):
    """Test agent profile-based filtering."""
    # Store memories with different agent types
    await memory_client.store(
        content="Planner created execution plan",
        metadata={"agent_type": "planner"}
    )
    await memory_client.store(
        content="Executor ran BWA tool",
        metadata={"agent_type": "executor"}
    )

    # Retrieve only planner memories
    memories = await memory_client.retrieve(
        query="execution",
        filters={"agent_type": "planner"},
        top_k=5
    )
    assert len(memories) == 1
    assert "Planner" in memories[0].content

@pytest.mark.integration
@pytest.mark.asyncio
async def test_neo4j_backend(neo4j_connection):
    """Test Neo4j backend integration."""
    config = MemoryConfig(backend="neo4j")
    client = MemorySystemClient(config, neo4j_connection)

    # Test graph relationships
    parent_id = await client.store(content="Parent memory")
    child_id = await client.store(
        content="Child memory",
        metadata={"follows": parent_id}
    )

    # Query graph relationship
    related = await client.get_related_memories(parent_id)
    assert len(related) == 1
    assert related[0].id == child_id
```

---

## Memory Integration Points Map

### Comprehensive Integration Architecture

```mermaid
graph TB
    subgraph "1. Configuration Layer"
        HydraConfig[Hydra Config<br/>configs/memory/]
        MemoryConfig[MemoryConfig<br/>Backend, TTL, Profiles]
    end

    subgraph "2. State Layer"
        ResearchState[ResearchState<br/>+ execution_context<br/>+ memory_session_id]
        ExecutionContext[ExecutionContext<br/>memory_client, config]
    end

    subgraph "3. Agent Layer"
        AgentDeps[AgentDependencies<br/>+ memory_client<br/>+ memory_config<br/>+ session_id]
        AgentTools[Agent Tools<br/>retrieve_memory()<br/>store_memory()]
    end

    subgraph "4. Workflow Layer"
        GraphNodes[Pydantic Graph Nodes<br/>Plan, Execute, Synthesize]
        NestedLoops[Nested Loops<br/>Hierarchical Memory Context]
    end

    subgraph "5. Tool Layer"
        ToolRunner[ToolRunner<br/>Auto-store executions]
        ExecHistory[ExecutionHistory<br/>→ Memory Bridge]
    end

    subgraph "6. Memory System Core"
        MemoryClient[MemorySystemClient<br/>store(), retrieve()]
        MemoryProfiles[AgentProfile Filter<br/>Bioinformatics, PRIME, etc.]
    end

    subgraph "7. Storage Layer"
        VectorStore[VectorStore ABC<br/>Unified Interface]
        Neo4jStore[Neo4j Vector Store<br/>Graph + Vectors]
        FAISSStore[FAISS Vector Store<br/>Local In-Memory]
    end

    HydraConfig --> MemoryConfig
    MemoryConfig --> ExecutionContext
    ExecutionContext --> ResearchState
    ResearchState --> GraphNodes
    GraphNodes --> AgentDeps
    AgentDeps --> AgentTools
    AgentTools --> MemoryClient
    GraphNodes --> ToolRunner
    ToolRunner --> ExecHistory
    ExecHistory --> MemoryClient
    GraphNodes --> NestedLoops
    NestedLoops --> MemoryClient
    MemoryClient --> MemoryProfiles
    MemoryProfiles --> VectorStore
    VectorStore --> Neo4jStore
    VectorStore --> FAISSStore

    style ResearchState fill:#ffeb3b,stroke:#f57c00,stroke-width:3px
    style AgentDeps fill:#4caf50,stroke:#2e7d32,stroke-width:3px
    style ExecHistory fill:#2196f3,stroke:#1565c0,stroke-width:3px
    style MemoryClient fill:#e91e63,stroke:#c2185b,stroke-width:3px
    style VectorStore fill:#9c27b0,stroke:#6a1b9a,stroke-width:3px
```

---

### Integration Points Summary Table

| # | Integration Point | Location | Type | Complexity | Priority |
|---|------------------|----------|------|------------|----------|
| 1 | ResearchState Fields | `app.py:45-96` | State | Low | P0 (Critical) |
| 2 | AgentDependencies | `datatypes/agents.py:28-34` | DI | Low | P0 (Critical) |
| 3 | Memory Tools | `agents.py:149-169` | Tools | Medium | P1 (High) |
| 4 | Nested Loop Context | `agent_orchestrator.py:37-150` | Orchestration | High | P2 (Medium) |
| 5 | Memory Config | `configs/` | Config | Low | P0 (Critical) |
| 6 | Tool Execution Tracking | `tools/base.py:10-64` | Tools | Medium | P1 (High) |
| 7 | ExecutionHistory Bridge | `utils/execution_history.py` | Bridge | Medium | P1 (High) |
| 8 | Graph Node Hooks | `app.py:99-1072` | Workflow | Low | P0 (Critical) |

**Total Integration Points**: 8
**Critical Path**: Points #1, #2, #5, #8 (State, Dependencies, Config, Node Hooks)
**Quick Wins**: Points #6, #7 (Tool tracking, ExecutionHistory bridge - already structured!)

---

## Recommendations

### Phase 1 Findings: Key Insights

1. **Architecture is Memory-Ready**
   - No major refactoring needed
   - All hooks already exist at every layer
   - VectorStore abstraction perfect for memory backends

2. **ExecutionHistory is a Goldmine**
   - Already tracks all tool executions
   - Serializable to JSON
   - Perfect for memory ingestion
   - **Quick win**: Bridge ExecutionHistory → Memory in <100 lines of code

3. **Neo4j is the Natural Choice**
   - Already configured and in use
   - Supports vectors + graph relationships
   - Can model temporal/causal memory (FOLLOWS, CAUSED_BY, etc.)
   - No new dependencies needed

4. **Agent Profiles Make Sense**
   - Different agents need different memory priorities
   - BioinformaticsAgent: papers > genes
   - PRIMEAgent: tool_history > chat
   - Configuration-driven (no code changes needed)

5. **Hydra Config Integration is Trivial**
   - Create `configs/memory/` directory
   - Add to defaults in `config.yaml`
   - Support CLI overrides
   - **Effort**: <1 hour

---

### Recommended Implementation Path (Aligns with Meta Plan Phase 4)

#### Vertical Slice 1: Minimal Viable Memory (MVP)
**Goal**: Prove the integration works end-to-end

**Tasks**:
1. Create `MemoryConfig` dataclass
2. Create `ExecutionContext` dataclass
3. Add fields to `ResearchState` and `AgentDependencies`
4. Implement in-memory backend (dict-based, no persistence)
5. Add memory tools to one agent (e.g., PlannerAgent)
6. Test in simple workflow (Plan → Execute → Synthesize)

**Acceptance Criteria**:
- Agent can `retrieve_memory()` and `store_memory()`
- Memories persist across nodes in same workflow run
- Tests pass

**Effort**: 4-6 hours
**Risk**: Low

---

#### Vertical Slice 2: ExecutionHistory Bridge
**Goal**: Automatically ingest tool executions into memory

**Tasks**:
1. Create `ExecutionHistoryMemoryBridge` class
2. Hook into `ExecutionHistory.add_item()`
3. Convert `ExecutionItem` → `MemoryDocument`
4. Test with bioinformatics workflow (lots of tool executions)

**Acceptance Criteria**:
- All tool executions auto-stored in memory
- Can retrieve previous tool failures
- Adaptive replanning can use memory

**Effort**: 3-4 hours
**Risk**: Low

---

#### Vertical Slice 3: Neo4j Backend Integration
**Goal**: Production-ready storage with graph relationships

**Tasks**:
1. Create `Neo4jMemoryBackend` adapter
2. Use existing `Neo4jVectorStore` infrastructure
3. Add graph relationships (FOLLOWS, CAUSED_BY, etc.)
4. Migrate in-memory data to Neo4j
5. Test retrieval with graph queries

**Acceptance Criteria**:
- Memories stored in Neo4j
- Can query temporal relationships (what happened before X?)
- Can query causal relationships (what caused Y?)
- Vector similarity search works

**Effort**: 6-8 hours
**Risk**: Medium (Neo4j schema design)

---

#### Vertical Slice 4: Agent Profiles & Filtering
**Goal**: Mario's profile-based selective memory

**Tasks**:
1. Create `AgentProfile` dataclass
2. Implement profile-based filtering in retrieval
3. Create profiles for BioinformaticsAgent, PRIMEAgent, etc.
4. Add profile configs to `configs/memory/`
5. Test with multiple agents (verify each gets relevant memories)

**Acceptance Criteria**:
- BioinformaticsAgent retrieves papers/genes (not tool history)
- PRIMEAgent retrieves tool history (not papers)
- Configurable via YAML

**Effort**: 5-6 hours
**Risk**: Low

---

#### Vertical Slice 5: Advanced Retrieval (MMR, Reranking)
**Goal**: High-quality retrieval with deduplication

**Tasks**:
1. Implement MMR (Maximal Marginal Relevance)
2. Add cross-encoder reranking
3. Temporal weighting (recent memories > old)
4. Importance weighting (critical decisions > routine)
5. Benchmark retrieval quality

**Acceptance Criteria**:
- Retrieval returns diverse results (MMR)
- Reranking improves relevance
- Temporal decay works
- Importance scores affect ranking

**Effort**: 8-10 hours
**Risk**: Medium (algorithm complexity)

---

### Total Estimated Effort: 26-34 hours (~1 week for experienced dev)

**Critical Success Factors**:
1. **Start with Slice 1** - Prove integration works before adding complexity
2. **Test each slice** - Don't move to next until current passes all tests
3. **Use existing abstractions** - Don't reinvent VectorStore, Document, etc.
4. **Leverage ExecutionHistory** - It's already tracking what we need

---

## Next Steps for Mario

### Immediate Actions

1. **Review This Document**
   - Does the integration map make sense?
   - Any missing integration points?
   - Does vertical slice approach feel right?

2. **Validate Assumptions**
   - Is Neo4j the right backend choice?
   - Are agent profiles correctly identified?
   - Is ExecutionHistory bridge approach sound?

3. **Prototype Slice 1**
   - Implement MVP in a branch
   - Test with simple workflow
   - Get feedback from team

4. **Iterate Based on Learnings**
   - Adjust approach based on prototype
   - Update vertical slices as needed
   - Ship incrementally

---

## Appendix: File Inventory

### Core architecture touchpoints (not exhaustive)
- **State & workflows**: `DeepResearch/app.py` (graph, ResearchState, node routing); statemachines under `DeepResearch/src/statemachines/` (prime/bioinformatics/rag/deepsearch/code_execution, etc.).
- **Agents**: `DeepResearch/agents.py` and `DeepResearch/src/agents/` (planner, parser, executor, bioinformatics, deepsearch, orchestrators, deep_agent variants).
- **Datatypes**: `DeepResearch/src/datatypes/` (agents.py deps/result/history, execution.py, rag.py Document/VectorStoreType, deep_agent_state.py, workflow_orchestration.py, pydantic_ai_tools.py, neo4j_types.py, etc.).
- **Tools**: `DeepResearch/src/tools/base.py` plus registries; bioinformatics MCP servers under `DeepResearch/src/tools/bioinformatics/`; web/deepsearch/integrated search tools nearby.
- **Vector stores**: Implemented in `DeepResearch/src/vector_stores/neo4j_vector_store.py` and `faiss_vector_store.py` (others referenced in docs/configs).
- **Execution tracking**: `DeepResearch/src/utils/execution_history.py`; tool registry variant at `DeepResearch/src/utils/tool_registry.py`.
- **Pydantic AI utils**: `DeepResearch/src/utils/pydantic_ai_utils.py` (agent/toolset builders).
- **Configuration**: `configs/config.yaml` entrypoint; flow configs under `configs/statemachines/flows/`; no `configs/memory/` yet.

---

## Conclusion

The DeepCritical/DeepResearch codebase is **exceptionally well-architected** for memory system integration. All necessary hooks exist:

✅ **State Management**: ResearchState flows through all nodes
✅ **Dependency Injection**: AgentDependencies ready for memory client
✅ **Tool Tracking**: ExecutionHistory already records everything
✅ **Storage Abstraction**: VectorStore interface supports any backend
✅ **Configuration**: Hydra composition makes memory config trivial

**No major refactoring needed** - memory can be integrated incrementally via vertical slices.

**Recommended Path**: Start with MVP (Slice 1), validate approach, then ship remaining slices iteratively.

**Mario's architecture is sound** - this document provides the confidence and roadmap to ship it.

---

**Status**: 🟢 Ready for Senior Review
**Next**: Phase 2 - Research best memory architectures (Nov 2025)
