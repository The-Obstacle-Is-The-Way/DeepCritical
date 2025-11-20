# PHASE 2: Memory Architectures Research (Nov 2025)

**Goal**: Survey the CURRENT (November 2025) state-of-the-art memory systems for agentic AI, with eyes on DeepCritical/DeepResearch codebase integration.

**Philosophy**: Focus on production-ready systems with proven performance, not experimental frameworks. Evaluate how each architecture fits our Phase 1 baseline: Hydra + Pydantic Graph + Pydantic AI multi-agent system with ~28 bioinformatics `_server.py` modules (FastMCP/CLI, mixed completeness) and tool-heavy workflows.

---

## Executive Summary

**Systems Evaluated in Detail**: 6 memory architectures (4 with public code, 2 pattern-only references):
1. **Production-Ready Platforms**: Mem0, Letta (MemGPT), Zep
2. **Novel Research Systems**: G-Memory (code available)
3. **Pattern References**: O-Mem (paper-only), H-MEM (paper-only)

**Systems Excluded**: MemOS (corrupted docs, no benchmarks), LangMem (LangGraph-only, incompatible with Pydantic Graph)

**Key Finding**: No single system wins across all dimensions. The choice depends on:
- **Multi-agent coordination needs**: G-Memory's hierarchical graphs shine
- **Production maturity**: Mem0 (SaaS) and Letta (open-source) are battle-tested
- **Temporal reasoning**: Zep's bi-temporal knowledge graphs are cutting-edge
- **Efficiency at scale**: H-MEM (paper-only) claims fastest hierarchical retrieval (<100ms) but has no public code
- **Hybrid flexibility**: Mem0's graph+vector+KV combo offers versatility

**Potential Hybrid Approach**: Combine architectural patterns (e.g., Mem0's hybrid storage + G-Memory's multi-agent hierarchy + Zep's temporal modeling) rather than marrying platforms.

**Official GitHub Repositories**: See `referencerepos.md` for verified links to all source repos
**Local Reference Copies**: All available systems cloned to `/reference_repo/` (`.git` directories removed)

**Code Availability Audit (Nov 19, 2025)**:
- ✅ **Evaluated with code**: Mem0, Letta, Zep, G-Memory (see `reference_repo/`)
- ❌ **No public code** (pattern references only): O-Mem, H-MEM
- 🗑️ **Excluded from evaluation**: MemOS (corrupted docs, no benchmarks, 3 confusing variants), LangMem (LangGraph-only, incompatible)
- 📦 **In reference_repo but not evaluated**: A-MEM, MemEngine, OpenMemory, Memori (may evaluate in Phase 5+ if needed)

---

## 1. Mem0: Hybrid Graph+Vector+KV Architecture

**Source**: arXiv 2504.19413 (April 2025), Y Combinator-backed, production SaaS + OSS

**Official GitHub**: https://github.com/mem0ai/mem0 (43,252 stars, Apache 2.0)
**arXiv Paper**: https://arxiv.org/abs/2504.19413
**Local Reference**: `/reference_repo/mem0-official/`

### Architecture

**Three-Component Hybrid Datastore**:
1. **Vector Store**: Embeddings for semantic similarity search
2. **Graph Store**: Entity relationships (directed labeled graph G=(V,E,L))
   - Nodes (V): Entities extracted from conversations
   - Edges (E): Relationships between entities
   - Labels (L): Semantic types for nodes
3. **Key-Value Store**: Fast lookups for structured facts, preferences, metadata

**Memory Pipeline**:
- **Extraction**: LLM dynamically extracts salient information from conversations
- **Consolidation**: Deduplication and entity resolution across stores
- **Retrieval**: Hybrid search across vector (semantic), graph (relational), and KV (exact match)

**Variants**:
- **Mem0 (Base)**: Uses vector + KV stores
- **Mem0ᵍ (Graph-Enhanced)**: Adds graph store for complex relational reasoning

### Performance Benchmarks

**LoCoMo Dataset** (5,000+ QA pairs, 4 categories: single-hop, temporal, multi-hop, open-domain):
| System | Accuracy | Median Latency | p95 Latency | Token Savings |
|--------|----------|----------------|-------------|---------------|
| OpenAI Memory | 52.9% | - | - | Baseline |
| Mem0 (Base) | 66.9% | 0.71s | 1.44s | 90% |
| Mem0ᵍ (Graph) | 68.4% | 1.09s | 2.59s | 90% |

- **26% accuracy improvement** over OpenAI Memory
- **91% lower p95 latency** vs. full-context baseline (115k tokens)
- **Graph variant adds ~2% accuracy** for temporal/multi-hop queries

### Integration Patterns

**Embedding Support**: OpenAI, Azure, Anthropic, Google, Hugging Face, Ollama, Together, Groq
**Vector Databases**: Qdrant, Chroma, Milvus, Pgvector, Redis, Azure AI Search
**Graph Databases**: Neo4j, AWS Neptune, Kuzu
**Framework Integration**: LangChain, LlamaIndex, CrewAI, MultiOn

**Deployment**: Managed SaaS (Mem0 Cloud) + self-hosted OSS

### Pros (For Our Codebase)

✅ **Production-ready**: Y Combinator backing, mature SaaS offering, extensive benchmarks
✅ **Hybrid flexibility**: Graph+vector+KV covers semantic, relational, and exact-match retrieval
✅ **Neo4j compatibility**: Already have Neo4j in Phase 1 baseline (currently for RAG)
✅ **Multi-framework support**: Works with LangChain, LlamaIndex (could bridge to Pydantic AI)
✅ **Token efficiency**: 90% savings critical for tool-heavy workflows (~28 bioinformatics modules + other tools)
✅ **Profile-friendly**: Graph structure naturally maps to Mario's agent profiles (BioinformaticsAgent needs papers/genes, PRIMEAgent needs tool_history)

### Cons

❌ **SaaS-first design**: OSS version may lag behind cloud features
❌ **LLM-driven extraction**: Adds latency and cost for real-time memory updates
❌ **Pydantic AI integration unclear**: No documented patterns (LangChain/LlamaIndex focus)
❌ **Multi-agent coordination**: Designed for single-agent contexts, not hierarchical orchestration
❌ **Consolidation complexity**: Deduplication across 3 stores could introduce bugs

### Suitability for DeepCritical/DeepResearch

**Score: 8/10**

Strong fit due to Neo4j compatibility, hybrid storage, and production maturity. Main concerns: multi-agent gaps and Pydantic AI integration.

---

## 2. Letta (MemGPT): Hierarchical Self-Editing Memory

**Source**: Open-source framework (MemGPT paper lineage), community-driven, Letta Cloud SaaS

**Official GitHub**: https://github.com/letta-ai/letta (Apache 2.0)
**Website**: https://www.letta.com/
**Local Reference**: `/reference_repo/letta-official/`

### Architecture

**OS-Inspired Memory Hierarchy**:
1. **Core Memory**: Immediate context within token limit (system prompts, active messages)
2. **Conversational Memory**: Recent interaction history (auto-summarized)
3. **Archival Memory**: Long-term storage with embeddings for retrieval
4. **External Files**: Additional data layers (docs, knowledge bases)

**Memory Self-Editing**:
- LLM uses **tool calls** to modify its own memory blocks (add, update, delete)
- **Agent Development Environment (ADE)**: Visual interface for monitoring memory state, reasoning, and tool invocations in real-time

**Filesystem-Based Retrieval**:
- Uses file operations (grep, search_files, open, close) with semantic search
- Files auto-parsed and embedded for vector-based retrieval
- Agents iteratively search and refine queries autonomously

### Performance Benchmarks

**LoCoMo Dataset** (GPT-4o mini):
- **Letta**: 74.0% accuracy
- **Mem0**: 68.5% accuracy (as reported by Letta team)

**Letta outperforms Mem0 by 5.5 percentage points** on LoCoMo benchmark.

### Integration Patterns

**LLM Support**: Anthropic Claude (Sonnet 4.5), OpenAI (GPT-4o), local models
**Storage Backends**: Filesystem + embeddings, Postgres, custom stores
**Framework Integration**: Standalone (no tight coupling to LangChain/LlamaIndex)
**Deployment**: Letta Cloud (managed) + self-hosted OSS (Docker, local)

### Pros (For Our Codebase)

✅ **Open-source first**: Active community (Discord), no vendor lock-in
✅ **Agent-native design**: Memory as LLM-managed tool calls fits Pydantic AI patterns
✅ **Debugging-friendly**: ADE provides visibility into agent reasoning and memory state
✅ **Claude Sonnet 4.5 support**: Our primary LLM (Anthropic)
✅ **Filesystem simplicity**: Lower complexity than graph+vector+KV hybrid
✅ **Extensible**: Can add custom memory backends (e.g., Neo4j adapter)

### Cons

❌ **Filesystem bottleneck**: File-based retrieval may not scale to large knowledge bases
❌ **Less structured**: No native graph/relational modeling (vs. Mem0/Zep)
❌ **Multi-agent gaps**: Designed for single-agent contexts, not hierarchical orchestration
❌ **LoCoMo focus**: Benchmark emphasizes QA tasks, not tool-heavy workflows
❌ **Community-driven pace**: Slower feature velocity than SaaS offerings

### Suitability for DeepCritical/DeepResearch

**Score: 7/10**

Strong open-source option with agent-native design. Concerns: filesystem scalability, multi-agent coordination, and structured memory gaps.

---

## 3. Zep: Bi-Temporal Knowledge Graphs

**Source**: arXiv 2501.13956 (Jan 2025), Zep Cloud SaaS + Community OSS, SOC 2 compliant

**Official GitHub**: https://github.com/getzep/zep (Apache 2.0)
**arXiv Paper**: https://arxiv.org/abs/2501.13956
**Website**: https://www.getzep.com/
**Local Reference**: `/reference_repo/zep-official/`
**Note**: Community Edition deprecated (moved to `legacy/` folder), Zep Cloud recommended

### Architecture

**Three-Tier Hierarchical Subgraphs**:
1. **Episode Subgraph (𝒢ₑ)**: Raw episodic nodes (non-lossy data store)
   - Bidirectional indices to semantic entities
   - Forward tracing: episode → derived concepts
   - Backward retrieval: semantic entity → source messages
2. **Semantic Entity Subgraph (𝒢ₛ)**: Entity nodes + relationship edges
   - LLM-driven entity extraction and resolution
   - Hybrid search: embeddings (1024-dim) + full-text
3. **Community Subgraph (𝒢ᶜ)**: High-level summaries of strongly connected entity clusters
   - Inspired by GraphRAG's hierarchical Leiden community detection

**Bi-Temporal Model**:
- **Timeline T**: Chronological event ordering (when things actually happened)
- **Timeline T'**: Transactional ingestion sequence (when Zep learned about events)

Each fact edge tracks **4 timestamps**:
- `t'ᶜʳᵉᵃᵗᵉᵈ`, `t'ᵉˣᵖⁱʳᵉᵈ` (transaction timeline)
- `tᵥₐₗᵢₐ`, `tᵢₙᵥₐₗᵢₐ` (event validity period)

**Edge Invalidation**: LLM-driven contradiction detection → newer information takes priority (per T')

### Performance Benchmarks

**DMR Benchmark** (Deep Memory Retrieval):
| System | Model | Accuracy |
|--------|-------|----------|
| Zep | gpt-4-turbo | 94.8% |
| MemGPT | gpt-4-turbo | 93.4% |
| Zep | gpt-4o-mini | 98.2% |

**LongMemEval Benchmark**:
| System | Model | Accuracy | Avg Context Tokens | Latency Reduction |
|--------|-------|----------|--------------------|--------------------|
| Zep | gpt-4o | 71.2% | 1.6k | 90% vs. baseline |
| Baseline | gpt-4o | 60.8% | 115k | - |

- **18.5% accuracy improvement** with **90% latency reduction**

### Integration Patterns

**Framework Integration**: LangChain, LangGraph, LlamaIndex
**Storage**: Proprietary Graphiti engine (temporal knowledge graph)
**Deployment**: Zep Cloud (SOC 2) + Community Edition (local/Docker)

### Pros (For Our Codebase)

✅ **Temporal reasoning**: Bi-temporal model handles contradictions, retroactive updates (critical for evolving research knowledge)
✅ **Hierarchical structure**: Episode → Semantic → Community mirrors human memory (episodic/semantic)
✅ **Non-lossy**: Episode subgraph preserves all raw interactions for citation/quotation
✅ **Production benchmarks**: Outperforms MemGPT on DMR, significant LongMemEval gains
✅ **LangGraph integration**: Could bridge to Pydantic Graph workflows
✅ **SOC 2 compliance**: Enterprise-ready for regulated domains (bioinformatics/healthcare)

### Cons

❌ **Proprietary engine**: Graphiti is Zep-specific, unlike Neo4j (open standard)
❌ **LangChain-centric**: Integration patterns assume LangChain/LangGraph stack
❌ **Pydantic AI compatibility unclear**: No documented patterns
❌ **Multi-agent gaps**: Designed for single-agent contexts, not hierarchical orchestration
❌ **Complexity**: 3-tier + bi-temporal model adds operational overhead
❌ **OSS maturity**: Community Edition may lack advanced features (vs. Cloud)

### Suitability for DeepCritical/DeepResearch

**Score: 7.5/10**

Excellent for temporal reasoning and structured memory. Concerns: proprietary engine, LangChain coupling, and multi-agent gaps.

---

## 4. G-Memory: Hierarchical Multi-Agent Memory

**Source**: arXiv 2506.07398 (June 2025), research system (not production SaaS)

**Official GitHub**: https://github.com/bingreeky/GMemory
**arXiv Paper**: https://arxiv.org/abs/2506.07398
**Local Reference**: `/reference_repo/gmemory-research/`

### Architecture

**Three-Tier Hierarchical Graphs** (inspired by organizational memory theory):
1. **Insight Graphs**: High-level, generalizable knowledge across multiple trials
   - Cross-trial learning: patterns that apply to multiple tasks
2. **Query Graphs**: Intermediate query-level information
   - Task-specific context for current queries
3. **Interaction Graphs**: Fine-grained, condensed collaboration trajectories
   - Agent-to-agent communication patterns and outcomes

**Bi-Directional Memory Traversal**:
- **Top-down**: High-level insights → query context → specific interactions
- **Bottom-up**: Specific interactions → query patterns → generalizable insights

**Multi-Agent Coordination**:
- **Cross-trial knowledge**: Leverage insights from previous multi-agent collaborations
- **Agent-specific customization**: Each agent has personalized memory slices
- **Collaboration trajectory encoding**: Compactly represents inter-agent interactions

### Performance Benchmarks

**5 Benchmarks** (embodied action + knowledge QA):
- **Success rates**: Up to **20.89% improvement** in embodied action tasks
- **Accuracy**: Up to **10.12% improvement** in knowledge QA tasks
- **3 LLM backbones tested**: GPT-4, Claude, LLaMA variants
- **3 MAS frameworks**: AutoGPT, MetaGPT, CrewAI (no framework modifications needed)

### Integration Patterns

**Framework Compatibility**: Drop-in replacement for memory modules in existing MAS frameworks
**Storage**: Research prototype (implementation details not specified)
**Deployment**: Not production-ready (academic research system)

### Pros (For Our Codebase)

✅ **Multi-agent native**: ONLY system explicitly designed for multi-agent coordination
✅ **Hierarchical orchestration**: Insight/Query/Interaction tiers map to Primary/REACT/Specialized agent levels
✅ **Cross-trial learning**: Agents leverage past multi-agent collaboration experiences
✅ **Agent profiles**: Customized memory per agent (BioinformaticsAgent, PRIMEAgent, etc.)
✅ **Framework-agnostic**: Proven compatibility with AutoGPT, MetaGPT, CrewAI (similar complexity to our stack)
✅ **Collaboration encoding**: Critical for nested orchestration (PRIME → Executor → Tool → MCP server workflows)

### Cons

❌ **Research system**: Not production-ready, no SaaS or mature OSS release
❌ **Implementation gaps**: Paper lacks storage backend, deployment, and API details
❌ **Unproven scalability**: Benchmarks use small-scale tasks (not ~28 bioinformatics modules + the broader tool stack)
❌ **Maintenance risk**: Academic project may not receive long-term support
❌ **Integration complexity**: Would require custom implementation to integrate with Pydantic Graph

### Suitability for DeepCritical/DeepResearch

**Score: 6.5/10**

BEST architectural fit for multi-agent coordination, but lacks production maturity. Could inspire custom implementation.

---

## 5. Pattern Reference: O-Mem (Paper-Only, No Code)

**Source**: arXiv 2511.13593 (Nov 2025) | **Code**: ❌ No public repository | **arXiv**: https://arxiv.org/abs/2511.13593

**Valuable Patterns** (for custom implementation):
1. **Persona Memory**: Agent-specific attribute profiles (maps to Mario's agent profile concept)
   - Long-term attributes (Pa) + significant events (Pf)
   - Self-evolving via LLM-augmented clustering (deduplication)
2. **Working Memory**: Topic-indexed interactions (useful for domain-specific agents like BioinformaticsAgent)
   - Maps conversation subjects → related interactions
   - Enables topical continuity across sessions
3. **Episodic Memory**: Keyword-based retrieval with distinctiveness filtering (1/dfw) → prioritizes rare/salient keywords
4. **Parallel Retrieval**: Query all 3 components simultaneously, concatenate results

**Key Claim**: 94% token reduction vs. comparable systems (unverified without code)

**Why Pattern-Only**: No implementation available; would require full custom build to test claims.

---

## 6. Pattern Reference: H-MEM (Paper-Only, No Code)

**Source**: arXiv 2507.22925 (July 2025) | **Code**: ❌ No public repository | **arXiv**: https://arxiv.org/abs/2507.22925

**Valuable Patterns** (for custom implementation):
1. **4-Layer Hierarchy**: Domain → Category → Trace → Episode (maps to flow-based routing)
   - **Domain Layer**: PRIME, Bioinformatics, DeepSearch flows
   - **Category Layer**: Alignment, Variant Calling, Quantification tool types
   - **Trace Layer**: Keyword summaries of interactions
   - **Episode Layer**: Full contextual memory + timestamps
2. **Index-Based Routing**: Layer-by-layer traversal (not exhaustive search)
   - Complexity reduction: O(a·10⁶·D) → O((a+k·300)·D)
3. **User Feedback Regulation**: Approved memories strengthened, refuted memories reduced

**Key Claims** (unverified without code):
- **<100ms retrieval latency** (vs. Mem0's 710ms)
- **+21.25 F1 points** for multi-hop reasoning (LoCoMo dataset)
- Effective across 1.5B–7B parameter models

**Why Pattern-Only**: No implementation available; claims unverifiable; would require full custom build.

---

## 7. MemOS: Task-Concept-Fact Memory OS

**Source**: PDF release July 2025 (statics.memtensor.com.cn), research system

**Official GitHub (3 variants)**:
- **MemTensor**: https://github.com/MemTensor/MemOS (Open-source framework for LLMs)
- **BAI-LAB**: https://github.com/BAI-LAB/MemoryOS (EMNLP 2025 Oral, arXiv 2507.03724)
- **AGIResearch**: https://github.com/agiresearch/MemOS (Memory layer for LLM agents)

**arXiv Paper**: https://arxiv.org/abs/2507.03724 (MemOS: A Memory OS for AI System)
**Local References**: `/reference_repo/memos-memtensor/`, `/reference_repo/memos-bailab/`, `/reference_repo/memos-agiresearch/`

### Architecture

**Hierarchical Graph Structure**:
- **Task-Concept-Fact Paths**: Organizes memory as directed graph
  - Tasks → Concepts → Facts (hierarchical traversal)
- **Three-Layer Architecture**:
  1. **Interface Layer**: MemReader, Memory API
  2. **Operation Layer**: MemOperator, MemScheduler, MemLifecycle
  3. **Infrastructure Layer**: MemGovernance, MemVault, MemStore

**Layered Memory Hierarchy** (OS-inspired):
- **Working Memory**: Active context
- **Long-Term Storage**: Persistent knowledge
- **Cold Archives**: Infrequently accessed data
- Governed by: recency, access frequency, importance

**Scheduling & Multi-Agent Support**:
- MemScheduler: Task-oriented memory management
- Cross-platform memory migration
- Multi-turn dialogue + continuous knowledge evolution
- Personalization + multi-role modeling

### Performance Benchmarks

**Not Reported**: PDF appears corrupted (details unavailable)

### Integration Patterns

**Modular Design**: Installable system supporting explicit memory operations + systematic governance
**Storage**: Custom (MemVault, MemStore)
**Deployment**: Research prototype (implementation details incomplete)

### Pros (For Our Codebase)

✅ **Task-oriented**: Task-Concept-Fact paths map to workflow nodes (Plan → Execute → Analyze)
✅ **Scheduling**: MemScheduler could optimize memory access for tool-heavy workflows
✅ **Multi-role modeling**: Could support agent profiles (BioinformaticsAgent, PRIMEAgent, etc.)
✅ **Memory governance**: Lifecycle management (recency, frequency, importance) reduces clutter

### Cons

❌ **Research system**: Not production-ready, incomplete documentation
❌ **Corrupted PDF**: Full architectural details unavailable
❌ **No benchmarks**: Performance unclear
❌ **Custom infrastructure**: MemVault/MemStore not compatible with Neo4j, Qdrant, etc.
❌ **Maintenance risk**: Academic project may not receive long-term support

### Suitability for DeepCritical/DeepResearch

**Score: 5/10**

Interesting task-oriented design, but immature and poorly documented. Custom infrastructure is a red flag.

---

## 8. LangGraph Memory (LangMem)

**Source**: LangChain/LangGraph ecosystem, template-based starting point (not standalone system)

**Official GitHub**: https://github.com/langchain-ai/langmem
**Documentation**: https://langchain-ai.github.io/langmem/
**Organization**: https://github.com/langchain-ai
**Local Reference**: `/reference_repo/langmem-official/`

### Architecture

**Three Memory Types**:
1. **Episodic Memory**: Interaction-specific context (short-term)
2. **Semantic Memory**: Generalized knowledge (long-term)
3. **Procedural Memory**: Skill/task execution patterns

**Integration Pattern**:
- LangGraph Memory Service: **example template** for custom solutions
- Seamless integration if already using LangGraph
- Flexible abstractions for diverse use cases

### Performance Benchmarks

**Not Reported**: Template-based approach (no standalone benchmarks)

### Integration Patterns

**Framework Integration**: Native LangGraph support
**Storage**: Pluggable (user-defined backends)
**Deployment**: Self-managed (no SaaS offering)

### Pros (For Our Codebase)

✅ **Framework-native**: If migrating to LangGraph, memory is built-in
✅ **Flexible abstractions**: Customize to fit Pydantic Graph patterns
✅ **Template-based**: Learn architectural patterns without vendor lock-in

### Cons

❌ **Not a standalone system**: Example template, not production-ready framework
❌ **Requires LangGraph**: Our codebase uses Pydantic Graph (different abstraction)
❌ **DIY implementation**: No out-of-box solution (vs. Mem0/Letta/Zep)
❌ **No benchmarks**: Performance unclear
❌ **Migration cost**: Would require rewriting workflows from Pydantic Graph → LangGraph

### Suitability for DeepCritical/DeepResearch

**Score: 4/10**

Low priority unless migrating to LangGraph (not recommended given Pydantic Graph investment).

---

## Comparison Matrix

| System | Multi-Agent | Prod Maturity | Latency | Accuracy | Pydantic AI Fit | Neo4j Compat | Hybrid Storage |
|--------|-------------|---------------|---------|----------|-----------------|--------------|----------------|
| **Mem0** | ❌ Single | ✅ SaaS + OSS | 710ms (p50) | 66.9% (LoCoMo) | ⚠️ Unclear | ✅ Yes | ✅ Graph+Vec+KV |
| **Letta** | ❌ Single | ✅ Cloud + OSS | - | 74.0% (LoCoMo) | ✅ Agent-native | ⚠️ Custom adapter | ❌ Filesystem |
| **Zep** | ❌ Single | ✅ Cloud + OSS | <100ms (est) | 94.8% (DMR) | ⚠️ Unclear | ❌ Proprietary | ❌ Graph-only |
| **G-Memory** | ✅ Native | ❌ Research | - | +20.89% (action) | ⚠️ Custom impl | ⚠️ Unclear | ⚠️ Hierarchical graphs |
| **O-Mem** | ❌ Single | ❌ Paper-only | - | - (94% token↓, per paper) | ⚠️ Custom impl | ⚠️ Unclear | ❌ No graph |
| **H-MEM** | ❌ Single | ❌ Paper-only | <100ms (per paper) | +14.98 F1 (per paper) | ⚠️ Custom impl | ⚠️ Unclear | ❌ Hierarchical only |
| **MemOS** | ⚠️ Multi-role | ❌ Research | - | - | ⚠️ Custom impl | ❌ Custom infra | ⚠️ Task-Concept-Fact |
| **LangMem** | ❌ Single | ⚠️ Template | - | - | ❌ LangGraph req | ⚠️ Pluggable | ⚠️ User-defined |

**Legend**: ✅ Strong | ⚠️ Partial/Unclear | ❌ Weak/Missing

---

## Key Architectural Patterns Identified

### 1. Hybrid Storage (Graph + Vector + Key-Value)

**Best Example**: Mem0
**Why It Matters**: Different retrieval types for different queries
- **Semantic search**: Vector embeddings (e.g., "find similar protein folding papers")
- **Relational queries**: Graph traversal (e.g., "what tools did BioinformaticsAgent use for RNA-Seq?")
- **Exact lookups**: Key-value store (e.g., "retrieve user preference for E-value threshold")

**Fit for DeepCritical/DeepResearch**:
- **Neo4j** already in baseline → graph store ready
- **Chroma/Qdrant** for vector embeddings
- **Redis/Postgres** for KV store (or Neo4j properties)

---

### 2. Hierarchical Multi-Agent Memory

**Best Example**: G-Memory
**Why It Matters**: Different abstraction levels for different agent roles
- **Insight Graphs**: Cross-trial patterns (Primary Orchestrator learns "PRIME flows usually need protein structure tools")
- **Query Graphs**: Task-specific context (REACT Orchestrator tracks current reasoning loop state)
- **Interaction Graphs**: Fine-grained tool call sequences (Executor logs MCP server responses)

**Fit for DeepCritical/DeepResearch**:
- Directly maps to **3-level orchestration** (Primary → REACT → Specialized agents)
- Enables **cross-trial learning** (e.g., "Last time user asked about RNA-Seq, STAR aligner failed, try HISAT2 first")

---

### 3. Temporal Knowledge Graphs

**Best Example**: Zep
**Why It Matters**: Research knowledge evolves (papers retracted, tools updated, results contradicted)
- **Event time vs. ingestion time**: Track when user learned about something vs. when it actually happened
- **Edge invalidation**: Handle contradictions (e.g., "Paper X claimed Y, but retracted; newer paper Z shows ¬Y")

**Fit for DeepCritical/DeepResearch**:
- **Bioinformatics domain**: Research findings change over time (tool benchmarks, best practices)
- **Tool version tracking**: STAR v2.7.10a behaves differently than v2.7.9a

---

### 4. Agent Profiles / Persona Memory

**Best Examples**: O-Mem, Mario's Ports & Adapters proposal
**Why It Matters**: Different agents need different memory slices
- **BioinformaticsAgent**: Papers, genes, proteins, tool benchmarks
- **PRIMEAgent**: Protein structures, molecular dynamics, tool_history (BLAST, AlphaFold)
- **DeepSearchAgent**: Web sources, citations, research synthesis notes

**Fit for DeepCritical/DeepResearch**:
- Aligns with Mario's vision: "selective memory retrieval based on agent needs"
- Reduces noise: CodeExecutionAgent doesn't need BioinformaticsAgent's protein database

---

### 5. Memory Self-Editing (LLM as Memory Manager)

**Best Example**: Letta (MemGPT)
**Why It Matters**: Agents actively manage what to remember/forget
- **Tool-based editing**: Agents call `memory_add()`, `memory_update()`, `memory_delete()`
- **Context window optimization**: Agents decide what stays in core memory vs. archival

**Fit for DeepCritical/DeepResearch**:
- **Pydantic AI** already uses tool calls → natural fit
- **AgentDependencies** could include memory tools
- **Adaptive re-planning**: When tool fails, agent updates "avoid BLAST for low-complexity sequences" in memory

---

### 6. Hierarchical Efficiency (Index-Based Routing)

**Best Example**: H-MEM (paper-only; no public code as of Nov 19, 2025)
**Why It Matters**: Scalability for large memory volumes (~28 bioinformatics modules × tool histories)
- **Layer-by-layer traversal**: Domain → Category → Trace → Episode (O((a+k·300)·D) vs. O(a·10⁶·D))
- **<100ms latency (per paper)**: Critical for interactive research workflows

**Fit for DeepCritical/DeepResearch**:
- **Flow-based routing**: PRIME, Bioinformatics, DeepSearch → natural hierarchy (Domain layer)
- **Tool categorization**: Alignment, Variant Calling, Quantification → Category layer
- **Episode layer**: Individual tool executions (ExecutionHistory already tracks this in Phase 1!)

---

## Retrieval Strategies Analyzed

### 1. Semantic Search (Vector Similarity)

**Used By**: All systems
**Implementation**: Cosine similarity on embeddings (OpenAI, Anthropic, sentence-transformers)
**Strengths**: Handles synonyms, paraphrasing, conceptual similarity
**Weaknesses**: No relational reasoning, no exact matches

**Fit**: ✅ Already have vector stores (Chroma, Qdrant) in Phase 1 baseline

---

### 2. Hybrid Search (Vector + Full-Text)

**Used By**: Zep, Mem0
**Implementation**: Combine vector similarity + BM25/full-text search
**Strengths**: Captures both semantic and keyword-based matches
**Weaknesses**: Tuning weight balance (semantic vs. keyword)

**Fit**: ✅ Neo4j supports full-text indexes + vector similarity

---

### 3. Graph Traversal (Relational Queries)

**Used By**: Mem0ᵍ, Zep, G-Memory
**Implementation**: Cypher queries (Neo4j), custom graph algorithms
**Strengths**: Multi-hop reasoning, relationship-based retrieval
**Weaknesses**: Slower than vector search for simple queries

**Fit**: ✅ Neo4j already in baseline → graph queries ready

---

### 4. Hierarchical Routing (Index-Based)

**Used By**: H-MEM, MemOS
**Implementation**: Layer-by-layer traversal with self-positional indices
**Strengths**: O((a+k·300)·D) complexity, <100ms latency
**Weaknesses**: Requires hierarchical structure design

**Fit**: ⚠️ Would need custom implementation (flow-based hierarchy)

---

### 5. Parallel Multi-Component Retrieval

**Used By**: O-Mem
**Implementation**: Query Working/Episodic/Persona memories simultaneously, concatenate results
**Strengths**: Fast, leverages different memory types
**Weaknesses**: Potential redundancy/noise in concatenated results

**Fit**: ⚠️ Would need custom implementation

---

## Integration with Pydantic AI / Pydantic Graph

### Challenge: No Direct Documentation

**None of the surveyed systems have explicit Pydantic AI integration guides.** Most focus on:
- **LangChain/LangGraph**: Mem0, Zep, LangMem
- **Standalone frameworks**: Letta (MemGPT)
- **Research prototypes**: G-Memory, O-Mem, H-MEM, MemOS

### Potential Bridging Strategies

#### Strategy 1: Memory as AgentDependencies

**Pattern**: Inject memory client into Pydantic AI agents via `deps_type`

```python
from pydantic_ai import Agent
from mem0 import MemoryClient  # or Letta, Zep client

@dataclass
class AgentDependencies:
    config: dict[str, Any]
    memory: MemoryClient  # <-- Inject memory client
    tools: list[str]
    other_agents: list[str]
    data_sources: list[str]

agent = Agent(
    model="anthropic:claude-sonnet-4-0",
    deps_type=AgentDependencies,
    result_type=ResultType,
    system_prompt="Use memory.search() to retrieve past context..."
)

@agent.tool_plain
def retrieve_memory(ctx: RunContext[AgentDependencies], query: str) -> dict:
    results = ctx.deps.memory.search(query, user_id="bioinformatics_agent")
    return {"memories": results}
```

**Pros**: ✅ Clean dependency injection, type-safe
**Cons**: ⚠️ Memory operations not auto-managed by agent (manual tool calls)

---

#### Strategy 2: Memory as Pydantic Graph State

**Pattern**: Store memory snapshots in `ResearchState` dataclass

```python
@dataclass
class ResearchState:
    question: str
    plan: list[str] | None = field(default_factory=list)
    memory_context: dict[str, Any] = field(default_factory=dict)  # <-- Memory snapshot
    # ... other fields

@dataclass
class Plan(BaseNode[ResearchState]):
    async def run(self, ctx: GraphRunContext[ResearchState]) -> NextNode:
        # Retrieve memory before planning
        memory_client = get_memory_client()  # Hydra config
        context = memory_client.search(ctx.state.question, user_id="primary_orchestrator")
        ctx.state.memory_context = context
        # ... planning logic
        return Execute()
```

**Pros**: ✅ Memory integrated into workflow state, persistent across nodes
**Cons**: ⚠️ State size grows (ResearchState already large), serialization overhead

---

#### Strategy 3: Memory Wrapper for Pydantic AI Models

**Pattern**: Wrap Pydantic AI's model interface to auto-inject memory into prompts

```python
from pydantic_ai.models import Model, KnownModelName

class MemoryAugmentedModel(Model):
    def __init__(self, base_model: KnownModelName, memory_client: MemoryClient):
        self.base_model = base_model
        self.memory_client = memory_client

    async def request(self, messages: list[Message]) -> Response:
        # Retrieve memory based on latest message
        query = messages[-1].content
        memory_context = self.memory_client.search(query)

        # Inject memory into system prompt
        augmented_messages = [
            Message(role="system", content=f"Relevant memory: {memory_context}"),
            *messages
        ]
        return await self.base_model.request(augmented_messages)

# Usage
agent = Agent(
    model=MemoryAugmentedModel("anthropic:claude-sonnet-4-0", memory_client),
    deps_type=AgentDependencies,
    result_type=ResultType,
)
```

**Pros**: ✅ Auto-injection, no manual tool calls, transparent to agent
**Cons**: ⚠️ Complex wrapper implementation, potential token bloat

---

#### Strategy 4: Ports & Adapters (Mario's Proposal)

**Pattern**: Memory provider interface with pluggable backends

```python
from abc import ABC, abstractmethod

class MemoryProvider(ABC):
    @abstractmethod
    async def store(self, agent_id: str, key: str, value: Any, profile: str) -> None:
        pass

    @abstractmethod
    async def retrieve(self, agent_id: str, query: str, profile: str) -> list[Any]:
        pass

class Mem0Provider(MemoryProvider):
    def __init__(self, client: MemoryClient):
        self.client = client

    async def store(self, agent_id: str, key: str, value: Any, profile: str) -> None:
        self.client.add(f"{agent_id}:{profile}:{key}", value)

    async def retrieve(self, agent_id: str, query: str, profile: str) -> list[Any]:
        return self.client.search(query, user_id=f"{agent_id}:{profile}")

# Inject into AgentDependencies
@dataclass
class AgentDependencies:
    config: dict[str, Any]
    memory_provider: MemoryProvider  # <-- Ports & Adapters
    tools: list[str]
```

**Pros**: ✅ Vendor-agnostic, testable (mock provider), aligns with Mario's vision
**Cons**: ⚠️ Requires custom adapter implementation for each backend

---

### Recommendation for Phase 3

**Hybrid Approach**:
1. **Strategy 4 (Ports & Adapters)** for architecture → vendor-agnostic interface
2. **Strategy 1 (AgentDependencies)** for Pydantic AI integration → clean dependency injection
3. **Strategy 2 (Graph State)** for workflow-level memory → persistent context across nodes

**Rationale**: Combines Mario's vision (Ports & Adapters), Pydantic AI's patterns (deps injection), and Pydantic Graph's state management.

---

## Production Readiness Assessment

### Tier 1: Production-Ready (Deploy Today)

1. **Mem0** (SaaS + OSS): Y Combinator backing, extensive benchmarks, multi-framework support
2. **Letta** (Cloud + OSS): Active community, Letta Cloud SaaS, Claude Sonnet 4.5 support
3. **Zep** (Cloud + Community): SOC 2 compliance, LangGraph integration, strong benchmarks

**Recommendation**: **Start with Mem0 OSS** (Neo4j compat, hybrid storage) or **Letta OSS** (agent-native, open-source first).

---

### Tier 2: Research-Informed (Phase 4 Prototypes)

1. **G-Memory**: Multi-agent patterns (code present; research maturity)
2. **H-MEM**: Hierarchical efficiency (paper-only; no public code as of Nov 19, 2025)
3. **O-Mem**: Persona memory (paper-only; no public code as of Nov 19, 2025)

**Recommendation**: **Borrow architectural patterns** (hierarchical graphs, agent profiles, index-based routing) for **custom implementation** in Phase 4.

---

### Tier 3: Not Recommended

1. **MemOS**: Poorly documented, custom infrastructure
2. **LangMem**: Requires LangGraph migration (not worth it)

**Recommendation**: **Avoid** unless specific use case emerges.

---

## Gaps Identified in All Systems

### 1. Multi-Agent Orchestration

**Problem**: All production systems (Mem0, Letta, Zep) designed for **single-agent contexts**.
**Gap**: No built-in support for hierarchical orchestration (Primary → REACT → Specialized agents).
**Solution**: Custom implementation borrowing G-Memory's hierarchical graph patterns.

---

### 2. Tool-Heavy Workflows

**Problem**: Benchmarks focus on **conversational QA** (LoCoMo, DMR, LongMemEval), not tool execution.
**Gap**: No evaluation of memory for workflows with **~28 bioinformatics modules + broader toolchains**.
**Solution**: Phase 4 prototyping with real-world PRIME/Bioinformatics flows.

---

### 3. Pydantic AI Integration

**Problem**: All systems assume **LangChain/LangGraph** or standalone frameworks.
**Gap**: No documented patterns for **Pydantic AI + Pydantic Graph**.
**Solution**: Custom adapters (Strategy 1 + Strategy 4 from Integration section).

---

### 4. Agent Profiles (Selective Memory)

**Problem**: Most systems retrieve **all memory** for every query.
**Gap**: No built-in filtering by **agent role** (BioinformaticsAgent vs. PRIMEAgent).
**Solution**: Implement profile-based filtering (inspired by O-Mem's Persona Memory + Mario's proposal).

---

### 5. Temporal Reasoning (Outside Zep)

**Problem**: Only **Zep** handles temporal contradictions, retroactive updates.
**Gap**: Mem0/Letta don't track **event time vs. ingestion time**.
**Solution**: If temporal reasoning critical, choose Zep or implement custom bi-temporal model.

---

## Recommendations for Phase 3

### Option A: Production-First (Mem0)

**Choose Mem0 OSS if**:
- Need **hybrid storage** (graph + vector + KV) → leverage existing Neo4j
- Want **production maturity** → Y Combinator, SaaS fallback
- Value **token efficiency** → 90% savings critical for tool-heavy workflows

**Phase 3 Work**:
- Design Mem0 → Pydantic AI adapter (Strategy 1 + Strategy 4)
- Implement profile-based filtering (agent_id:profile namespacing)
- Test with BioinformaticsAgent + PRIMEAgent
- Prototype multi-agent memory (inspired by G-Memory)

---

### Option B: Open-Source First (Letta)

**Choose Letta OSS if**:
- Prioritize **open-source community** → no vendor lock-in
- Want **agent-native design** → memory as tool calls
- Need **debugging visibility** → ADE for memory state inspection
- Value **Claude Sonnet 4.5 support** → our primary LLM

**Phase 3 Work**:
- Design Letta → Pydantic AI adapter (Strategy 1 + Strategy 4)
- Implement Neo4j custom backend (replace filesystem)
- Add profile-based filtering
- Prototype multi-agent memory (inspired by G-Memory)

---

### Option C: Best-of-Breed Hybrid (Mem0 + G-Memory Patterns)

**Choose Hybrid if**:
- Want **production backend** (Mem0) + **multi-agent architecture** (G-Memory)
- Willing to invest in **custom implementation**
- Need **hierarchical memory** (Insight/Query/Interaction graphs)

**Phase 3 Work**:
- Use Mem0 as storage layer (graph + vector + KV)
- Implement G-Memory's 3-tier hierarchy on top of Mem0
- Map tiers to orchestration levels (Primary → REACT → Specialized)
- Design Ports & Adapters interface (MemoryProvider)
- Test with full PRIME flow (Primary Orchestrator → PRIMEAgent → BLAST MCP server)

---

### Option D: Research-Driven Custom (H-MEM + Zep Patterns)

**Choose Custom if**:
- Need **best performance** (H-MEM's <100ms latency per paper; no public code)
- Require **temporal reasoning** (Zep's bi-temporal model)
- Have engineering bandwidth for **full custom implementation**

**Phase 3 Work**:
- Design H-MEM-inspired 4-layer hierarchy (Domain/Category/Trace/Episode)
- Map Domain → flows (PRIME, Bioinformatics), Category → tool types (Alignment, Variant Calling)
- Implement Zep-style bi-temporal tracking (event time, ingestion time)
- Use Neo4j as storage backend
- Build custom retrieval engine (index-based routing)

---

## Final Recommendation

**START WITH OPTION A (Mem0) or OPTION B (Letta)** for Phase 3:
- ✅ Production-ready (can deploy to users in Phase 4)
- ✅ Existing OSS implementations (no R&D risk)
- ✅ Neo4j compatible (Mem0) or extensible (Letta custom backend)
- ✅ Token efficiency (critical for the tool-heavy stack)

**INCORPORATE PATTERNS FROM**:
- **G-Memory**: Multi-agent hierarchical memory (Insight/Query/Interaction graphs)
- **H-MEM**: Index-based routing for scalability (pattern reference; no code)
- **O-Mem**: Agent profile-based filtering (Persona Memory; pattern reference)
- **Zep**: Temporal reasoning (if needed for research domain)

**Phase 3 Decision Tree**:
1. **Evaluate Mem0 OSS integration** with single-agent (BioinformaticsAgent) → If smooth, proceed
2. **Evaluate Letta OSS integration** with single-agent (PRIMEAgent) → If Mem0 blocked, pivot to Letta
3. **Prototype multi-agent memory** (G-Memory patterns) with chosen backend
4. **Compare performance** (latency, accuracy, token usage) → Final pick for Phase 4

**Phase 4 Vertical Slices**:
- Slice 1: Minimal viable memory (chosen backend + 1 agent + local storage)
- Slice 2: Profile-based filtering (BioinformaticsAgent vs. PRIMEAgent)
- Slice 3: Multi-agent orchestration (G-Memory 3-tier hierarchy)
- Slice 4: Neo4j production backend (replace local storage)
- Slice 5: Retrieval optimization (H-MEM index-based routing)
- Slice 6: Full rollout (all agents + all flows)

---

## Appendix: Benchmark Datasets

### LoCoMo (Long Context Memory)

**Source**: Mem0 paper (arXiv 2504.19413)
**Size**: 5,000+ QA pairs
**Categories**: Single-hop, temporal, multi-hop, open-domain
**Focus**: Conversational memory retrieval accuracy

**Top Performers**:
1. Letta: 74.0% (GPT-4o mini)
2. Mem0ᵍ: 68.4% (graph-enhanced)
3. Mem0: 66.9% (base)

---

### DMR (Deep Memory Retrieval)

**Source**: MemGPT team (Letta lineage)
**Focus**: Deep conversation history retrieval

**Top Performers**:
1. Zep: 98.2% (gpt-4o-mini), 94.8% (gpt-4-turbo)
2. MemGPT: 93.4% (gpt-4-turbo)

---

### LongMemEval

**Source**: Research benchmark (long-context evaluation)
**Baseline**: 115k tokens (full conversation history)

**Top Performer**:
- Zep: 71.2% accuracy, 1.6k avg tokens (90% latency reduction, 18.5% accuracy gain vs. baseline)

---

### Embodied Action + Knowledge QA (G-Memory)

**Source**: G-Memory paper (arXiv 2506.07398)
**Benchmarks**: 5 datasets (embodied tasks + QA)

**Improvements**:
- Success rates: +20.89% (embodied action)
- Accuracy: +10.12% (knowledge QA)

---

## Appendix: Integration Checklist

For **Phase 3 Implementation Spec**, validate chosen system against:

- [ ] **Pydantic AI Compatibility**: Can inject memory client into AgentDependencies?
- [ ] **Pydantic Graph Integration**: Can store memory snapshots in ResearchState?
- [ ] **Neo4j Backend**: Does system support Neo4j or can we build adapter?
- [ ] **Agent Profiles**: Can filter memory by agent role (agent_id:profile)?
- [ ] **Multi-Agent Coordination**: Can implement hierarchical memory (Insight/Query/Interaction)?
- [ ] **Tool Execution Tracking**: Can store ExecutionHistory items (tool, status, result, error)?
- [ ] **Hydra Configuration**: Can configure memory provider via YAML (provider: mem0/letta/zep)?
- [ ] **MCP Server Workflows**: Can handle ~28 bioinformatics modules + the wider tool stack without token limits?
- [ ] **Retrieval Performance**: <500ms latency for interactive research workflows?
- [ ] **Production Deployment**: SaaS fallback or self-hosted OSS (Docker)?
- [ ] **Testing Strategy**: Can mock memory provider for unit tests?
- [ ] **Rollback Plan**: Can disable memory layer without breaking workflows?

---

## Appendix B: Additional Memory Systems (Local Reference Only)

The following systems were also cloned for reference but not evaluated in detail for Phase 2:

### A-MEM (Agentic Memory)

**GitHub**: https://github.com/agiresearch/A-mem
**Conference**: NeurIPS 2025
**Local Reference**: `/reference_repo/amem-official/`
**Architecture**: Zettelkasten-inspired note-based structure with LLM-generated metadata and dynamic links

### MemEngine

**GitHub**: https://github.com/nuster1128/MemEngine
**arXiv Paper**: https://arxiv.org/abs/2505.02099
**Local Reference**: `/reference_repo/memengine-official/`
**Architecture**: Unified library implementing various memory models for LLM agents

### OpenMemory

**GitHub**: https://github.com/CaviraOSS/OpenMemory
**Local Reference**: `/reference_repo/openmemory-official/`
**Architecture**: Self-hosted, framework-free memory system for any AI application

### Memori

**GitHub**: https://github.com/GibsonAI/Memori
**Local Reference**: `/reference_repo/memori-official/`
**Architecture**: Open-source memory engine for LLMs, AI agents, and multi-agent systems

These systems may be evaluated in future phases if Phase 3 prototyping reveals gaps in the top contenders (Mem0, Letta, Zep, G-Memory).

---

**Status**: ✅ Phase 2 Research Complete
**Next**: Phase 3 - Combined Implementation Spec (marry Phase 1 baseline + Phase 2 research + Mario's Ports & Adapters proposal)

**Related Documentation**:
- `referencerepos.md` - Official GitHub links for all memory systems
- `PHASE_1_BASELINE_FOUNDATION.md` - DeepCritical/DeepResearch codebase baseline
- `META_PLAN.MD` - 4-phase approach overview
- `/reference_repo/README.md` - Local reference copies index
