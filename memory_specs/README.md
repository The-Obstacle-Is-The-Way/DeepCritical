# Memory System Implementation Specs (Issue #31)

**Status**: ✅ **IRON-CLAD & READY FOR IMPLEMENTATION**
**Date**: 2025-11-20
**Philosophy**: Ship iterative vertical slices. Build, test, learn, iterate.

---

## 📋 Documentation Structure

This directory contains the complete specification for implementing a long-term memory system for DeepCritical's multi-agent bioinformatics platform.

### Phase 1: Baseline Foundation
**File**: `PHASE_1_BASELINE_FOUNDATION.md` (1,542 lines)

**What It Contains**:
- Complete architecture map of DeepCritical codebase
- All memory integration points (ResearchState, AgentDependencies, ExecutionHistory, Nodes)
- Current agent patterns (Pydantic AI framework)
- Current workflow patterns (Pydantic Graph)
- Configuration structure (Hydra)
- Existing storage backends (Neo4j, ChromaDB, Qdrant)

**Why It Exists**: You can't integrate memory without understanding where it needs to hook in.

---

### Phase 2: Memory Research
**File**: `PHASE_2_MEMORY_RESEARCH.md` (1,000 lines)

**What It Contains**:
- Survey of 6 modern memory systems (Mem0, Letta, Zep, G-Memory, O-Mem, H-MEM)
- Architectural patterns (graph+vector+KV, hierarchical, brain-inspired)
- Performance benchmarks (90% token savings, sub-second retrieval)
- Integration patterns with Pydantic AI
- Decision tree: Option A (pure Mem0) vs B (Letta) vs C (hybrid) vs D (custom)

**Why It Exists**: Standing on the shoulders of giants (Y Combinator-backed Mem0, research papers from Nov 2025).

---

### Phase 3: Implementation Spec
**File**: `PHASE_3_IMPLEMENTATION_SPEC.md` (964 lines)

**What It Contains**:
- **Section 0**: Architecture Decision Record (WHY pure Mem0, WHEN to pivot)
- **Section 0.5**: Scope clarification (long-term memory vs session state)
- Core interfaces (MemoryProvider protocol, MemoryItem model)
- Mem0Adapter implementation strategy
- Integration with Phase 1 baseline (AgentDependencies, ExecutionHistory, ResearchState)
- Hydra configuration structure
- Testing strategy (unit, integration, performance)
- Rollback plans (if Mem0 fails → pivot to Letta)
- Success criteria checklist

**Why It Exists**: This is the marriage of Phase 1 (codebase) + Phase 2 (research). Single source of truth for Phase 4 implementation.

---

### Phase 4: Vertical Slice Implementation

**Phase 4 breaks the work into 4 shippable increments:**

#### Phase 4A: Core Interface & Mock Adapter
**File**: `PHASE_4A_CORE_INTERFACE.md` (121 lines)

**Deliverables**:
- `DeepResearch/src/memory/core.py` (MemoryProvider protocol, MemoryItem model)
- `DeepResearch/src/memory/adapters/mock_adapter.py` (MockMemoryAdapter)
- `DeepResearch/src/memory/factory.py` (get_memory_provider factory)
- Unit tests with 100% coverage

**Acceptance Criteria**:
- Zero external dependencies (no Neo4j, no Mem0)
- Type checker passes (`uvx ty check`)
- All tests pass

**Effort**: 1-2 days

---

#### Phase 4B: Mem0 Adapter & Neo4j Config
**File**: `PHASE_4B_MEM0_ADAPTER.md` (205 lines)

**Deliverables**:
- `DeepResearch/src/memory/adapters/mem0_adapter.py` (Mem0Adapter with normalization)
- `DeepResearch/configs/memory/default.yaml` (Hydra config)
- Update `DeepResearch/configs/config.yaml` (add `memory: default` to defaults)
- Integration tests with testcontainers-neo4j

**Key Features**:
- Dual modes: OSS (Neo4j) and Cloud (Mem0 Platform)
- Reuses existing `db.neo4j.*` config (no duplication)
- Normalizes Mem0's variable response formats
- Optional: FileLock pattern for local cache (from `analytics.py:78-97`)

**Acceptance Criteria**:
- `uv add mem0ai neo4j` succeeds
- Integration test with real Neo4j container passes
- No hardcoded credentials

**Effort**: 2-3 days

---

#### Phase 4C: Agent Wiring & Middleware
**File**: `PHASE_4C_AGENT_WIRING.md` (128 lines)

**Deliverables**:
- Modify `DeepResearch/src/datatypes/agents.py` (add `memory` field to AgentDependencies)
- Modify `DeepResearch/src/datatypes/execution.py` (add `memory` field to ExecutionContext)
- Modify `DeepResearch/app.py` (add `memory_context` to ResearchState)
- Create `DeepResearch/src/tools/memory_tools.py` (recall_memory, save_note tools)
- Modify `DeepResearch/src/agents/agent_orchestrator.py` (inject memory_provider via DI)

**Key Design**:
- Double wiring: AgentDependencies (Pydantic AI) + ExecutionContext (Workflow DAG)
- Dependency injection (not singleton) for testability
- Backward compatible (all memory fields `Optional`)

**Acceptance Criteria**:
- Agents can call `recall_memory` tool
- Existing agents run without crashing (memory=None)

**Effort**: 1 day

---

#### Phase 4D: Pilot Execution & Tracing
**File**: `PHASE_4D_PILOT_EXECUTION.md` (155 lines)

**Deliverables**:
- Modify `DeepResearch/src/utils/execution_history.py` (add memory_provider field, hook add_item)
- Modify executor (pass memory from ExecutionContext to ExecutionHistory)
- End-to-end test: BioinformaticsAgent tool calls appear in Neo4j

**Key Pattern**:
- ExecutionHistory ALREADY has persistence (lines 146-179: save_to_file, load_from_file)
- We're AUGMENTING (add memory hook), not replacing
- Optional: FileLock for concurrent access (defer to Phase 5 multi-agent)

**Acceptance Criteria**:
- BioinformaticsAgent runs BLAST → trace stored in Memory
- Days later: "What did I do with P53?" → Memory retrieves trace
- Original ExecutionHistory functionality unchanged

**Effort**: 1-2 days

---

## 🎯 Total Implementation Timeline

**Conservative Estimate**: 6-7 days (1 week)
**Optimistic Estimate**: 4-5 days

**Breakdown**:
- Phase 4A: 1-2 days (Core + Mock)
- Phase 4B: 2-3 days (Mem0 + Neo4j)
- Phase 4C: 1 day (Agent wiring)
- Phase 4D: 1-2 days (Pilot execution)

**Risk**: LOW (all integration points verified, rollback plan in place)

---

## 📚 Supporting Documents

### META_PLAN.MD
The 4-phase approach overview. Start here if new to the project.

### referencerepos.md
Official GitHub repositories for all memory systems evaluated in Phase 2 (Mem0, Letta, Zep, G-Memory, etc.).

---

## ✅ Quality Assurance

All specs have been:
- ✅ **Audited from first principles** (all file paths verified against actual codebase)
- ✅ **Cross-validated** (no hallucinations, every claim backed by grep/read)
- ✅ **Pattern-verified** (FileLock pattern from `analytics.py`, ExecutionHistory persistence validated)
- ✅ **Concrete** (exact file paths, method signatures, field names, not vague "implement the thing")
- ✅ **Testable** (TDD strategy with specific test cases)
- ✅ **Maintainable** (backward compatible, dependency injection, SOLID principles)

---

## 🚀 Getting Started

1. **Read Phase 1** to understand the codebase
2. **Skim Phase 2** to understand why we chose Mem0
3. **Study Phase 3** for the complete architecture
4. **Implement Phase 4A** first (Core Interface + Mock)
5. **Test**, **validate**, **ship**
6. **Repeat** for 4B, 4C, 4D

**Philosophy**: Ship working increments, not perfect plans.

---

## 🔍 Scope Clarification

**What This Memory System IS**:
- ✅ Long-term, cross-session memory (survives restarts, days/weeks/months)
- ✅ Semantic search ("What did I do with P53 last month?")
- ✅ Agent-specific namespacing (BioinformaticsAgent doesn't see PRIMEAgent's memories)
- ✅ Execution trace persistence (tool calls, results, errors)

**What This IS NOT**:
- ❌ Session-scoped state (that's `DeepAgentState` - todos, files)
- ❌ In-memory conversation buffer (that's middleware)
- ❌ RAG workflow nodes (that's vector store integration)

**Relationship to Existing Systems**:
- **DeepAgentState**: COEXISTS (different scopes, no conflict)
- **ExecutionHistory**: WE AUGMENT (add memory persistence hook)
- **Middleware**: INDEPENDENT (different layer)
- **Vector Stores**: WE CONSUME (via Mem0)

---

## 📞 Questions?

Refer to the specific Phase doc for details:
- **Codebase questions**: Phase 1
- **Architecture questions**: Phase 2 or Phase 3
- **Implementation questions**: Phase 4A/B/C/D

**All specs are iron-clad and ready for implementation.** 🔩

Let's ship this! 🚀
