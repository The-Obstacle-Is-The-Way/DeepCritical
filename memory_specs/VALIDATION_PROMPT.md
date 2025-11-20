# Validation Prompt for Phase 4 Implementation Specs

**Context**: You are reviewing implementation specifications for a long-term memory system integration into a production bioinformatics platform (DeepCritical/DeepResearch). The codebase uses Pydantic AI for agents, Pydantic Graph for workflows, and Hydra for configuration.

**Your Mission**: Validate that Phase 4A-D specs are COMPLETE, ACCURATE, and IMPLEMENTABLE by any engineer or AI agent without ambiguity.

---

## Background Reading (Do This First)

1. **Read PHASE_1_BASELINE_FOUNDATION.md** to understand:
   - Existing integration points (AgentDependencies, ExecutionHistory, ResearchState)
   - Current architecture (Pydantic AI agents, Pydantic Graph workflows)
   - Existing storage (Neo4j, ChromaDB, Qdrant)

2. **Read PHASE_2_MEMORY_RESEARCH.md** to understand:
   - Why we chose Mem0 over alternatives
   - Architecture patterns (graph+vector+KV)
   - Decision tree (Option A: pure Mem0)

3. **Read PHASE_3_IMPLEMENTATION_SPEC.md** to understand:
   - Overall architecture (MemoryProvider protocol, Mem0Adapter)
   - Integration strategy (AgentDependencies, ExecutionHistory augmentation)
   - Scope clarification (long-term memory vs session state)

---

## Your Task: Deep Review of Phase 4A-D

For each Phase 4 document (4A, 4B, 4C, 4D), evaluate:

### ✅ Completeness Checklist

**1. Implementation Details**:
- [ ] Are ALL method implementations described with sufficient detail?
- [ ] Are edge cases handled (None values, empty lists, errors)?
- [ ] Are async patterns clear (await, asyncio.create_task)?
- [ ] Are type hints complete and accurate?

**2. Code Examples**:
- [ ] Does the spec include COMPLETE code examples (not just signatures)?
- [ ] Can an AI agent copy-paste and adapt the examples?
- [ ] Are examples consistent with existing codebase patterns?
- [ ] Do examples show error handling?

**3. Integration Points**:
- [ ] Are ALL file paths verified against Phase 1 baseline?
- [ ] Are ALL import statements listed?
- [ ] Are ALL dependencies (Pydantic, Mem0, Neo4j) explicit?
- [ ] Are configuration values mapped to existing Hydra structure?

**4. Test Strategy**:
- [ ] Are test cases SPECIFIC enough to implement?
- [ ] Are test assertions clear (not vague "verify it works")?
- [ ] Are fixtures described (test data, mock configs)?
- [ ] Are integration test requirements clear (testcontainers, Neo4j)?

**5. Acceptance Criteria**:
- [ ] Are acceptance criteria MEASURABLE?
- [ ] Can you objectively verify each criterion?
- [ ] Are success metrics clear (100% coverage, 0 type errors)?

---

## Critical Questions to Answer

### For Phase 4A (Core Interface + Mock):
1. **MockMemoryAdapter Implementation**:
   - How does it store memories? (in-memory list, dict?)
   - How does it implement "user_id:agent_id" namespacing?
   - How does substring search work?
   - How does metadata filtering work?
   - Show COMPLETE implementation (not just description)

2. **MemoryProvider Protocol**:
   - Is the protocol definition complete?
   - Are all method signatures correct?
   - Are return types explicit?
   - Does it match runtime_checkable from typing?

3. **Factory Pattern**:
   - How does `get_memory_provider` work?
   - How does it read config?
   - What does it return for "mock" vs "mem0"?
   - Show COMPLETE implementation

### For Phase 4B (Mem0 Adapter):
1. **Mem0Adapter Implementation**:
   - How does it initialize Mem0 client?
   - How does it construct mem0_config from Hydra config?
   - How does it normalize variable response formats?
   - How does `add_trace` serialize trace_data?
   - Show COMPLETE implementation (50+ lines)

2. **Configuration Mapping**:
   - Exactly how does `${db.neo4j.uri}` get interpolated?
   - What if db.neo4j config doesn't exist?
   - How do we validate config at startup?
   - Show COMPLETE config example

3. **Integration Tests**:
   - How do we spin up Neo4j container?
   - How do we verify data in Neo4j?
   - What Cypher queries verify memory storage?
   - Show COMPLETE test implementation

### For Phase 4C (Agent Wiring):
1. **Dependency Injection**:
   - Where exactly does AgentOrchestrator get memory_provider?
   - How does it flow from app.py → orchestrator → AgentDependencies?
   - Show COMPLETE wiring code (app.py changes)

2. **Memory Tools**:
   - How do tools access memory from context?
   - How do tools handle memory=None?
   - Show COMPLETE tool implementation

### For Phase 4D (Pilot Execution):
1. **ExecutionHistory Augmentation**:
   - How does add_item trigger memory write?
   - Is it fire-and-forget or await?
   - How does it serialize ExecutionItem?
   - Show COMPLETE add_item implementation

2. **Executor Wiring**:
   - Which executor do we modify first?
   - How does memory flow from ExecutionContext → ExecutionHistory?
   - Show COMPLETE executor changes

---

## Output Format

For each Phase 4 document, provide:

### 1. Gaps Found
```markdown
**Phase 4A Gaps**:
- [ ] MockMemoryAdapter: Missing complete implementation of `search()` method
- [ ] Factory: Not clear how config is validated
- [ ] Tests: Missing fixture setup code
...
```

### 2. Recommended Additions
```markdown
**Phase 4A Additions**:

Add to Section 2.B (Mock Implementation):

\`\`\`python
# Complete MockMemoryAdapter implementation
class MockMemoryAdapter:
    def __init__(self):
        self._memories: list[dict] = []
        self._id_counter = 0

    async def add(self, content: str, user_id: str, agent_id: str,
                  metadata: dict | None = None) -> str:
        memory_id = f"mem_{self._id_counter}"
        self._id_counter += 1

        memory = {
            "id": memory_id,
            "content": content,
            "user_id": user_id,
            "agent_id": agent_id,
            "metadata": metadata or {},
            "created_at": datetime.now(timezone.utc),
            "namespace": f"{user_id}:{agent_id}"
        }
        self._memories.append(memory)
        return memory_id

    async def search(self, query: str, user_id: str, agent_id: str,
                    limit: int = 5, filters: dict | None = None) -> list[MemoryItem]:
        namespace = f"{user_id}:{agent_id}"

        # Filter by namespace
        candidates = [m for m in self._memories if m["namespace"] == namespace]

        # Apply metadata filters
        if filters:
            for key, value in filters.items():
                candidates = [m for m in candidates if m["metadata"].get(key) == value]

        # Substring search
        results = [m for m in candidates if query.lower() in m["content"].lower()]

        # Convert to MemoryItem
        return [
            MemoryItem(
                id=m["id"],
                content=m["content"],
                score=1.0,  # Mock score
                metadata=m["metadata"],
                created_at=m["created_at"],
                agent_id=m["agent_id"],
                user_id=m["user_id"]
            )
            for m in results[:limit]
        ]

    # ... rest of methods ...
\`\`\`
```

### 3. Ambiguities Needing Clarification
```markdown
**Phase 4B Ambiguities**:
- How does Mem0 OSS handle graph writes? (Neo4j cypher? Mem0 abstraction?)
- What if Neo4j connection fails? (Graceful degradation? Error?)
- How do we test without real Mem0 API key?
```

### 4. Verification Checklist
```markdown
**Phase 4A Verification**:
- [x] All file paths exist in Phase 1 baseline
- [x] All imports are valid
- [ ] Complete code examples provided (MISSING)
- [ ] Test fixtures described (MISSING)
- [x] Acceptance criteria measurable
```

---

## Success Criteria for Your Review

Your review is successful if:
1. ✅ You identify ALL missing implementation details
2. ✅ You provide COMPLETE code examples for gaps
3. ✅ You verify ALL integration points against Phase 1
4. ✅ You flag ALL ambiguities needing clarification
5. ✅ An AI agent could implement Phase 4A-D ONLY from the revised specs

---

## Important Constraints

**DO NOT**:
- ❌ Add unnecessary complexity (keep thin wrappers thin)
- ❌ Add features not in Phase 3 architecture
- ❌ Change the chosen architecture (pure Mem0, Ports & Adapters)
- ❌ Add external dependencies beyond Mem0, Neo4j, Pydantic

**DO**:
- ✅ Fill gaps with COMPLETE, WORKING code examples
- ✅ Verify against existing codebase patterns (Phase 1)
- ✅ Keep examples consistent with Pydantic AI / Pydantic Graph patterns
- ✅ Add edge case handling (None, empty, errors)
- ✅ Make specs copy-paste implementable

---

## Example of GOOD vs BAD Spec

**BAD** (current Phase 4A):
```markdown
### B. Mock Implementation
**File**: `DeepResearch/src/memory/adapters/mock_adapter.py`
**Key Features**:
- Simulates "user_id:agent_id" namespacing.
- Basic substring matching for `search()`.
```

**GOOD** (what we need):
```markdown
### B. Mock Implementation
**File**: `DeepResearch/src/memory/adapters/mock_adapter.py`

**Complete Implementation**:
\`\`\`python
from typing import Optional
from datetime import datetime, timezone
from ..core import MemoryProvider, MemoryItem

class MockMemoryAdapter:
    """In-memory implementation of MemoryProvider for testing."""

    def __init__(self):
        self._memories: list[dict] = []
        self._id_counter = 0

    async def add(self, content: str, user_id: str, agent_id: str,
                  metadata: Optional[dict] = None) -> str:
        """Add memory to in-memory store."""
        memory_id = f"mem_{self._id_counter}"
        self._id_counter += 1

        memory = {
            "id": memory_id,
            "content": content,
            "user_id": user_id,
            "agent_id": agent_id,
            "metadata": metadata or {},
            "created_at": datetime.now(timezone.utc),
            "namespace": f"{user_id}:{agent_id}"
        }
        self._memories.append(memory)
        return memory_id

    async def search(self, query: str, user_id: str, agent_id: str,
                    limit: int = 5, filters: Optional[dict] = None) -> list[MemoryItem]:
        """Search memories with substring matching."""
        namespace = f"{user_id}:{agent_id}"

        # 1. Filter by namespace (agent isolation)
        candidates = [m for m in self._memories if m["namespace"] == namespace]

        # 2. Apply metadata filters
        if filters:
            for key, value in filters.items():
                candidates = [m for m in candidates if m["metadata"].get(key) == value]

        # 3. Substring search (case-insensitive)
        results = [m for m in candidates if query.lower() in m["content"].lower()]

        # 4. Convert to MemoryItem
        return [
            MemoryItem(
                id=m["id"],
                content=m["content"],
                score=1.0,  # Mock: all results have score 1.0
                metadata=m["metadata"],
                created_at=m["created_at"],
                agent_id=m["agent_id"],
                user_id=m["user_id"]
            )
            for m in results[:limit]
        ]

    # ... (show rest of methods: get_all, delete, reset, add_trace)
\`\`\`

**Why This Works**:
- Complete, runnable code (not just description)
- Shows namespace isolation logic
- Shows metadata filtering logic
- Shows substring search logic
- Shows MemoryItem construction
- Has inline comments explaining logic
```

---

## Your Deliverable

**File**: `PHASE_4_VALIDATION_REPORT.md`

**Structure**:
```markdown
# Phase 4 Validation Report

## Executive Summary
- Total gaps found: X
- Critical gaps: Y (block implementation)
- Minor gaps: Z (clarifications)
- Recommendation: READY / NEEDS REVISION

## Phase 4A: Core Interface + Mock
### Gaps Found
...
### Recommended Additions
...
### Ambiguities
...

## Phase 4B: Mem0 Adapter
### Gaps Found
...
### Recommended Additions
...
### Ambiguities
...

## Phase 4C: Agent Wiring
### Gaps Found
...
### Recommended Additions
...
### Ambiguities
...

## Phase 4D: Pilot Execution
### Gaps Found
...
### Recommended Additions
...
### Ambiguities
...

## Overall Assessment
- Implementation readiness: X%
- Confidence level: HIGH / MEDIUM / LOW
- Estimated revision time: X hours
```

---

**Start with Phase 4A. Be thorough. Be specific. Show COMPLETE code.**
