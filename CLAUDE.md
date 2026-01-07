# CLAUDE.md - Project Guide & State Memo

> [!IMPORTANT]
> **META-RULE**: DO NOT remove or modify the "Operational Meta-Rules" section during automated syncs. This section defines the core collaboration protocol.

## 📜 Operational Meta-Rules (PERSISTENT)
1. **Self-Explanatory Updates**: After each stage, update this file with enough detail that a "fresh" Claude instance can understand the current state, architectural decisions, and logic flow without re-scanning the entire codebase.
2. **Context Compression**: Before executing `/clear`, migrate all "implicit knowledge" (why things were done a certain way) and the current "Skeleton Graph" state into the **Memory & Decision Log**.
3. **Plan Alignment**: All code generation must strictly follow the Phase-based roadmap in `polymorphic-dazzling-meadow.md`.
4. **Sync Integrity**: During any `sync` or automated update, the "Operational Meta-Rules", "Key Concepts", and "Memory & Decision Log" sections must be preserved or appended to, never truncated.

## 🚀 Project Overview
- **Project**: vibecoding (Graph-Driven DFS Code Agent Architecture)
- **Goal**: A next-gen Code Agent using **GraphDB-driven DFS** to handle implicit "Magic Dependencies".
- **Reference**: See `polymorphic-dazzling-meadow.md` for the full Project Implementation Plan.
- **Tech Stack**: Python 3.13+, LangGraph, Claude API, Neo4j, Pydantic.

## Critical Commands
```bash
# Environment
.venv/bin/python

# Install dependencies
uv sync

# Start Neo4j (required for demos)
docker run -d --name neo4j-vibecoding \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:5.20-community

# Test imports
.venv/bin/python -c "from src.agents import *; from src.engine import *; print('OK')"

# Run demos
export ANTHROPIC_API_KEY="your-key"
python examples/demo_magic_link.py    # Phase 2 demo
python examples/demo_dfs_engine.py    # Phase 3 demo
```

## Project Structure
```
src/
├── __init__.py
├── agents/                    # Multi-Agent system (Phase 2)
│   ├── __init__.py
│   ├── base.py               # BaseAgent, AgentConfig, AgentState
│   ├── state.py              # WorkflowState, ProjectSpec, ModuleSpec, FileSpec
│   ├── architect.py          # GlobalArchitectAgent - top-level design
│   ├── magic_link.py         # MagicLinkAgent - identifies magic dependencies
│   ├── magic_detector.py     # MagicDetector - pattern-based detection
│   ├── magic_templates.py    # Pre-built templates for common frameworks
│   ├── codex.py              # Dependency Codex - rules and contracts
│   └── orchestrator.py       # MultiAgentOrchestrator - pipeline coordination
├── engine/                    # DFS Generation Engine (Phase 3)
│   ├── __init__.py
│   ├── context.py            # ContextAggregator - collects generation context
│   ├── walker.py             # DFSWalker - recursive traversal, CycleResolver
│   ├── backtrack.py          # BacktrackingEngine - violation recovery
│   └── dfs_agent.py          # DFSImplementationAgent - code generation
├── graph/                     # Graph Database Layer (Phase 1)
│   ├── __init__.py
│   ├── client.py             # Neo4j async client
│   └── repository.py         # GraphRepository - CRUD + queries
├── models/                    # Data Models (Phase 1)
│   ├── __init__.py
│   ├── graph_nodes.py        # ProjectNode, ModuleNode, FileNode, MagicProtocolNode
│   ├── graph_edges.py        # ImportsEdge, DependsOnEdge, InfluencedByEdge
│   └── magic_protocol_schema.py  # JSON-LD schema for magic protocols
└── audit/                     # Audit System (Phase 4 - TODO)
    └── __init__.py

examples/
├── demo_architect.py          # Phase 1 demo (architecture generation)
├── demo_magic_link.py         # Phase 2 demo (magic dependency detection)
└── demo_dfs_engine.py         # Phase 3 demo (DFS generation)
```

## Implementation Status

### Phase 0: Infrastructure [COMPLETE]
- [x] Project structure initialized
- [x] Dependencies in pyproject.toml
- [x] Neo4j Docker setup

### Phase 1: Graph Data Model [COMPLETE]
- [x] Node types: Project, Module, File, MagicProtocol, Interface
- [x] Edge types: IMPORTS, DEPENDS_ON, INFLUENCED_BY, IMPLEMENTS, CONTAINS
- [x] JSON-LD schema for magic protocols
- [x] GraphRepository with CRUD operations
- [x] Neo4j async client

### Phase 2: Agent Core Framework [COMPLETE]
- [x] BaseAgent with LangGraph integration
- [x] GlobalArchitectAgent - generates project skeleton
- [x] MagicLinkAgent - identifies magic dependencies
- [x] MagicDetector - pattern-based magic detection
- [x] Codex - dependency rules and contracts
- [x] CodexBuilder - builds codex from detected magic
- [x] MultiAgentOrchestrator - coordinates agent pipeline
- [x] Magic templates for FastAPI, SQLAlchemy, Pydantic, Redis, Celery

### Phase 3: DFS Generation Engine [COMPLETE]
- [x] DFSWalker - core recursive generation algorithm
- [x] ContextAggregator - collects ancestor, sibling, magic context
- [x] BacktrackingEngine - handles violations, multi-level recovery
- [x] DFSImplementationAgent - generates code with full context
- [x] CycleResolver - detects and suggests fixes for circular deps

### Phase 4: Audit System [TODO]
- [ ] Codex Loader
- [ ] Rule Engine
- [ ] Code Analyzer (tree-sitter integration)
- [ ] Violation Detector & Reporter
- [ ] Auto-Fix Suggester

### Phase 5: Integration & Optimization [TODO]
- [ ] End-to-end testing
- [ ] Performance benchmarks
- [ ] Context compression optimization
- [ ] Incremental update support

## Key Concepts

### Magic Dependencies
Implicit contracts not visible in import statements:
- **ENV**: Environment variables (e.g., `DATABASE_URL`)
- **MIDDLEWARE**: Request interceptors (e.g., auth middleware → `current_user`)
- **DI**: Dependency injection (e.g., `Depends(get_db)`)
- **HOOK**: Lifecycle hooks (e.g., `@app.on_event("startup")`)
- **ORM**: Relationship magic (e.g., lazy loading)

### DFS Walker Algorithm
```python
def dfs_generate(node_id, context):
    # 1. Get dependencies in topological order
    # 2. Recursively generate all dependencies first
    # 3. Aggregate context (ancestors + magic + siblings)
    # 4. Generate code for current node
    # 5. Validate against codex rules
    # 6. Backtrack if violations detected
```

### Backtracking Levels
| Level | Trigger | Action |
|-------|---------|--------|
| CURRENT_NODE | Type mismatch | Regenerate with fix suggestions |
| PARENT_NODE | Magic violation | Notify parent to adjust |
| MODULE_LEVEL | Circular dependency | Redesign module boundaries |
| GLOBAL | Critical violation | Trigger architect redesign |

## Memory & Decision Log

### 2026-01-07
- Initialized project with Phase 0
- Completed Phase 1: Graph data model with Neo4j
- Completed Phase 2: Multi-agent framework with LangGraph
  - Fixed Pydantic v2 compatibility (added `model_config = ConfigDict(populate_by_name=True)`)
- Completed Phase 3: DFS Generation Engine
  - Fixed Neo4j Cypher syntax (parameters can't be used in path length)
  - Added JSON serialization for complex node properties

### Architecture Decisions
1. **Neo4j for graph storage**: Cypher queries for dependency traversal
2. **LangGraph for agents**: State machine orchestration
3. **Pydantic v2 for models**: Validation + JSON serialization
4. **Async throughout**: Non-blocking I/O for LLM calls and DB

## Next Steps
1. **Phase 4**: Implement audit system with tree-sitter for code analysis
2. **Phase 5**: End-to-end testing with real project generation
3. **Enhancement**: Add LogicalDesignerAgent for file-level design

---
*Reference: root/polymorphic-dazzling-meadow.md*
