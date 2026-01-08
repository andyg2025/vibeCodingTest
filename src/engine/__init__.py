"""DFS Generation Engine for the Graph-Driven Code Agent Architecture.

This module implements Phase 3 and Phase 5 components:
- DFS Walker: Core recursive generation algorithm
- Context Aggregator: Collects and merges context from the graph
- Backtracking Engine: Handles violations and triggers recovery
- DFS Implementation Agent: The agent that generates actual code
- Context Window Manager: Intelligent context compression (Phase 5)
- Incremental Update Pipeline: Minimal regeneration (Phase 5)
"""

from .context import (
    AggregatedContext,
    ContextAggregator,
    ContextItem,
    ContextPriority,
)
from .walker import (
    CycleResolver,
    DFSState,
    DFSWalker,
    GenerationResult,
    GeneratorFn,
    NodeStatus,
)
from .backtrack import (
    BacktrackAction,
    BacktrackingEngine,
    BacktrackLevel,
    BacktrackReason,
    BacktrackResult,
)
from .dfs_agent import DFSImplementationAgent
from .context_manager import (
    ContextWindowManager,
    ContextBuilder,
    CompressedContext,
    CompressionStrategy,
    ContextItem as ManagedContextItem,
)
from .incremental import (
    IncrementalUpdatePipeline,
    ChangeDetector,
    ImpactAnalyzer,
    SelectiveRegenerator,
    LazyMagicValidator,
    FileChange,
    ChangeType,
    ImpactAnalysis,
    RegenerationPlan,
)

__all__ = [
    # Context
    "AggregatedContext",
    "ContextAggregator",
    "ContextItem",
    "ContextPriority",
    # Walker
    "CycleResolver",
    "DFSState",
    "DFSWalker",
    "GenerationResult",
    "GeneratorFn",
    "NodeStatus",
    # Backtracking
    "BacktrackAction",
    "BacktrackingEngine",
    "BacktrackLevel",
    "BacktrackReason",
    "BacktrackResult",
    # Agent
    "DFSImplementationAgent",
    # Context Window Manager (Phase 5)
    "ContextWindowManager",
    "ContextBuilder",
    "CompressedContext",
    "CompressionStrategy",
    "ManagedContextItem",
    # Incremental Updates (Phase 5)
    "IncrementalUpdatePipeline",
    "ChangeDetector",
    "ImpactAnalyzer",
    "SelectiveRegenerator",
    "LazyMagicValidator",
    "FileChange",
    "ChangeType",
    "ImpactAnalysis",
    "RegenerationPlan",
]
