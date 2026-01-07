"""DFS Generation Engine for the Graph-Driven Code Agent Architecture.

This module implements Phase 3 of the plan:
- DFS Walker: Core recursive generation algorithm
- Context Aggregator: Collects and merges context from the graph
- Backtracking Engine: Handles violations and triggers recovery
- DFS Implementation Agent: The agent that generates actual code
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
]
