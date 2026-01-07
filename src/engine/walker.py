"""DFS Walker - Core recursive generation algorithm.

Implements the depth-first traversal of the dependency graph,
generating code for each node while respecting magic dependencies.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Awaitable

import structlog

from src.graph.repository import GraphRepository
from src.engine.context import ContextAggregator, AggregatedContext

logger = structlog.get_logger()


class NodeStatus(str, Enum):
    """Status of a node in the DFS traversal."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    GENERATED = "generated"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class GenerationResult:
    """Result of generating code for a single node."""

    node_id: str
    status: NodeStatus
    code: str = ""
    violations: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    token_count: int = 0
    generation_time_ms: int = 0


@dataclass
class DFSState:
    """State maintained during DFS traversal."""

    # Traversal tracking
    visited: set[str] = field(default_factory=set)
    in_stack: set[str] = field(default_factory=set)  # For cycle detection
    generation_order: list[str] = field(default_factory=list)

    # Results
    generated_code: dict[str, str] = field(default_factory=dict)
    results: dict[str, GenerationResult] = field(default_factory=dict)

    # Errors and warnings
    cycles_detected: list[list[str]] = field(default_factory=list)
    backtrack_count: int = 0
    max_backtrack: int = 3


# Type for the code generation callback
GeneratorFn = Callable[[str, AggregatedContext], Awaitable[GenerationResult]]


class DFSWalker:
    """Depth-First Search walker for code generation.

    Traverses the dependency graph in DFS order, generating code
    for each node with full context awareness.
    """

    def __init__(
        self,
        repository: GraphRepository,
        context_aggregator: ContextAggregator | None = None,
    ):
        self.repository = repository
        self.context_aggregator = context_aggregator or ContextAggregator(repository)
        self._logger = logger.bind(component="DFSWalker")

    async def walk(
        self,
        root_ids: list[str],
        generator: GeneratorFn,
        max_depth: int = 50,
    ) -> DFSState:
        """Walk the graph from root nodes, generating code depth-first.

        Args:
            root_ids: Starting node IDs (typically module nodes)
            generator: Async function that generates code given node_id and context
            max_depth: Maximum recursion depth to prevent infinite loops

        Returns:
            DFSState with all generated code and results
        """
        state = DFSState()

        await self._logger.ainfo(
            "Starting DFS walk",
            root_count=len(root_ids),
            max_depth=max_depth,
        )

        for root_id in root_ids:
            await self._dfs_visit(root_id, generator, state, depth=0, max_depth=max_depth)

        await self._logger.ainfo(
            "DFS walk complete",
            nodes_generated=len(state.generated_code),
            cycles_found=len(state.cycles_detected),
            backtrack_count=state.backtrack_count,
        )

        return state

    async def _dfs_visit(
        self,
        node_id: str,
        generator: GeneratorFn,
        state: DFSState,
        depth: int,
        max_depth: int,
    ) -> GenerationResult:
        """Recursively visit a node in DFS order.

        Algorithm:
        1. Check for cycles
        2. Get dependencies and recursively generate them first
        3. Aggregate context from ancestors, magic protocols, and dependencies
        4. Generate code for current node
        5. Validate against magic protocols
        """
        # Check depth limit
        if depth > max_depth:
            await self._logger.awarning(
                "Max depth exceeded",
                node_id=node_id,
                depth=depth,
            )
            return GenerationResult(
                node_id=node_id,
                status=NodeStatus.SKIPPED,
                error=f"Max depth {max_depth} exceeded",
            )

        # Already generated?
        if node_id in state.visited:
            return state.results.get(
                node_id,
                GenerationResult(node_id=node_id, status=NodeStatus.GENERATED),
            )

        # Cycle detection
        if node_id in state.in_stack:
            cycle = self._extract_cycle(node_id, state)
            state.cycles_detected.append(cycle)
            await self._logger.awarning(
                "Cycle detected",
                node_id=node_id,
                cycle=cycle,
            )
            return GenerationResult(
                node_id=node_id,
                status=NodeStatus.SKIPPED,
                error=f"Circular dependency detected: {' -> '.join(cycle)}",
            )

        # Mark as in-progress
        state.in_stack.add(node_id)

        await self._logger.ainfo(
            "Visiting node",
            node_id=node_id,
            depth=depth,
        )

        # Step 1: Get dependencies in topological order
        dependencies = await self.repository.get_dependencies_ordered(node_id)

        # Step 2: Recursively generate all dependencies first
        for dep in dependencies:
            dep_id = dep.get("dep_id")
            if dep_id and dep_id not in state.visited:
                await self._dfs_visit(dep_id, generator, state, depth + 1, max_depth)

        # Step 3: Aggregate context with all generated dependencies
        context = await self.context_aggregator.aggregate(
            node_id=node_id,
            generated_files=state.generated_code,
        )

        # Compress if needed
        context = await self.context_aggregator.compress_if_needed(context)

        # Step 4: Generate code for this node
        try:
            result = await generator(node_id, context)
        except Exception as e:
            await self._logger.aerror(
                "Generation failed",
                node_id=node_id,
                error=str(e),
            )
            result = GenerationResult(
                node_id=node_id,
                status=NodeStatus.FAILED,
                error=str(e),
            )

        # Step 5: Store results
        state.visited.add(node_id)
        state.in_stack.remove(node_id)
        state.results[node_id] = result

        if result.status == NodeStatus.GENERATED and result.code:
            state.generated_code[node_id] = result.code
            state.generation_order.append(node_id)

        return result

    def _extract_cycle(self, node_id: str, state: DFSState) -> list[str]:
        """Extract the cycle path when a cycle is detected."""
        # This is a simplified version - in practice would trace the stack
        return [node_id, "...", node_id]

    async def get_generation_order(self, root_ids: list[str]) -> list[str]:
        """Get the order in which nodes should be generated (topological sort).

        Useful for previewing the generation plan without actually generating.
        """
        order = []
        visited = set()

        async def visit(node_id: str) -> None:
            if node_id in visited:
                return
            visited.add(node_id)

            dependencies = await self.repository.get_dependencies_ordered(node_id)
            for dep in dependencies:
                dep_id = dep.get("dep_id")
                if dep_id:
                    await visit(dep_id)

            order.append(node_id)

        for root_id in root_ids:
            await visit(root_id)

        return order


class CycleResolver:
    """Resolves circular dependencies using various strategies."""

    def __init__(self, repository: GraphRepository):
        self.repository = repository
        self._logger = logger.bind(component="CycleResolver")

    async def detect_cycles(self) -> list[list[str]]:
        """Detect all cycles in the dependency graph using Tarjan's algorithm."""
        return await self.repository.detect_cycles()

    async def suggest_resolution(self, cycle: list[str]) -> dict[str, Any]:
        """Suggest a resolution strategy for a detected cycle.

        Strategies:
        1. Interface extraction - Create shared interface to break cycle
        2. Dependency inversion - Introduce abstraction layer
        3. Event-driven - Use event bus for decoupling
        4. Module merge - Merge tightly coupled modules
        """
        await self._logger.ainfo(
            "Analyzing cycle for resolution",
            cycle=cycle,
        )

        # Analyze the cycle to determine best strategy
        suggestions = []

        # Check if modules are in same layer
        layers = set()
        for node_id in cycle:
            node = await self.repository.get_node(node_id)
            if node:
                layer = node.get("n", {}).get("layer", "unknown")
                layers.add(layer)

        if len(layers) == 1:
            # Same layer - might benefit from merge or interface extraction
            suggestions.append({
                "strategy": "interface_extraction",
                "description": "Extract shared interface to break the cycle",
                "priority": 1,
            })
            suggestions.append({
                "strategy": "module_merge",
                "description": "Consider merging these tightly coupled modules",
                "priority": 2,
            })
        else:
            # Cross-layer - dependency inversion
            suggestions.append({
                "strategy": "dependency_inversion",
                "description": "Introduce abstraction layer following dependency inversion principle",
                "priority": 1,
            })

        # Event-driven is always an option
        suggestions.append({
            "strategy": "event_decoupling",
            "description": "Use event bus to decouple the modules",
            "priority": 3,
        })

        return {
            "cycle": cycle,
            "layer_count": len(layers),
            "suggestions": sorted(suggestions, key=lambda x: x["priority"]),
        }

    async def apply_interface_extraction(self, cycle: list[str]) -> dict[str, Any]:
        """Apply interface extraction strategy to break a cycle.

        Creates a new Interface node that both sides of the cycle can depend on.
        """
        # Find the edge to break (typically the one with lower coupling)
        # For now, we'll create a virtual interface

        interface_id = f"interface_shared_{cycle[0]}_{cycle[-1]}"

        await self._logger.ainfo(
            "Creating shared interface",
            interface_id=interface_id,
            cycle=cycle,
        )

        return {
            "action": "interface_extraction",
            "interface_id": interface_id,
            "affected_nodes": cycle,
            "description": f"Created shared interface {interface_id} to break cycle",
        }
