"""Context Aggregator for DFS Code Generation.

Responsible for collecting and merging context from:
- Ancestor nodes (vertical context)
- Sibling nodes (horizontal context)
- Magic protocols (cross-cutting concerns)
"""

from typing import Any

import structlog
from pydantic import BaseModel, Field

from src.graph.repository import GraphRepository
from src.models import MagicProtocolNode, FileNode, ModuleNode

logger = structlog.get_logger()


class ContextPriority:
    """Priority levels for context items."""

    MAGIC_PROTOCOL = 10  # Highest - must be included
    DIRECT_PARENT = 20
    INTERFACE_SIGNATURE = 30
    SIBLING_EXPORTS = 40
    ANCESTOR_SUMMARY = 50
    HISTORICAL_CODE = 60  # Lowest - can be compressed


class ContextItem(BaseModel):
    """A single item of context."""

    source_id: str
    source_type: str  # file, module, magic_protocol, interface
    priority: int
    content: str
    token_estimate: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class AggregatedContext(BaseModel):
    """Aggregated context for code generation."""

    current_node_id: str
    current_node_type: str
    items: list[ContextItem] = Field(default_factory=list)
    magic_protocols: list[dict[str, Any]] = Field(default_factory=list)
    ancestor_chain: list[str] = Field(default_factory=list)
    sibling_interfaces: list[dict[str, Any]] = Field(default_factory=list)
    generated_dependencies: dict[str, str] = Field(default_factory=dict)

    # Token management
    total_tokens: int = 0
    max_tokens: int = 150000  # Reserve space for output

    def add_item(self, item: ContextItem) -> bool:
        """Add a context item if it fits within token limit."""
        if self.total_tokens + item.token_estimate > self.max_tokens:
            return False
        self.items.append(item)
        self.total_tokens += item.token_estimate
        return True

    def get_sorted_items(self) -> list[ContextItem]:
        """Get items sorted by priority (lower = higher priority)."""
        return sorted(self.items, key=lambda x: x.priority)

    def to_prompt_context(self) -> str:
        """Convert to a formatted string for LLM prompt."""
        sections = []

        # Magic protocols (always first)
        if self.magic_protocols:
            sections.append("## Magic Dependencies (MUST comply)")
            for mp in self.magic_protocols:
                sections.append(f"### {mp.get('name', 'Unknown')}")
                sections.append(f"Type: {mp.get('protocol_type', 'unknown')}")
                if mp.get('provides'):
                    sections.append(f"Provides: {mp['provides']}")
                if mp.get('requires'):
                    sections.append(f"Requires: {mp['requires']}")
                if mp.get('codex_rules'):
                    sections.append(f"Rules: {mp['codex_rules']}")
                sections.append("")

        # Ancestor context
        if self.ancestor_chain:
            sections.append("## Ancestor Context")
            sections.append(f"Path: {' -> '.join(self.ancestor_chain)}")
            sections.append("")

        # Sibling interfaces
        if self.sibling_interfaces:
            sections.append("## Available Interfaces")
            for iface in self.sibling_interfaces:
                sections.append(f"- {iface.get('name', 'Unknown')}: {iface.get('signature', '')}")
            sections.append("")

        # Generated code from dependencies
        if self.generated_dependencies:
            sections.append("## Already Generated Dependencies")
            for file_id, code in self.generated_dependencies.items():
                # Only include interface/exports, not full code
                sections.append(f"### {file_id}")
                sections.append("```python")
                sections.append(self._extract_interface(code))
                sections.append("```")
                sections.append("")

        return "\n".join(sections)

    def _extract_interface(self, code: str) -> str:
        """Extract just the interface (imports, class/function signatures) from code."""
        lines = code.split("\n")
        interface_lines = []
        in_docstring = False

        for line in lines:
            stripped = line.strip()

            # Track docstrings
            if '"""' in stripped or "'''" in stripped:
                if stripped.count('"""') == 2 or stripped.count("'''") == 2:
                    # Single-line docstring
                    interface_lines.append(line)
                    continue
                in_docstring = not in_docstring
                interface_lines.append(line)
                continue

            if in_docstring:
                interface_lines.append(line)
                continue

            # Keep imports, class/function definitions, type hints
            if any([
                stripped.startswith("import "),
                stripped.startswith("from "),
                stripped.startswith("class "),
                stripped.startswith("def "),
                stripped.startswith("async def "),
                stripped.startswith("@"),
                ": " in stripped and "=" not in stripped,  # Type annotations
            ]):
                interface_lines.append(line)
            elif interface_lines and interface_lines[-1].strip().endswith(":"):
                # Keep first line after def/class (usually docstring or pass)
                interface_lines.append(line)

        return "\n".join(interface_lines[:50])  # Limit to 50 lines


class ContextAggregator:
    """Aggregates context from various sources for DFS code generation."""

    def __init__(self, repository: GraphRepository):
        self.repository = repository
        self._logger = logger.bind(component="ContextAggregator")

    async def aggregate(
        self,
        node_id: str,
        generated_files: dict[str, str] | None = None,
        max_ancestor_depth: int = 5,
    ) -> AggregatedContext:
        """Aggregate all context for a given node.

        Args:
            node_id: The ID of the node to generate context for
            generated_files: Map of file_id -> generated code for already-generated dependencies
            max_ancestor_depth: Maximum depth to traverse for ancestor context

        Returns:
            AggregatedContext with all relevant context
        """
        generated_files = generated_files or {}

        # Get node info
        node = await self.repository.get_node(node_id)
        if not node:
            raise ValueError(f"Node not found: {node_id}")

        node_type = node.get("labels", ["Unknown"])[0] if "labels" in node else "File"

        context = AggregatedContext(
            current_node_id=node_id,
            current_node_type=node_type,
        )

        # Step 1: Collect magic protocol context (highest priority)
        await self._add_magic_context(context, node_id)

        # Step 2: Collect ancestor context (vertical)
        await self._add_ancestor_context(context, node_id, max_ancestor_depth)

        # Step 3: Collect sibling/dependency context (horizontal)
        await self._add_dependency_context(context, node_id, generated_files)

        await self._logger.ainfo(
            "Context aggregated",
            node_id=node_id,
            magic_count=len(context.magic_protocols),
            ancestor_count=len(context.ancestor_chain),
            dependency_count=len(context.generated_dependencies),
            total_tokens=context.total_tokens,
        )

        return context

    async def _add_magic_context(self, context: AggregatedContext, node_id: str) -> None:
        """Add magic protocol context."""
        magic_influences = await self.repository.get_magic_influences(node_id)

        for influence in magic_influences:
            mp = influence.get("mp", {})
            if mp:
                protocol_data = {
                    "id": mp.get("id"),
                    "name": mp.get("name"),
                    "protocol_type": mp.get("protocol_type"),
                    "provides": mp.get("provides", {}),
                    "requires": mp.get("requires", {}),
                    "side_effects": mp.get("side_effects", []),
                    "codex_rules": mp.get("codex_rules", []),
                }
                context.magic_protocols.append(protocol_data)

                # Add as context item
                context.add_item(ContextItem(
                    source_id=mp.get("id", "unknown"),
                    source_type="magic_protocol",
                    priority=ContextPriority.MAGIC_PROTOCOL,
                    content=str(protocol_data),
                    token_estimate=len(str(protocol_data)) // 4,
                ))

    async def _add_ancestor_context(
        self,
        context: AggregatedContext,
        node_id: str,
        max_depth: int,
    ) -> None:
        """Add ancestor chain context."""
        ancestors = await self.repository.get_ancestor_context(node_id, max_depth)

        for ancestor in ancestors:
            ancestor_node = ancestor.get("ancestor", {})
            if ancestor_node:
                ancestor_id = ancestor_node.get("id", "unknown")
                context.ancestor_chain.append(ancestor_id)

                # Add summary as context item
                context.add_item(ContextItem(
                    source_id=ancestor_id,
                    source_type="ancestor",
                    priority=ContextPriority.DIRECT_PARENT + ancestor.get("depth", 1) * 5,
                    content=f"Ancestor: {ancestor_node.get('name', 'Unknown')}",
                    token_estimate=50,
                    metadata={"depth": ancestor.get("depth", 1)},
                ))

    async def _add_dependency_context(
        self,
        context: AggregatedContext,
        node_id: str,
        generated_files: dict[str, str],
    ) -> None:
        """Add context from dependencies that have already been generated."""
        dependencies = await self.repository.get_dependencies_ordered(node_id)

        for dep in dependencies:
            dep_id = dep.get("dep_id")
            if dep_id and dep_id in generated_files:
                context.generated_dependencies[dep_id] = generated_files[dep_id]

                # Add as context item
                code = generated_files[dep_id]
                context.add_item(ContextItem(
                    source_id=dep_id,
                    source_type="generated_dependency",
                    priority=ContextPriority.SIBLING_EXPORTS,
                    content=code[:2000],  # Truncate
                    token_estimate=len(code) // 4,
                ))

    async def compress_if_needed(
        self,
        context: AggregatedContext,
        target_tokens: int = 100000,
    ) -> AggregatedContext:
        """Compress context if it exceeds target token limit.

        Uses prioritized compression:
        1. Keep all magic protocols (never compress)
        2. Keep direct parent context
        3. Summarize ancestor context
        4. Extract only interfaces from generated code
        """
        if context.total_tokens <= target_tokens:
            return context

        await self._logger.ainfo(
            "Compressing context",
            current_tokens=context.total_tokens,
            target_tokens=target_tokens,
        )

        # Create new context with only essential items
        compressed = AggregatedContext(
            current_node_id=context.current_node_id,
            current_node_type=context.current_node_type,
            magic_protocols=context.magic_protocols,  # Keep all magic
            ancestor_chain=context.ancestor_chain[:3],  # Keep closest 3
            max_tokens=target_tokens,
        )

        # Re-add items by priority until we hit limit
        for item in context.get_sorted_items():
            if item.source_type == "magic_protocol":
                compressed.add_item(item)
            elif compressed.total_tokens + item.token_estimate <= target_tokens:
                compressed.add_item(item)

        await self._logger.ainfo(
            "Context compressed",
            original_tokens=context.total_tokens,
            compressed_tokens=compressed.total_tokens,
        )

        return compressed
