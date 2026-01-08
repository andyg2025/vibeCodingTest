"""Context Window Manager for intelligent context compression.

Implements the context compression strategy from the architecture plan:
1. Preserve essential magic protocols (non-compressible)
2. Summarize ancestor context
3. Extract only interfaces from siblings
4. Select most relevant code by similarity
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class CompressionStrategy(str, Enum):
    """Strategies for context compression."""

    NONE = "none"  # No compression
    SUMMARIZE = "summarize"  # Summarize to key points
    INTERFACE_ONLY = "interface_only"  # Keep only signatures
    RELEVANCE_FILTER = "relevance_filter"  # Filter by relevance score
    TRUNCATE = "truncate"  # Simple truncation


@dataclass
class ContextItem:
    """A single item in the context window."""

    id: str
    content: str
    category: str  # magic_protocol, ancestor, sibling, generated_code
    priority: int = 50  # 0-100, higher = more important
    tokens: int = 0
    compressible: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CompressedContext:
    """Result of context compression."""

    magic_protocols: list[ContextItem]  # Always preserved
    ancestor_summary: str
    sibling_interfaces: list[str]
    relevant_code: list[ContextItem]
    total_tokens: int
    compression_ratio: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_prompt(self) -> str:
        """Convert to a prompt string for LLM."""
        sections = []

        # Magic protocols (highest priority)
        if self.magic_protocols:
            sections.append("## Magic Dependencies (MUST FOLLOW)")
            for mp in self.magic_protocols:
                sections.append(f"### {mp.id}")
                sections.append(mp.content)

        # Ancestor context
        if self.ancestor_summary:
            sections.append("## Project Context")
            sections.append(self.ancestor_summary)

        # Sibling interfaces
        if self.sibling_interfaces:
            sections.append("## Related Interfaces")
            for sig in self.sibling_interfaces:
                sections.append(f"- {sig}")

        # Relevant code
        if self.relevant_code:
            sections.append("## Related Code")
            for item in self.relevant_code:
                sections.append(f"### {item.id}")
                sections.append(f"```python\n{item.content}\n```")

        return "\n\n".join(sections)


class ContextWindowManager:
    """Manages context window to prevent overflow.

    Implements intelligent compression strategies:
    1. Magic protocols are NEVER compressed (essential for correctness)
    2. Ancestor context is summarized to key architectural decisions
    3. Sibling nodes contribute only their interface signatures
    4. Generated code is filtered by relevance to current node
    """

    # Claude Sonnet context limit
    MAX_TOKENS = 180_000
    RESERVE_FOR_OUTPUT = 20_000
    DEFAULT_AVAILABLE = MAX_TOKENS - RESERVE_FOR_OUTPUT

    def __init__(
        self,
        max_tokens: int | None = None,
        reserve_for_output: int = RESERVE_FOR_OUTPUT,
    ):
        self.max_tokens = max_tokens or self.DEFAULT_AVAILABLE
        self.reserve_for_output = reserve_for_output
        self._logger = logger.bind(component="ContextWindowManager")

    def compress_context(
        self,
        items: list[ContextItem],
        current_node_id: str | None = None,
    ) -> CompressedContext:
        """Compress context items to fit within token budget.

        Args:
            items: All context items to consider
            current_node_id: Current node for relevance scoring

        Returns:
            CompressedContext with compressed/filtered items
        """
        # Categorize items
        magic_protocols = [i for i in items if i.category == "magic_protocol"]
        ancestors = [i for i in items if i.category == "ancestor"]
        siblings = [i for i in items if i.category == "sibling"]
        generated = [i for i in items if i.category == "generated_code"]

        # Calculate tokens used by non-compressible items
        magic_tokens = sum(i.tokens for i in magic_protocols)

        # Budget for other categories
        remaining_budget = self.max_tokens - magic_tokens

        # Allocate budget proportionally
        ancestor_budget = int(remaining_budget * 0.25)
        sibling_budget = int(remaining_budget * 0.15)
        code_budget = int(remaining_budget * 0.55)

        # Compress each category
        ancestor_summary = self._compress_ancestors(ancestors, ancestor_budget)
        sibling_interfaces = self._extract_interfaces(siblings, sibling_budget)
        relevant_code = self._filter_by_relevance(
            generated, code_budget, current_node_id
        )

        # Calculate totals
        total_tokens = (
            magic_tokens
            + self._estimate_tokens(ancestor_summary)
            + sum(self._estimate_tokens(s) for s in sibling_interfaces)
            + sum(i.tokens for i in relevant_code)
        )

        original_tokens = sum(i.tokens for i in items)
        compression_ratio = total_tokens / original_tokens if original_tokens > 0 else 1.0

        self._logger.info(
            "Context compressed",
            original_tokens=original_tokens,
            compressed_tokens=total_tokens,
            compression_ratio=f"{compression_ratio:.2%}",
            magic_count=len(magic_protocols),
            code_count=len(relevant_code),
        )

        return CompressedContext(
            magic_protocols=magic_protocols,
            ancestor_summary=ancestor_summary,
            sibling_interfaces=sibling_interfaces,
            relevant_code=relevant_code,
            total_tokens=total_tokens,
            compression_ratio=compression_ratio,
        )

    def _compress_ancestors(
        self, ancestors: list[ContextItem], budget: int
    ) -> str:
        """Compress ancestor context to summary."""
        if not ancestors:
            return ""

        # Sort by priority (project > module > file)
        sorted_ancestors = sorted(ancestors, key=lambda x: -x.priority)

        # Build summary
        lines = []
        tokens_used = 0

        for ancestor in sorted_ancestors:
            # Extract key information
            summary_line = self._summarize_item(ancestor)
            line_tokens = self._estimate_tokens(summary_line)

            if tokens_used + line_tokens > budget:
                break

            lines.append(summary_line)
            tokens_used += line_tokens

        return "\n".join(lines)

    def _extract_interfaces(
        self, siblings: list[ContextItem], budget: int
    ) -> list[str]:
        """Extract only interface signatures from siblings."""
        if not siblings:
            return []

        interfaces = []
        tokens_used = 0

        for sibling in siblings:
            # Extract function/class signatures
            signatures = self._extract_signatures(sibling.content)

            for sig in signatures:
                sig_tokens = self._estimate_tokens(sig)
                if tokens_used + sig_tokens > budget:
                    break
                interfaces.append(sig)
                tokens_used += sig_tokens

        return interfaces

    def _filter_by_relevance(
        self,
        code_items: list[ContextItem],
        budget: int,
        current_node_id: str | None,
    ) -> list[ContextItem]:
        """Filter code by relevance to current node."""
        if not code_items:
            return []

        # Score each item by relevance
        scored_items = []
        for item in code_items:
            score = self._calculate_relevance(item, current_node_id)
            scored_items.append((score, item))

        # Sort by score (highest first)
        scored_items.sort(key=lambda x: -x[0])

        # Select items within budget
        selected = []
        tokens_used = 0

        for score, item in scored_items:
            if tokens_used + item.tokens > budget:
                # Try to fit a truncated version
                truncated = self._truncate_item(item, budget - tokens_used)
                if truncated:
                    selected.append(truncated)
                break

            selected.append(item)
            tokens_used += item.tokens

        return selected

    def _summarize_item(self, item: ContextItem) -> str:
        """Create a summary line for an item."""
        metadata = item.metadata

        if item.category == "ancestor":
            node_type = metadata.get("type", "node")
            name = metadata.get("name", item.id)

            if node_type == "project":
                tech_stack = metadata.get("tech_stack", [])
                return f"Project: {name} (stack: {', '.join(tech_stack[:3])})"
            elif node_type == "module":
                layer = metadata.get("layer", "unknown")
                return f"Module: {name} (layer: {layer})"
            else:
                return f"{node_type.title()}: {name}"

        return f"{item.category}: {item.id}"

    def _extract_signatures(self, code: str) -> list[str]:
        """Extract function and class signatures from code."""
        signatures = []
        lines = code.split("\n")

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Function definitions
            if stripped.startswith("def ") or stripped.startswith("async def "):
                # Get full signature (may span multiple lines)
                sig = stripped
                if ":" not in sig:
                    # Multi-line signature
                    for j in range(i + 1, min(i + 5, len(lines))):
                        sig += " " + lines[j].strip()
                        if ":" in sig:
                            break
                # Extract up to the colon
                sig = sig.split(":")[0] + ":"
                signatures.append(sig)

            # Class definitions
            elif stripped.startswith("class "):
                sig = stripped.split(":")[0] + ":"
                signatures.append(sig)

        return signatures

    def _calculate_relevance(
        self, item: ContextItem, current_node_id: str | None
    ) -> float:
        """Calculate relevance score for an item."""
        score = item.priority / 100.0

        if current_node_id:
            # Boost items that share path components
            current_parts = set(current_node_id.split("/"))
            item_parts = set(item.id.split("/"))
            overlap = len(current_parts & item_parts)
            score += overlap * 0.1

            # Boost items in same module
            if item.metadata.get("module") == current_node_id.split("/")[0]:
                score += 0.2

        return min(score, 1.0)

    def _truncate_item(
        self, item: ContextItem, max_tokens: int
    ) -> ContextItem | None:
        """Truncate an item to fit within token budget."""
        if max_tokens < 50:  # Too small to be useful
            return None

        # Estimate characters per token (rough approximation)
        chars_per_token = 4
        max_chars = max_tokens * chars_per_token

        truncated_content = item.content[:max_chars]
        if len(item.content) > max_chars:
            truncated_content += "\n# ... (truncated)"

        return ContextItem(
            id=item.id,
            content=truncated_content,
            category=item.category,
            priority=item.priority,
            tokens=max_tokens,
            compressible=item.compressible,
            metadata={**item.metadata, "truncated": True},
        )

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.

        Uses a simple heuristic: ~4 characters per token.
        For more accuracy, use tiktoken or similar.
        """
        if not text:
            return 0
        return len(text) // 4 + 1

    def check_budget(self, items: list[ContextItem]) -> dict[str, Any]:
        """Check if items fit within budget.

        Returns:
            Dict with budget analysis
        """
        total_tokens = sum(i.tokens for i in items)
        over_budget = total_tokens > self.max_tokens

        return {
            "total_tokens": total_tokens,
            "max_tokens": self.max_tokens,
            "over_budget": over_budget,
            "overflow": max(0, total_tokens - self.max_tokens),
            "utilization": total_tokens / self.max_tokens,
        }

    def prioritize_items(
        self, items: list[ContextItem]
    ) -> list[ContextItem]:
        """Sort items by priority for inclusion.

        Priority order:
        1. Magic protocols (never dropped)
        2. Direct dependencies
        3. Ancestor context
        4. Sibling interfaces
        5. Generated code (by relevance)
        """
        category_priority = {
            "magic_protocol": 100,
            "dependency": 80,
            "ancestor": 60,
            "sibling": 40,
            "generated_code": 20,
        }

        def sort_key(item: ContextItem) -> tuple[int, int]:
            cat_priority = category_priority.get(item.category, 0)
            return (-cat_priority, -item.priority)

        return sorted(items, key=sort_key)


class ContextBuilder:
    """Builder for constructing context with automatic management."""

    def __init__(self, manager: ContextWindowManager | None = None):
        self.manager = manager or ContextWindowManager()
        self.items: list[ContextItem] = []
        self._logger = logger.bind(component="ContextBuilder")

    def add_magic_protocol(
        self,
        protocol_id: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> "ContextBuilder":
        """Add a magic protocol (non-compressible)."""
        self.items.append(ContextItem(
            id=protocol_id,
            content=content,
            category="magic_protocol",
            priority=100,
            tokens=self.manager._estimate_tokens(content),
            compressible=False,
            metadata=metadata or {},
        ))
        return self

    def add_ancestor(
        self,
        node_id: str,
        content: str,
        node_type: str = "node",
        priority: int = 60,
        metadata: dict[str, Any] | None = None,
    ) -> "ContextBuilder":
        """Add ancestor context."""
        meta = metadata or {}
        meta["type"] = node_type
        self.items.append(ContextItem(
            id=node_id,
            content=content,
            category="ancestor",
            priority=priority,
            tokens=self.manager._estimate_tokens(content),
            compressible=True,
            metadata=meta,
        ))
        return self

    def add_sibling(
        self,
        node_id: str,
        content: str,
        priority: int = 40,
        metadata: dict[str, Any] | None = None,
    ) -> "ContextBuilder":
        """Add sibling context."""
        self.items.append(ContextItem(
            id=node_id,
            content=content,
            category="sibling",
            priority=priority,
            tokens=self.manager._estimate_tokens(content),
            compressible=True,
            metadata=metadata or {},
        ))
        return self

    def add_generated_code(
        self,
        file_id: str,
        code: str,
        priority: int = 50,
        metadata: dict[str, Any] | None = None,
    ) -> "ContextBuilder":
        """Add previously generated code."""
        self.items.append(ContextItem(
            id=file_id,
            content=code,
            category="generated_code",
            priority=priority,
            tokens=self.manager._estimate_tokens(code),
            compressible=True,
            metadata=metadata or {},
        ))
        return self

    def build(self, current_node_id: str | None = None) -> CompressedContext:
        """Build and compress the context."""
        return self.manager.compress_context(self.items, current_node_id)

    def check_budget(self) -> dict[str, Any]:
        """Check current budget status."""
        return self.manager.check_budget(self.items)
