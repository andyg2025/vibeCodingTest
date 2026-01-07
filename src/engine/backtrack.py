"""Backtracking Engine for DFS Code Generation.

Handles violations detected during code generation and triggers
appropriate recovery strategies:
- Regenerate current node with modified constraints
- Notify parent agents to adjust design
- Trigger full redesign for critical violations
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

from src.graph.repository import GraphRepository
from src.agents.codex import Codex, Violation, ViolationType

logger = structlog.get_logger()


class BacktrackLevel(str, Enum):
    """How far back to go when recovering from a violation."""

    CURRENT_NODE = "current_node"  # Regenerate just this node
    PARENT_NODE = "parent_node"  # Go up one level
    MODULE_LEVEL = "module_level"  # Redesign entire module
    GLOBAL = "global"  # Trigger architect to redesign


class BacktrackReason(str, Enum):
    """Why backtracking was triggered."""

    TYPE_MISMATCH = "type_mismatch"
    MAGIC_VIOLATION = "magic_violation"
    CYCLE_DETECTED = "cycle_detected"
    CONSTRAINT_CONFLICT = "constraint_conflict"
    COMPILATION_ERROR = "compilation_error"
    TEST_FAILURE = "test_failure"


@dataclass
class BacktrackAction:
    """An action to take during backtracking."""

    level: BacktrackLevel
    reason: BacktrackReason
    node_id: str
    violations: list[Violation] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    modified_constraints: dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class BacktrackResult:
    """Result of a backtrack operation."""

    success: bool
    action: BacktrackAction
    new_code: str | None = None
    redesign_required: bool = False
    redesign_scope: BacktrackLevel | None = None
    message: str = ""


class BacktrackingEngine:
    """Engine for handling violations and recovery during DFS generation.

    Implements the backtracking logic described in the plan:
    - Type signature mismatch: Regenerate current node
    - Magic protocol violation: Notify Magic-Link Agent
    - Circular dependency: Trigger Architect redesign
    - Runtime constraint conflict: Adjust Logical Designer output
    """

    def __init__(self, repository: GraphRepository, codex: Codex | None = None):
        self.repository = repository
        self.codex = codex
        self._logger = logger.bind(component="BacktrackingEngine")

        # Track backtrack history to avoid infinite loops
        self._backtrack_history: dict[str, list[BacktrackAction]] = {}

    async def handle_violations(
        self,
        node_id: str,
        code: str,
        violations: list[Violation],
    ) -> BacktrackResult:
        """Handle violations detected in generated code.

        Args:
            node_id: The node where violations were detected
            code: The generated code with violations
            violations: List of detected violations

        Returns:
            BacktrackResult with recovery action
        """
        await self._logger.ainfo(
            "Handling violations",
            node_id=node_id,
            violation_count=len(violations),
        )

        # Categorize violations by severity
        categorized = self._categorize_violations(violations)

        # Determine backtrack level based on most severe violation
        level, reason = self._determine_backtrack_level(categorized)

        # Create backtrack action
        action = BacktrackAction(
            level=level,
            reason=reason,
            node_id=node_id,
            violations=violations,
            suggestions=self._generate_suggestions(violations),
            retry_count=self._get_retry_count(node_id),
        )

        # Check if we've exceeded max retries
        if action.retry_count >= action.max_retries:
            await self._logger.awarning(
                "Max retries exceeded, escalating",
                node_id=node_id,
                retry_count=action.retry_count,
            )
            return BacktrackResult(
                success=False,
                action=action,
                redesign_required=True,
                redesign_scope=self._escalate_level(level),
                message=f"Max retries ({action.max_retries}) exceeded for {node_id}",
            )

        # Record this backtrack attempt
        self._record_backtrack(node_id, action)

        # Execute backtrack based on level
        result = await self._execute_backtrack(action, code)

        return result

    def _categorize_violations(
        self, violations: list[Violation]
    ) -> dict[ViolationType, list[Violation]]:
        """Group violations by type."""
        categorized: dict[ViolationType, list[Violation]] = {}
        for v in violations:
            if v.violation_type not in categorized:
                categorized[v.violation_type] = []
            categorized[v.violation_type].append(v)
        return categorized

    def _determine_backtrack_level(
        self, categorized: dict[ViolationType, list[Violation]]
    ) -> tuple[BacktrackLevel, BacktrackReason]:
        """Determine how far to backtrack based on violation types."""

        # Critical violations require global redesign
        if ViolationType.CRITICAL in categorized:
            return BacktrackLevel.GLOBAL, BacktrackReason.MAGIC_VIOLATION

        # Circular dependency requires module-level redesign
        if ViolationType.CIRCULAR_DEPENDENCY in categorized:
            return BacktrackLevel.MODULE_LEVEL, BacktrackReason.CYCLE_DETECTED

        # Magic violations need parent agent intervention
        if ViolationType.MISSING_MAGIC_IMPORT in categorized:
            return BacktrackLevel.PARENT_NODE, BacktrackReason.MAGIC_VIOLATION

        if ViolationType.UNDECLARED_ENV_VAR in categorized:
            return BacktrackLevel.PARENT_NODE, BacktrackReason.MAGIC_VIOLATION

        # Type mismatches can be fixed at current level
        if ViolationType.TYPE_MISMATCH in categorized:
            return BacktrackLevel.CURRENT_NODE, BacktrackReason.TYPE_MISMATCH

        # Default: try to fix at current level
        return BacktrackLevel.CURRENT_NODE, BacktrackReason.CONSTRAINT_CONFLICT

    def _escalate_level(self, current: BacktrackLevel) -> BacktrackLevel:
        """Escalate to next higher level when retries are exhausted."""
        escalation = {
            BacktrackLevel.CURRENT_NODE: BacktrackLevel.PARENT_NODE,
            BacktrackLevel.PARENT_NODE: BacktrackLevel.MODULE_LEVEL,
            BacktrackLevel.MODULE_LEVEL: BacktrackLevel.GLOBAL,
            BacktrackLevel.GLOBAL: BacktrackLevel.GLOBAL,  # Can't escalate further
        }
        return escalation[current]

    def _generate_suggestions(self, violations: list[Violation]) -> list[str]:
        """Generate fix suggestions for violations."""
        suggestions = []

        for v in violations:
            if v.suggestion:
                suggestions.append(v.suggestion)
            else:
                # Generate default suggestions based on type
                match v.violation_type:
                    case ViolationType.MISSING_MAGIC_IMPORT:
                        suggestions.append(
                            f"Add import/dependency for magic protocol: {v.details.get('protocol', 'unknown')}"
                        )
                    case ViolationType.UNDECLARED_ENV_VAR:
                        suggestions.append(
                            f"Declare environment variable in config: {v.details.get('var_name', 'unknown')}"
                        )
                    case ViolationType.TYPE_MISMATCH:
                        suggestions.append(
                            f"Fix type annotation: expected {v.details.get('expected')}, got {v.details.get('actual')}"
                        )
                    case ViolationType.LIFECYCLE_ORDER:
                        suggestions.append(
                            "Check initialization order - ensure dependencies are available before use"
                        )
                    case _:
                        suggestions.append(f"Review and fix: {v.message}")

        return suggestions

    def _get_retry_count(self, node_id: str) -> int:
        """Get current retry count for a node."""
        history = self._backtrack_history.get(node_id, [])
        return len(history)

    def _record_backtrack(self, node_id: str, action: BacktrackAction) -> None:
        """Record a backtrack attempt for a node."""
        if node_id not in self._backtrack_history:
            self._backtrack_history[node_id] = []
        self._backtrack_history[node_id].append(action)

    async def _execute_backtrack(
        self, action: BacktrackAction, original_code: str
    ) -> BacktrackResult:
        """Execute the backtrack action."""
        await self._logger.ainfo(
            "Executing backtrack",
            node_id=action.node_id,
            level=action.level.value,
            reason=action.reason.value,
        )

        match action.level:
            case BacktrackLevel.CURRENT_NODE:
                return await self._backtrack_current_node(action, original_code)

            case BacktrackLevel.PARENT_NODE:
                return await self._backtrack_to_parent(action)

            case BacktrackLevel.MODULE_LEVEL:
                return await self._backtrack_to_module(action)

            case BacktrackLevel.GLOBAL:
                return await self._backtrack_global(action)

    async def _backtrack_current_node(
        self, action: BacktrackAction, original_code: str
    ) -> BacktrackResult:
        """Try to fix issues at current node level."""
        # Build modified constraints based on violations
        constraints = {
            "previous_violations": [v.message for v in action.violations],
            "suggestions": action.suggestions,
            "retry_attempt": action.retry_count + 1,
        }

        action.modified_constraints = constraints

        return BacktrackResult(
            success=True,
            action=action,
            new_code=None,  # Will be regenerated by caller
            message=f"Retry generation with {len(action.suggestions)} fix suggestions",
        )

    async def _backtrack_to_parent(self, action: BacktrackAction) -> BacktrackResult:
        """Escalate to parent node for redesign."""
        # Find parent node
        ancestors = await self.repository.get_ancestor_context(action.node_id, max_depth=1)

        if not ancestors:
            # No parent, escalate to module level
            return BacktrackResult(
                success=False,
                action=action,
                redesign_required=True,
                redesign_scope=BacktrackLevel.MODULE_LEVEL,
                message="No parent found, escalating to module level",
            )

        parent_id = ancestors[0].get("ancestor", {}).get("id")

        return BacktrackResult(
            success=True,
            action=action,
            redesign_required=True,
            redesign_scope=BacktrackLevel.PARENT_NODE,
            message=f"Redesign required at parent level: {parent_id}",
        )

    async def _backtrack_to_module(self, action: BacktrackAction) -> BacktrackResult:
        """Escalate to module level for redesign."""
        return BacktrackResult(
            success=False,
            action=action,
            redesign_required=True,
            redesign_scope=BacktrackLevel.MODULE_LEVEL,
            message="Module-level redesign required - notify Logical Designer Agent",
        )

    async def _backtrack_global(self, action: BacktrackAction) -> BacktrackResult:
        """Escalate to global level for architecture redesign."""
        return BacktrackResult(
            success=False,
            action=action,
            redesign_required=True,
            redesign_scope=BacktrackLevel.GLOBAL,
            message="Architecture redesign required - notify Global Architect Agent",
        )

    def clear_history(self, node_id: str | None = None) -> None:
        """Clear backtrack history for a node or all nodes."""
        if node_id:
            self._backtrack_history.pop(node_id, None)
        else:
            self._backtrack_history.clear()

    def get_history(self, node_id: str) -> list[BacktrackAction]:
        """Get backtrack history for a node."""
        return self._backtrack_history.get(node_id, [])
