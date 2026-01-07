"""Graph Edge models for the DFS Code Agent Architecture.

Defines the relationship types between nodes:
- IMPORTS: Explicit import dependencies
- DEPENDS_ON: Module-level dependencies
- INFLUENCED_BY: Magic/implicit dependencies (core innovation)
- IMPLEMENTS: Interface implementation
- CONTAINS: Hierarchical containment
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EdgeType(str, Enum):
    """Enumeration of all graph edge types."""

    IMPORTS = "IMPORTS"
    DEPENDS_ON = "DEPENDS_ON"
    INFLUENCED_BY = "INFLUENCED_BY"
    IMPLEMENTS = "IMPLEMENTS"
    CONTAINS = "CONTAINS"


class DependencyStrength(str, Enum):
    """Strength of module dependencies."""

    HARD = "hard"  # Cannot function without this dependency
    SOFT = "soft"  # Optional/enhancement dependency


class InjectionPoint(str, Enum):
    """Where magic is injected into the code."""

    CONSTRUCTOR = "constructor"
    METHOD = "method"
    LIFECYCLE = "lifecycle"  # startup/shutdown hooks
    DECORATOR = "decorator"
    MIDDLEWARE = "middleware"


class BehaviorModification(str, Enum):
    """How magic modifies behavior."""

    INPUT = "input"  # Modifies input data
    OUTPUT = "output"  # Modifies output/return values
    SIDE_EFFECT = "sideEffect"  # Causes side effects
    EXCEPTION = "exception"  # May raise exceptions
    VALIDATION = "validation"  # Validates data


class ActivationCondition(str, Enum):
    """When magic is activated."""

    ALWAYS = "always"
    CONDITIONAL = "conditional"
    ON_ERROR = "on_error"
    ON_SUCCESS = "on_success"


class BaseEdge(BaseModel):
    """Base class for all graph edges."""

    source_id: str = Field(..., description="Source node ID")
    target_id: str = Field(..., description="Target node ID")
    edge_type: EdgeType
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_cypher_properties(self) -> dict[str, Any]:
        """Convert to Neo4j relationship property dictionary."""
        data = self.model_dump(exclude={"source_id", "target_id", "edge_type", "metadata"})
        data.update(self.metadata)
        return data


class ImportsEdge(BaseEdge):
    """Explicit import dependency between files.

    Represents: from module import symbol
    """

    edge_type: EdgeType = EdgeType.IMPORTS
    symbols: list[str] = Field(default_factory=list, description="Imported symbols")
    is_type_only: bool = Field(default=False, description="TYPE_CHECKING import")


class DependsOnEdge(BaseEdge):
    """Module-level dependency.

    Represents logical dependency between modules, which may be
    inferred from file imports or explicitly declared.
    """

    edge_type: EdgeType = EdgeType.DEPENDS_ON
    strength: DependencyStrength = DependencyStrength.HARD
    reason: str = ""


class InfluencedByEdge(BaseEdge):
    """Magic/implicit dependency edge (CORE INNOVATION).

    This edge captures hidden dependencies that are not visible
    in import statements but fundamentally affect code behavior.

    Examples:
    - FastAPI route depends on auth middleware (injects current_user)
    - Function reads environment variable
    - Class method is intercepted by AOP
    - SQLAlchemy model has lazy-loaded relationships
    """

    edge_type: EdgeType = EdgeType.INFLUENCED_BY

    # How the magic is injected
    injection_point: InjectionPoint

    # What behavior is modified
    behavior_modifications: list[BehaviorModification] = Field(default_factory=list)

    # When the magic activates
    activation_condition: ActivationCondition = ActivationCondition.ALWAYS

    # Priority for ordering (lower = higher priority)
    priority: int = Field(default=50, ge=1, le=100)

    # Detailed description of the influence
    description: str = ""

    # Contract enforcement
    codex_rule_id: str | None = None


class ImplementsEdge(BaseEdge):
    """Interface implementation relationship."""

    edge_type: EdgeType = EdgeType.IMPLEMENTS
    partial: bool = Field(default=False, description="Partial implementation")


class ContainsEdge(BaseEdge):
    """Hierarchical containment relationship.

    Project -> Module -> File
    """

    edge_type: EdgeType = EdgeType.CONTAINS
    order: int = Field(default=0, description="Ordering within parent")
