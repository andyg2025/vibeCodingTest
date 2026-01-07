"""Graph Node models for the DFS Code Agent Architecture.

Defines the core node types used in the dependency graph:
- Project: Top-level project container
- Module: Logical code modules (domain, infrastructure, application)
- File: Individual source files
- MagicProtocol: Hidden/implicit dependency definitions
- Interface: Abstract contracts between modules
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class NodeType(str, Enum):
    """Enumeration of all graph node types."""

    PROJECT = "Project"
    MODULE = "Module"
    FILE = "File"
    MAGIC_PROTOCOL = "MagicProtocol"
    INTERFACE = "Interface"


class ModuleLayer(str, Enum):
    """Module architectural layer classification."""

    DOMAIN = "domain"
    APPLICATION = "application"
    INFRASTRUCTURE = "infrastructure"
    PRESENTATION = "presentation"


class MagicProtocolType(str, Enum):
    """Types of magic/implicit dependencies."""

    ENV = "env"  # Environment variables
    MIDDLEWARE = "middleware"  # Request/response interceptors
    HOOK = "hook"  # Lifecycle hooks (startup, shutdown)
    AOP = "aop"  # Aspect-oriented programming
    DECORATOR = "decorator"  # Python decorators with side effects
    DEPENDENCY_INJECTION = "di"  # Framework DI containers
    ORM = "orm"  # ORM relationship magic


class BaseNode(BaseModel):
    """Base class for all graph nodes."""

    id: str = Field(..., description="Unique identifier for the node")
    name: str = Field(..., description="Human-readable name")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_cypher_properties(self) -> dict[str, Any]:
        """Convert to Neo4j property dictionary."""
        data = self.model_dump(exclude={"metadata"})
        data["created_at"] = data["created_at"].isoformat()
        data.update(self.metadata)
        return data


class ProjectNode(BaseNode):
    """Top-level project container node."""

    node_type: NodeType = NodeType.PROJECT
    tech_stack: list[str] = Field(default_factory=list)
    description: str = ""
    root_path: str = ""


class ModuleNode(BaseNode):
    """Logical code module node."""

    node_type: NodeType = NodeType.MODULE
    layer: ModuleLayer
    path: str = Field(..., description="Relative path within project")
    description: str = ""


class FileNode(BaseNode):
    """Individual source file node."""

    node_type: NodeType = NodeType.FILE
    path: str = Field(..., description="Full file path")
    language: str = "python"
    content_hash: str = ""
    line_count: int = 0
    exports: list[str] = Field(default_factory=list, description="Exported symbols")


class MagicProtocolNode(BaseNode):
    """Magic/implicit dependency protocol node.

    This is the core innovation - tracking hidden dependencies that
    are not visible in import statements but affect runtime behavior.
    """

    node_type: NodeType = NodeType.MAGIC_PROTOCOL
    protocol_type: MagicProtocolType
    framework: str = Field(..., description="Framework that provides this magic")

    # Contract definition
    provides: dict[str, Any] = Field(
        default_factory=dict, description="What this protocol provides to consumers"
    )
    requires: dict[str, Any] = Field(
        default_factory=dict, description="What this protocol requires to function"
    )
    side_effects: list[dict[str, Any]] = Field(
        default_factory=list, description="Side effects of this protocol"
    )

    # Codex rules
    codex_rules: list[str] = Field(
        default_factory=list, description="Rules that govern usage of this protocol"
    )
    enforcement_level: str = Field(
        default="strict", description="How strictly rules are enforced: strict|warn|info"
    )

    def to_cypher_properties(self) -> dict[str, Any]:
        """Convert to Neo4j properties, serializing complex types as JSON."""
        import json

        # Get base properties
        data = {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "node_type": self.node_type.value,
            "protocol_type": self.protocol_type.value,
            "framework": self.framework,
            "enforcement_level": self.enforcement_level,
            # Serialize complex types as JSON strings
            "provides_json": json.dumps(self.provides) if self.provides else "",
            "requires_json": json.dumps(self.requires) if self.requires else "",
            "side_effects_json": json.dumps(self.side_effects) if self.side_effects else "[]",
            "codex_rules": self.codex_rules,  # List of strings is OK
        }
        # Add simple metadata fields
        for k, v in self.metadata.items():
            if isinstance(v, (str, int, float, bool)):
                data[k] = v
            elif isinstance(v, list) and all(isinstance(i, (str, int, float, bool)) for i in v):
                data[k] = v
        return data


class InterfaceNode(BaseNode):
    """Abstract interface/contract node."""

    node_type: NodeType = NodeType.INTERFACE
    signature: str = Field(..., description="Interface signature definition")
    methods: list[dict[str, Any]] = Field(default_factory=list)
    doc: str = ""

    def to_cypher_properties(self) -> dict[str, Any]:
        """Convert to Neo4j properties, serializing methods as JSON."""
        import json

        data = super().to_cypher_properties()
        # Override methods with JSON string
        data["methods_json"] = json.dumps(self.methods) if self.methods else "[]"
        data.pop("methods", None)
        return data
