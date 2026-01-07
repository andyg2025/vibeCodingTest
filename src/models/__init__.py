"""Data models for the DFS Code Agent Architecture."""

from .graph_nodes import (
    BaseNode,
    FileNode,
    InterfaceNode,
    MagicProtocolNode,
    ModuleLayer,
    ModuleNode,
    NodeType,
    ProjectNode,
    MagicProtocolType,
)
from .graph_edges import (
    ActivationCondition,
    BaseEdge,
    BehaviorModification,
    ContainsEdge,
    DependencyStrength,
    DependsOnEdge,
    EdgeType,
    ImplementsEdge,
    ImportsEdge,
    InfluencedByEdge,
    InjectionPoint,
)
from .magic_protocol_schema import (
    CodexRuleSpec,
    ContractSpec,
    InjectionMechanismSpec,
    MagicProtocolSchema,
    ProvisionSpec,
    RequirementSpec,
    SideEffectSpec,
    AUTH_MIDDLEWARE_TEMPLATE,
    DATABASE_SESSION_TEMPLATE,
    ENV_CONFIG_TEMPLATE,
)

__all__ = [
    # Nodes
    "BaseNode",
    "FileNode",
    "InterfaceNode",
    "MagicProtocolNode",
    "ModuleLayer",
    "ModuleNode",
    "NodeType",
    "ProjectNode",
    "MagicProtocolType",
    # Edges
    "ActivationCondition",
    "BaseEdge",
    "BehaviorModification",
    "ContainsEdge",
    "DependencyStrength",
    "DependsOnEdge",
    "EdgeType",
    "ImplementsEdge",
    "ImportsEdge",
    "InfluencedByEdge",
    "InjectionPoint",
    # Magic Protocol Schema
    "CodexRuleSpec",
    "ContractSpec",
    "InjectionMechanismSpec",
    "MagicProtocolSchema",
    "ProvisionSpec",
    "RequirementSpec",
    "SideEffectSpec",
    "AUTH_MIDDLEWARE_TEMPLATE",
    "DATABASE_SESSION_TEMPLATE",
    "ENV_CONFIG_TEMPLATE",
]
