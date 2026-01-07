"""State definitions for the Multi-Agent workflow.

Defines the shared state that flows through the agent pipeline:
1. Global Architect -> project skeleton
2. Governance & Magic-Link -> magic protocols
3. Logical Designer -> file-level design
4. DFS Implementation -> actual code
"""

from typing import Any, TypedDict

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field


class ProjectSpec(BaseModel):
    """Project specification from user requirements."""

    name: str
    description: str
    tech_stack: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    target_path: str = ""


class ModuleSpec(BaseModel):
    """Module specification from architecture design."""

    id: str
    name: str
    layer: str  # domain, application, infrastructure, presentation
    description: str
    responsibilities: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)  # Module IDs


class FileSpec(BaseModel):
    """File specification from logical design."""

    id: str
    module_id: str
    name: str
    path: str
    purpose: str
    exports: list[str] = Field(default_factory=list)
    imports: list[str] = Field(default_factory=list)  # File IDs
    magic_dependencies: list[str] = Field(default_factory=list)  # MagicProtocol IDs


class MagicProtocolSpec(BaseModel):
    """Magic protocol specification from governance agent."""

    id: str
    name: str
    protocol_type: str
    framework: str
    provides: dict[str, Any] = Field(default_factory=dict)
    requires: dict[str, Any] = Field(default_factory=dict)
    side_effects: list[dict[str, Any]] = Field(default_factory=list)
    codex_rules: list[str] = Field(default_factory=list)
    influenced_files: list[str] = Field(default_factory=list)  # File IDs


class ArchitectureGraph(BaseModel):
    """Complete architecture graph from the multi-agent pipeline."""

    project: ProjectSpec
    modules: list[ModuleSpec] = Field(default_factory=list)
    files: list[FileSpec] = Field(default_factory=list)
    magic_protocols: list[MagicProtocolSpec] = Field(default_factory=list)


class WorkflowState(TypedDict, total=False):
    """Complete state for the multi-agent workflow."""

    # Input
    user_requirements: str
    project_spec: dict[str, Any] | None

    # Messages
    messages: list[BaseMessage]

    # Architecture outputs
    architecture_graph: dict[str, Any] | None
    modules: list[dict[str, Any]]
    files: list[dict[str, Any]]
    magic_protocols: list[dict[str, Any]]

    # DFS generation state
    current_node_id: str | None
    generated_files: dict[str, str]  # file_id -> code content
    generation_order: list[str]  # File IDs in DFS order

    # Context for current generation
    ancestor_context: list[dict[str, Any]]
    magic_context: list[dict[str, Any]]
    sibling_context: list[dict[str, Any]]

    # Control flow
    phase: str  # architect, governance, designer, dfs, audit
    iteration: int
    max_iterations: int
    errors: list[str]
    warnings: list[str]
    should_stop: bool

    # Audit results
    violations: list[dict[str, Any]]
    audit_passed: bool


def create_initial_state(requirements: str) -> WorkflowState:
    """Create initial workflow state from user requirements."""
    return WorkflowState(
        user_requirements=requirements,
        project_spec=None,
        messages=[],
        architecture_graph=None,
        modules=[],
        files=[],
        magic_protocols=[],
        current_node_id=None,
        generated_files={},
        generation_order=[],
        ancestor_context=[],
        magic_context=[],
        sibling_context=[],
        phase="architect",
        iteration=0,
        max_iterations=10,
        errors=[],
        warnings=[],
        should_stop=False,
        violations=[],
        audit_passed=False,
    )
