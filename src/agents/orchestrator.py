"""Multi-Agent Orchestrator.

Coordinates the execution of multiple agents in the code generation pipeline:

1. Global Architect Agent -> Project skeleton, modules
2. Governance & Magic-Link Agent -> Magic protocols, Codex
3. Logical Designer Agent -> File-level design (TODO)
4. DFS Implementation Agent -> Code generation (TODO)
5. Audit Agent -> Compliance verification (TODO)

The orchestrator manages:
- Agent sequencing and dependencies
- State propagation between agents
- Error handling and backtracking
- Parallel execution where possible
"""

from typing import Any

import structlog
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

from src.graph.repository import GraphRepository

from .base import AgentState, AgentType
from .architect import GlobalArchitectAgent
from .magic_link import MagicLinkAgent
from .codex import Codex, CodexBuilder

logger = structlog.get_logger()


class MultiAgentOrchestrator:
    """Orchestrates the multi-agent code generation pipeline."""

    def __init__(self, repository: GraphRepository | None = None):
        self.repository = repository or GraphRepository()

        # Initialize agents
        self.architect = GlobalArchitectAgent(repository=self.repository)
        self.magic_link = MagicLinkAgent(repository=self.repository)

        # Codex will be built after magic-link phase
        self.codex: Codex | None = None

    async def run(self, requirements: str) -> dict[str, Any]:
        """Run the full agent pipeline.

        Args:
            requirements: User requirements for the project

        Returns:
            Final state with generated architecture and magic protocols
        """
        # Phase 1: Global Architecture
        await logger.ainfo("Starting Phase 1: Global Architecture")
        architect_state = await self._run_architect(requirements)

        if architect_state.get("should_stop"):
            return {
                "success": False,
                "phase": "architect",
                "errors": architect_state.get("errors", []),
            }

        # Phase 2: Magic Dependencies
        await logger.ainfo("Starting Phase 2: Magic Dependencies")
        magic_state = await self._run_magic_link(architect_state)

        if magic_state.get("should_stop"):
            return {
                "success": False,
                "phase": "magic_link",
                "errors": magic_state.get("errors", []),
            }

        # Build Codex from magic protocols
        await logger.ainfo("Building Codex")
        self.codex = self._build_codex(magic_state)

        # Return combined results
        context = magic_state.get("context", {})
        return {
            "success": True,
            "phase": "magic_link_complete",
            "architecture": context.get("architecture", {}),
            "magic_protocols": context.get("magic_protocols", []),
            "codex": self.codex.to_dict() if self.codex else {},
            "warnings": magic_state.get("warnings", []),
        }

    async def _run_architect(self, requirements: str) -> AgentState:
        """Run the Global Architect Agent."""
        initial_state: AgentState = {
            "messages": [],
            "context": {"user_requirements": requirements},
            "errors": [],
            "iteration": 0,
            "should_stop": False,
        }

        return await self.architect.run(initial_state)

    async def _run_magic_link(self, architect_state: AgentState) -> AgentState:
        """Run the Magic-Link Agent with architect output."""
        # Propagate context from architect
        context = architect_state.get("context", {})

        magic_state: AgentState = {
            "messages": architect_state.get("messages", []),
            "context": context,
            "errors": [],
            "iteration": 0,
            "should_stop": False,
        }

        return await self.magic_link.run(magic_state)

    def _build_codex(self, magic_state: AgentState) -> Codex:
        """Build the Codex from magic protocols."""
        context = magic_state.get("context", {})
        magic_protocols = context.get("magic_protocols", [])

        builder = CodexBuilder()

        # Add each magic protocol
        for mp in magic_protocols:
            builder.add_magic_protocol(mp)

        # Add standard rules
        builder.add_standard_rules()

        return builder.build(freeze=True)

    def create_workflow_graph(self) -> StateGraph:
        """Create a LangGraph workflow for the full pipeline.

        This creates a visual representation of the agent pipeline
        that can be executed step-by-step.
        """
        workflow = StateGraph(AgentState)

        # Add agent nodes
        workflow.add_node("architect", self._architect_node)
        workflow.add_node("magic_link", self._magic_link_node)
        workflow.add_node("build_codex", self._build_codex_node)

        # Set entry point
        workflow.set_entry_point("architect")

        # Add conditional edges
        workflow.add_conditional_edges(
            "architect",
            self._should_continue,
            {
                "continue": "magic_link",
                "stop": END,
            },
        )

        workflow.add_conditional_edges(
            "magic_link",
            self._should_continue,
            {
                "continue": "build_codex",
                "stop": END,
            },
        )

        workflow.add_edge("build_codex", END)

        return workflow

    async def _architect_node(self, state: AgentState) -> AgentState:
        """LangGraph node for architect agent."""
        return await self.architect.run(state)

    async def _magic_link_node(self, state: AgentState) -> AgentState:
        """LangGraph node for magic-link agent."""
        return await self.magic_link.run(state)

    async def _build_codex_node(self, state: AgentState) -> AgentState:
        """LangGraph node for building the Codex."""
        self.codex = self._build_codex(state)

        context = state.get("context", {})
        return {
            **state,
            "context": {
                **context,
                "codex_built": True,
                "codex": self.codex.to_dict(),
            },
        }

    def _should_continue(self, state: AgentState) -> str:
        """Determine if pipeline should continue or stop."""
        if state.get("should_stop") or state.get("errors"):
            return "stop"
        return "continue"


class PipelineResult:
    """Result of a pipeline execution."""

    def __init__(
        self,
        success: bool,
        phase: str,
        architecture: dict[str, Any] | None = None,
        magic_protocols: list[dict[str, Any]] | None = None,
        codex: dict[str, Any] | None = None,
        errors: list[str] | None = None,
        warnings: list[str] | None = None,
    ):
        self.success = success
        self.phase = phase
        self.architecture = architecture or {}
        self.magic_protocols = magic_protocols or []
        self.codex = codex or {}
        self.errors = errors or []
        self.warnings = warnings or []

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "phase": self.phase,
            "architecture": self.architecture,
            "magic_protocols": self.magic_protocols,
            "codex": self.codex,
            "errors": self.errors,
            "warnings": self.warnings,
        }

    @property
    def project_name(self) -> str:
        return self.architecture.get("project", {}).get("name", "Unknown")

    @property
    def module_count(self) -> int:
        return len(self.architecture.get("modules", []))

    @property
    def magic_count(self) -> int:
        return len(self.magic_protocols)

    @property
    def rule_count(self) -> int:
        return len(self.codex.get("rules", []))
