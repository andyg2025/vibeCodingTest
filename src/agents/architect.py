"""Global Architect Agent.

Responsible for:
- Top-level framework selection
- Module boundary definition
- Core dependency topology construction
- Technology stack decisions

This is the first agent in the pipeline, transforming user requirements
into a high-level project skeleton graph.
"""

import json
from typing import Any

import structlog
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

from src.graph.repository import GraphRepository
from src.models import (
    ModuleLayer,
    ModuleNode,
    ProjectNode,
    ContainsEdge,
    DependsOnEdge,
    DependencyStrength,
)

from .base import AgentConfig, AgentState, AgentType, BaseAgent
from .state import ArchitectureGraph, ModuleSpec, ProjectSpec, WorkflowState

logger = structlog.get_logger()

ARCHITECT_SYSTEM_PROMPT = """You are the Global Architect Agent in a graph-driven code generation system.

Your responsibilities:
1. Analyze user requirements and extract project specifications
2. Select appropriate technology stack and frameworks
3. Design high-level module architecture using clean architecture principles
4. Define module boundaries and dependencies
5. Identify potential "magic dependencies" that will need special handling

You must output structured JSON following this schema:

{
  "project": {
    "name": "string",
    "description": "string",
    "tech_stack": ["string"],
    "constraints": ["string"]
  },
  "modules": [
    {
      "id": "module_<name>",
      "name": "string",
      "layer": "domain|application|infrastructure|presentation",
      "description": "string",
      "responsibilities": ["string"],
      "dependencies": ["module_id"]
    }
  ],
  "potential_magic": [
    {
      "type": "env|middleware|hook|di|orm",
      "description": "string",
      "affected_modules": ["module_id"]
    }
  ]
}

Architecture Guidelines:
- Use Clean Architecture: Domain (core) -> Application (use cases) -> Infrastructure (external)
- Minimize cross-layer dependencies
- Domain layer should have NO external dependencies
- Infrastructure implements interfaces defined in Application layer
- Identify all implicit/hidden dependencies (environment variables, middleware, etc.)

Be thorough but concise. Focus on the structure, not implementation details."""


class GlobalArchitectAgent(BaseAgent):
    """Agent responsible for top-level architecture design."""

    def __init__(self, repository: GraphRepository | None = None):
        config = AgentConfig(
            name="GlobalArchitect",
            agent_type=AgentType.GLOBAL_ARCHITECT,
            system_prompt=ARCHITECT_SYSTEM_PROMPT,
            temperature=0.1,  # Slight creativity for architecture
        )
        super().__init__(config)
        self.repository = repository or GraphRepository()

    def get_system_prompt(self) -> str:
        return ARCHITECT_SYSTEM_PROMPT

    async def process(self, state: AgentState) -> AgentState:
        """Process requirements and generate architecture."""
        messages = state.get("messages", [])
        context = state.get("context", {})

        # Get user requirements from context or messages
        requirements = context.get("user_requirements", "")
        if not requirements and messages:
            for msg in reversed(messages):
                if isinstance(msg, HumanMessage):
                    requirements = msg.content
                    break

        if not requirements:
            return {
                **state,
                "errors": state.get("errors", []) + ["No user requirements provided"],
                "should_stop": True,
            }

        # Invoke LLM to generate architecture
        prompt = f"""Analyze the following requirements and design the project architecture:

{requirements}

Provide your response as valid JSON following the schema in your instructions."""

        response = await self.invoke([HumanMessage(content=prompt)])

        # Parse the response
        try:
            architecture = self._parse_architecture(response.content)
        except Exception as e:
            await logger.aerror("Failed to parse architecture", error=str(e))
            return {
                **state,
                "errors": state.get("errors", []) + [f"Architecture parsing failed: {e}"],
                "should_stop": True,
            }

        # Store in graph database
        try:
            await self._store_architecture(architecture)
        except Exception as e:
            await logger.aerror("Failed to store architecture", error=str(e))
            return {
                **state,
                "warnings": state.get("warnings", []) + [f"Graph storage failed: {e}"],
            }

        # Update state with architecture
        new_messages = messages + [
            HumanMessage(content=prompt),
            response,
        ]

        return {
            **state,
            "messages": new_messages,
            "context": {
                **context,
                "architecture": architecture,
                "project_spec": architecture.get("project"),
                "modules": architecture.get("modules", []),
                "potential_magic": architecture.get("potential_magic", []),
            },
        }

    def _parse_architecture(self, content: str) -> dict[str, Any]:
        """Parse LLM response into architecture dict."""
        # Extract JSON from response (may be wrapped in markdown)
        content = content.strip()

        if content.startswith("```"):
            # Extract from code block
            lines = content.split("\n")
            json_lines = []
            in_json = False
            for line in lines:
                if line.startswith("```") and not in_json:
                    in_json = True
                    continue
                elif line.startswith("```") and in_json:
                    break
                elif in_json:
                    json_lines.append(line)
            content = "\n".join(json_lines)

        return json.loads(content)

    async def _store_architecture(self, architecture: dict[str, Any]) -> None:
        """Store architecture in the graph database."""
        # Initialize schema
        await self.repository.init_schema()

        project_data = architecture.get("project", {})
        modules_data = architecture.get("modules", [])

        # Create project node
        project = ProjectNode(
            id=f"project_{project_data.get('name', 'unnamed').lower().replace(' ', '_')}",
            name=project_data.get("name", "Unnamed Project"),
            description=project_data.get("description", ""),
            tech_stack=project_data.get("tech_stack", []),
        )
        await self.repository.create_node(project)

        # Create module nodes
        module_ids = {}
        for mod_data in modules_data:
            module = ModuleNode(
                id=mod_data["id"],
                name=mod_data["name"],
                layer=ModuleLayer(mod_data.get("layer", "application")),
                path=f"src/{mod_data['name'].lower()}",
                description=mod_data.get("description", ""),
                metadata={
                    "responsibilities": mod_data.get("responsibilities", []),
                },
            )
            await self.repository.create_node(module)
            module_ids[mod_data["id"]] = module

            # Create CONTAINS edge from project to module
            contains_edge = ContainsEdge(
                source_id=project.id,
                target_id=module.id,
            )
            await self.repository.create_edge(contains_edge)

        # Create module dependency edges
        for mod_data in modules_data:
            for dep_id in mod_data.get("dependencies", []):
                if dep_id in module_ids:
                    depends_edge = DependsOnEdge(
                        source_id=mod_data["id"],
                        target_id=dep_id,
                        strength=DependencyStrength.HARD,
                    )
                    await self.repository.create_edge(depends_edge)

        await logger.ainfo(
            "Stored architecture in graph",
            project_id=project.id,
            module_count=len(modules_data),
        )

    def create_graph(self) -> StateGraph:
        """Create workflow graph for the architect agent."""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("analyze_requirements", self._analyze_requirements_node)
        workflow.add_node("design_architecture", self._design_architecture_node)
        workflow.add_node("validate_architecture", self._validate_architecture_node)

        # Set entry point
        workflow.set_entry_point("analyze_requirements")

        # Add edges
        workflow.add_edge("analyze_requirements", "design_architecture")
        workflow.add_edge("design_architecture", "validate_architecture")
        workflow.add_edge("validate_architecture", END)

        return workflow

    async def _analyze_requirements_node(self, state: AgentState) -> AgentState:
        """Analyze and structure user requirements."""
        context = state.get("context", {})
        requirements = context.get("user_requirements", "")

        await self._logger.ainfo("Analyzing requirements", length=len(requirements))

        # Simple analysis - could be expanded
        return {
            **state,
            "context": {
                **context,
                "requirements_analyzed": True,
            },
        }

    async def _design_architecture_node(self, state: AgentState) -> AgentState:
        """Design the project architecture."""
        return await self.process(state)

    async def _validate_architecture_node(self, state: AgentState) -> AgentState:
        """Validate the designed architecture."""
        context = state.get("context", {})
        architecture = context.get("architecture", {})

        warnings = state.get("warnings", [])

        # Validation checks
        modules = architecture.get("modules", [])

        # Check for orphan modules
        all_deps = set()
        module_ids = set()
        for mod in modules:
            module_ids.add(mod["id"])
            all_deps.update(mod.get("dependencies", []))

        orphan_deps = all_deps - module_ids
        if orphan_deps:
            warnings.append(f"References to undefined modules: {orphan_deps}")

        # Check layer violations
        layer_order = {"domain": 0, "application": 1, "infrastructure": 2, "presentation": 3}
        for mod in modules:
            mod_layer = layer_order.get(mod.get("layer", "application"), 1)
            for dep_id in mod.get("dependencies", []):
                dep_mod = next((m for m in modules if m["id"] == dep_id), None)
                if dep_mod:
                    dep_layer = layer_order.get(dep_mod.get("layer", "application"), 1)
                    if dep_layer > mod_layer:
                        warnings.append(
                            f"Layer violation: {mod['name']} ({mod.get('layer')}) "
                            f"depends on {dep_mod['name']} ({dep_mod.get('layer')})"
                        )

        await self._logger.ainfo(
            "Architecture validated",
            module_count=len(modules),
            warning_count=len(warnings),
        )

        return {
            **state,
            "warnings": warnings,
            "context": {
                **context,
                "architecture_validated": True,
            },
        }
