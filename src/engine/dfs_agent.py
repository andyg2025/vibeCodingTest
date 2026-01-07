"""DFS Implementation Agent.

The agent responsible for actual code generation during DFS traversal.
Uses the DFS Walker, Context Aggregator, and Backtracking Engine.
"""

import time
from typing import Any

import structlog
from langchain_core.messages import HumanMessage, AIMessage

from src.agents.base import AgentConfig, AgentState, AgentType, BaseAgent
from src.agents.codex import Codex
from src.graph.repository import GraphRepository
from src.engine.context import AggregatedContext, ContextAggregator
from src.engine.walker import DFSWalker, DFSState, GenerationResult, NodeStatus
from src.engine.backtrack import BacktrackingEngine, BacktrackLevel

logger = structlog.get_logger()


DFS_IMPLEMENTATION_PROMPT = """You are the DFS Implementation Agent in a graph-driven code generation system.

Your role is to generate production-quality Python code for a specific file node, given:
1. The file specification (name, purpose, exports)
2. Aggregated context from the dependency graph
3. Magic protocol constraints that MUST be followed
4. Already-generated dependency code

## Code Generation Rules

1. **Follow Magic Protocols Strictly**
   - If a magic protocol provides something (e.g., `current_user`), use it correctly
   - If a magic protocol requires something (e.g., env vars), ensure they're declared
   - Never violate codex rules associated with magic protocols

2. **Respect Dependencies**
   - Import from already-generated files using the provided interfaces
   - Use the exact type signatures shown in dependency code
   - Don't create circular imports

3. **Code Quality Standards**
   - Use type hints for all function parameters and returns
   - Add docstrings to classes and public functions
   - Follow PEP 8 style guidelines
   - Handle errors appropriately

4. **Output Format**
   - Return ONLY the Python code, no explanations
   - Include all necessary imports at the top
   - Ensure the code is complete and runnable

## Context Provided

You will receive:
- FILE_SPEC: The file you need to generate
- MAGIC_CONTEXT: Magic protocols that affect this file
- DEPENDENCY_CODE: Interface signatures from already-generated dependencies
- ANCESTOR_CONTEXT: Information about parent modules

Generate complete, production-ready code that integrates correctly with all dependencies and magic protocols."""


class DFSImplementationAgent(BaseAgent):
    """Agent that generates code during DFS traversal."""

    def __init__(
        self,
        repository: GraphRepository | None = None,
        codex: Codex | None = None,
    ):
        config = AgentConfig(
            name="DFSImplementation",
            agent_type=AgentType.DFS_IMPLEMENTATION,
            system_prompt=DFS_IMPLEMENTATION_PROMPT,
            temperature=0.0,  # Deterministic for code generation
            max_tokens=8192,
        )
        super().__init__(config)

        self.repository = repository or GraphRepository()
        self.codex = codex
        self.context_aggregator = ContextAggregator(self.repository)
        self.walker = DFSWalker(self.repository, self.context_aggregator)
        self.backtracking = BacktrackingEngine(self.repository, codex)

    def get_system_prompt(self) -> str:
        return DFS_IMPLEMENTATION_PROMPT

    async def process(self, state: AgentState) -> AgentState:
        """Process generation request from workflow state."""
        context = state.get("context", {})
        node_id = state.get("current_node_id")

        if not node_id:
            return {
                **state,
                "errors": state.get("errors", []) + ["No node_id specified for generation"],
                "should_stop": True,
            }

        # Get already-generated files from state
        generated_files = context.get("generated_files", {})

        # Generate code for this node
        result = await self.generate_single_node(node_id, generated_files)

        if result.status == NodeStatus.GENERATED:
            generated_files[node_id] = result.code
            return {
                **state,
                "context": {
                    **context,
                    "generated_files": generated_files,
                    "last_generated": node_id,
                },
            }
        else:
            return {
                **state,
                "errors": state.get("errors", []) + [result.error or "Generation failed"],
            }

    async def generate_single_node(
        self,
        node_id: str,
        generated_files: dict[str, str],
    ) -> GenerationResult:
        """Generate code for a single node.

        Args:
            node_id: The file node to generate
            generated_files: Already-generated dependency code

        Returns:
            GenerationResult with status and code
        """
        start_time = time.time()

        await self._logger.ainfo(
            "Generating code for node",
            node_id=node_id,
        )

        # Aggregate context
        context = await self.context_aggregator.aggregate(
            node_id=node_id,
            generated_files=generated_files,
        )

        # Compress if needed
        context = await self.context_aggregator.compress_if_needed(context)

        # Get file spec from graph
        node = await self.repository.get_node(node_id)
        if not node:
            return GenerationResult(
                node_id=node_id,
                status=NodeStatus.FAILED,
                error=f"Node not found: {node_id}",
            )

        # Build prompt
        prompt = self._build_generation_prompt(node, context)

        # Invoke LLM
        try:
            response = await self.invoke([HumanMessage(content=prompt)])
            code = self._extract_code(response.content)
        except Exception as e:
            await self._logger.aerror(
                "LLM invocation failed",
                node_id=node_id,
                error=str(e),
            )
            return GenerationResult(
                node_id=node_id,
                status=NodeStatus.FAILED,
                error=str(e),
            )

        # Validate against codex if available
        violations = []
        if self.codex:
            violations = self.codex.check_compliance(code, context.magic_protocols)

            if violations:
                await self._logger.awarning(
                    "Violations detected",
                    node_id=node_id,
                    violation_count=len(violations),
                )

                # Attempt backtracking
                backtrack_result = await self.backtracking.handle_violations(
                    node_id=node_id,
                    code=code,
                    violations=violations,
                )

                if backtrack_result.redesign_required:
                    return GenerationResult(
                        node_id=node_id,
                        status=NodeStatus.FAILED,
                        code=code,
                        violations=[v.model_dump() for v in violations],
                        error=backtrack_result.message,
                    )

                # Retry with modified constraints
                if backtrack_result.action.modified_constraints:
                    retry_prompt = self._build_retry_prompt(
                        node, context, backtrack_result.action.modified_constraints
                    )
                    response = await self.invoke([HumanMessage(content=retry_prompt)])
                    code = self._extract_code(response.content)

        elapsed_ms = int((time.time() - start_time) * 1000)

        await self._logger.ainfo(
            "Code generated successfully",
            node_id=node_id,
            code_lines=code.count("\n") + 1,
            elapsed_ms=elapsed_ms,
        )

        return GenerationResult(
            node_id=node_id,
            status=NodeStatus.GENERATED,
            code=code,
            violations=[v.model_dump() for v in violations] if violations else [],
            token_count=len(code) // 4,  # Rough estimate
            generation_time_ms=elapsed_ms,
        )

    async def generate_all(
        self,
        root_ids: list[str],
        max_depth: int = 50,
    ) -> DFSState:
        """Generate code for all nodes starting from roots.

        Uses DFS Walker to traverse in correct order.
        """
        await self._logger.ainfo(
            "Starting full DFS generation",
            root_count=len(root_ids),
        )

        # Define generator function for walker
        async def generator(node_id: str, context: AggregatedContext) -> GenerationResult:
            # Get already generated code from walker state (passed via closure)
            return await self.generate_single_node(node_id, {})

        # Walk and generate
        state = await self.walker.walk(
            root_ids=root_ids,
            generator=generator,
            max_depth=max_depth,
        )

        await self._logger.ainfo(
            "DFS generation complete",
            total_generated=len(state.generated_code),
            total_failed=sum(1 for r in state.results.values() if r.status == NodeStatus.FAILED),
        )

        return state

    def _build_generation_prompt(
        self,
        node: dict[str, Any],
        context: AggregatedContext,
    ) -> str:
        """Build the prompt for code generation."""
        node_data = node.get("n", node)

        parts = []

        # File specification
        parts.append("## FILE_SPEC")
        parts.append(f"ID: {node_data.get('id', 'unknown')}")
        parts.append(f"Name: {node_data.get('name', 'unknown')}")
        parts.append(f"Path: {node_data.get('path', 'unknown')}")
        if node_data.get("description"):
            parts.append(f"Purpose: {node_data['description']}")
        if node_data.get("exports"):
            parts.append(f"Exports: {node_data['exports']}")
        parts.append("")

        # Context
        parts.append(context.to_prompt_context())

        parts.append("## INSTRUCTIONS")
        parts.append("Generate the complete Python code for this file.")
        parts.append("Return ONLY the code, no explanations or markdown.")

        return "\n".join(parts)

    def _build_retry_prompt(
        self,
        node: dict[str, Any],
        context: AggregatedContext,
        constraints: dict[str, Any],
    ) -> str:
        """Build prompt for retry generation with additional constraints."""
        base_prompt = self._build_generation_prompt(node, context)

        parts = [base_prompt]
        parts.append("\n## PREVIOUS ATTEMPT FAILED")
        parts.append("Please fix the following issues:\n")

        if constraints.get("previous_violations"):
            for v in constraints["previous_violations"]:
                parts.append(f"- {v}")

        if constraints.get("suggestions"):
            parts.append("\nSuggested fixes:")
            for s in constraints["suggestions"]:
                parts.append(f"- {s}")

        return "\n".join(parts)

    def _extract_code(self, content: str) -> str:
        """Extract Python code from LLM response."""
        content = content.strip()

        # Remove markdown code blocks if present
        if content.startswith("```python"):
            content = content[9:]
        elif content.startswith("```"):
            content = content[3:]

        if content.endswith("```"):
            content = content[:-3]

        return content.strip()
