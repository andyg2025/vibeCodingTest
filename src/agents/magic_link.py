"""Governance & Magic-Link Agent.

The core innovation of this architecture - responsible for:
- Identifying and defining "magic dependencies" that are not visible in imports
- Environment variable injection
- Middleware interception
- Global hooks and lifecycle events
- AOP (Aspect-Oriented Programming) patterns
- Decorator side effects
- ORM relationship magic

This agent creates MagicProtocol nodes and the Dependency Codex (law book)
that governs how other agents must handle these hidden dependencies.
"""

import json
from typing import Any

import structlog
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

from src.graph.repository import GraphRepository
from src.models import (
    MagicProtocolNode,
    MagicProtocolType,
    InfluencedByEdge,
    InjectionPoint,
    BehaviorModification,
    ActivationCondition,
)
from src.models.magic_protocol_schema import (
    MagicProtocolSchema,
    ContractSpec,
    ProvisionSpec,
    RequirementSpec,
    SideEffectSpec,
    InjectionMechanismSpec,
    CodexRuleSpec,
)

from .base import AgentConfig, AgentState, AgentType, BaseAgent

logger = structlog.get_logger()

MAGIC_LINK_SYSTEM_PROMPT = """You are the Governance & Magic-Link Agent in a graph-driven code generation system.

Your critical responsibility is to identify and formally define ALL "magic dependencies" - hidden/implicit dependencies that are NOT visible in import statements but fundamentally affect runtime behavior.

Types of Magic Dependencies to identify:

1. **Environment Variables (env)**
   - Any configuration loaded from environment
   - Secret keys, database URLs, API keys
   - Feature flags

2. **Middleware (middleware)**
   - Request/response interceptors
   - Authentication/authorization middleware
   - Logging and monitoring middleware
   - CORS, rate limiting

3. **Lifecycle Hooks (hook)**
   - Application startup/shutdown events
   - Database connection lifecycle
   - Cache initialization

4. **Dependency Injection (di)**
   - Framework-managed dependencies (FastAPI Depends)
   - Database sessions
   - Service instances

5. **Decorators with Side Effects (decorator)**
   - Caching decorators
   - Retry logic
   - Permission checks
   - Transaction management

6. **ORM Magic (orm)**
   - Lazy-loaded relationships
   - Cascade operations
   - Query generation
   - Session management

For each magic dependency, you must output:

```json
{
  "magic_protocols": [
    {
      "id": "magic:<unique-id>",
      "name": "Human readable name",
      "protocol_type": "env|middleware|hook|di|decorator|orm",
      "framework": "FastAPI|SQLAlchemy|Pydantic|etc",
      "injection_mechanism": {
        "pattern": "dependency_injection|middleware|decorator|singleton|lifecycle",
        "scope": "request|session|application"
      },
      "contract": {
        "provides": {
          "symbol_name": {
            "type": "TypeAnnotation",
            "availability": "when available",
            "nullable": true/false
          }
        },
        "requires": {
          "headers": ["Header-Name"],
          "env_vars": ["ENV_VAR_NAME"],
          "dependencies": ["other_magic_id"]
        },
        "side_effects": [
          {
            "type": "exception|database|cache|log|network",
            "condition": "when this happens",
            "description": "what happens"
          }
        ]
      },
      "influenced_modules": ["module_id"],
      "codex_rules": [
        {
          "rule_id": "RULE-001",
          "description": "What must be done",
          "enforcement_level": "strict|warn|info"
        }
      ]
    }
  ]
}
```

IMPORTANT GUIDELINES:
1. Be EXHAUSTIVE - missing a magic dependency will cause runtime failures
2. Each magic dependency must have clear CONTRACT (what it provides, requires, side effects)
3. CODEX RULES are critical - they define the "laws" that all generated code must follow
4. Consider TRANSITIVE magic - if A depends on magic B, and B depends on magic C, capture all relationships
5. Think about ERROR CASES - what exceptions can be raised? Under what conditions?

Analyze the architecture and tech stack provided, then identify ALL magic dependencies."""


class MagicLinkAgent(BaseAgent):
    """Agent responsible for identifying and defining magic dependencies.

    This is the core innovation of the architecture - capturing hidden
    dependencies that traditional code generation systems miss.
    """

    def __init__(self, repository: GraphRepository | None = None):
        config = AgentConfig(
            name="GovernanceMagicLink",
            agent_type=AgentType.GOVERNANCE_MAGIC_LINK,
            system_prompt=MAGIC_LINK_SYSTEM_PROMPT,
            temperature=0.0,  # Precise, deterministic output
            max_tokens=16384,  # May need more tokens for comprehensive analysis
        )
        super().__init__(config)
        self.repository = repository or GraphRepository()

    def get_system_prompt(self) -> str:
        return MAGIC_LINK_SYSTEM_PROMPT

    async def process(self, state: AgentState) -> AgentState:
        """Analyze architecture and identify magic dependencies."""
        messages = state.get("messages", [])
        context = state.get("context", {})

        # Get architecture from previous agent
        architecture = context.get("architecture", {})
        if not architecture:
            return {
                **state,
                "errors": state.get("errors", []) + ["No architecture provided"],
                "should_stop": True,
            }

        # Build prompt with architecture context
        prompt = self._build_analysis_prompt(architecture)

        # Invoke LLM
        response = await self.invoke([HumanMessage(content=prompt)])

        # Parse magic dependencies
        try:
            magic_protocols = self._parse_magic_protocols(response.content)
        except Exception as e:
            await logger.aerror("Failed to parse magic protocols", error=str(e))
            return {
                **state,
                "errors": state.get("errors", []) + [f"Magic protocol parsing failed: {e}"],
                "should_stop": True,
            }

        # Store in graph database
        try:
            await self._store_magic_protocols(magic_protocols, architecture)
        except Exception as e:
            await logger.aerror("Failed to store magic protocols", error=str(e))
            return {
                **state,
                "warnings": state.get("warnings", []) + [f"Graph storage failed: {e}"],
            }

        # Build the Codex (dependency law book)
        codex = self._build_codex(magic_protocols)

        # Update state
        new_messages = messages + [
            HumanMessage(content=prompt),
            response,
        ]

        return {
            **state,
            "messages": new_messages,
            "context": {
                **context,
                "magic_protocols": magic_protocols,
                "codex": codex,
            },
        }

    def _build_analysis_prompt(self, architecture: dict[str, Any]) -> str:
        """Build the analysis prompt with architecture context."""
        project = architecture.get("project", {})
        modules = architecture.get("modules", [])
        potential_magic = architecture.get("potential_magic", [])

        prompt = f"""Analyze the following project architecture and identify ALL magic dependencies.

## Project
- Name: {project.get('name', 'Unknown')}
- Description: {project.get('description', '')}
- Tech Stack: {', '.join(project.get('tech_stack', []))}

## Modules
"""
        for mod in modules:
            prompt += f"""
### {mod.get('name', 'Unknown')} ({mod.get('layer', 'unknown')} layer)
- ID: {mod.get('id', '')}
- Description: {mod.get('description', '')}
- Responsibilities: {', '.join(mod.get('responsibilities', []))}
- Dependencies: {', '.join(mod.get('dependencies', []))}
"""

        if potential_magic:
            prompt += "\n## Potential Magic (identified by Architect)\n"
            for m in potential_magic:
                prompt += f"- [{m.get('type', 'unknown')}] {m.get('description', '')}\n"
                prompt += f"  Affects: {', '.join(m.get('affected_modules', []))}\n"

        prompt += """

Based on the tech stack and architecture, identify ALL magic dependencies.
Be thorough - consider:
1. FastAPI dependency injection patterns
2. SQLAlchemy ORM magic (sessions, relationships, lazy loading)
3. Pydantic settings and validation
4. Authentication/authorization flows
5. Background task handling
6. Caching mechanisms
7. Email/notification services
8. Logging and monitoring

Output valid JSON with the magic_protocols array."""

        return prompt

    def _parse_magic_protocols(self, content: str) -> list[dict[str, Any]]:
        """Parse LLM response into magic protocols list."""
        content = content.strip()

        # Extract JSON from markdown code blocks if present
        if "```" in content:
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

        data = json.loads(content)
        return data.get("magic_protocols", [])

    async def _store_magic_protocols(
        self, magic_protocols: list[dict[str, Any]], architecture: dict[str, Any]
    ) -> None:
        """Store magic protocols in the graph database."""
        modules = {m["id"]: m for m in architecture.get("modules", [])}

        for mp_data in magic_protocols:
            # Create MagicProtocol node
            mp_node = MagicProtocolNode(
                id=mp_data["id"],
                name=mp_data["name"],
                protocol_type=MagicProtocolType(mp_data.get("protocol_type", "di")),
                framework=mp_data.get("framework", "unknown"),
                provides=mp_data.get("contract", {}).get("provides", {}),
                requires=mp_data.get("contract", {}).get("requires", {}),
                side_effects=mp_data.get("contract", {}).get("side_effects", []),
                codex_rules=[r.get("rule_id", "") for r in mp_data.get("codex_rules", [])],
                metadata={
                    "injection_mechanism": mp_data.get("injection_mechanism", {}),
                    "codex_rules_full": mp_data.get("codex_rules", []),
                },
            )
            await self.repository.create_node(mp_node)

            # Create INFLUENCED_BY edges to modules
            for module_id in mp_data.get("influenced_modules", []):
                if module_id in modules:
                    injection = mp_data.get("injection_mechanism", {})
                    edge = InfluencedByEdge(
                        source_id=module_id,
                        target_id=mp_node.id,
                        injection_point=InjectionPoint(
                            injection.get("pattern", "method").replace("dependency_injection", "method")
                            if injection.get("pattern") not in [e.value for e in InjectionPoint]
                            else injection.get("pattern", "method")
                        ),
                        behavior_modifications=[BehaviorModification.INPUT],
                        activation_condition=ActivationCondition.ALWAYS,
                        priority=50,
                    )
                    await self.repository.create_edge(edge)

        await logger.ainfo(
            "Stored magic protocols",
            count=len(magic_protocols),
        )

    def _build_codex(self, magic_protocols: list[dict[str, Any]]) -> dict[str, Any]:
        """Build the Dependency Codex (law book) from magic protocols.

        The Codex is the central authority that governs how all
        generated code must handle magic dependencies.
        """
        codex = {
            "version": "1.0",
            "rules": [],
            "contracts": {},
            "enforcement_levels": {
                "strict": "Code MUST comply, violations block generation",
                "warn": "Code SHOULD comply, violations generate warnings",
                "info": "Code MAY comply, for documentation purposes",
            },
        }

        for mp in magic_protocols:
            mp_id = mp.get("id", "")
            mp_name = mp.get("name", "")

            # Add contract
            codex["contracts"][mp_id] = {
                "name": mp_name,
                "provides": mp.get("contract", {}).get("provides", {}),
                "requires": mp.get("contract", {}).get("requires", {}),
                "side_effects": mp.get("contract", {}).get("side_effects", []),
            }

            # Add rules
            for rule in mp.get("codex_rules", []):
                codex["rules"].append({
                    "rule_id": rule.get("rule_id", ""),
                    "magic_protocol": mp_id,
                    "description": rule.get("description", ""),
                    "enforcement_level": rule.get("enforcement_level", "strict"),
                })

        return codex

    def create_graph(self) -> StateGraph:
        """Create workflow graph for the magic-link agent."""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("analyze_architecture", self._analyze_architecture_node)
        workflow.add_node("identify_magic", self._identify_magic_node)
        workflow.add_node("build_codex", self._build_codex_node)
        workflow.add_node("validate_completeness", self._validate_completeness_node)

        # Set entry point
        workflow.set_entry_point("analyze_architecture")

        # Add edges
        workflow.add_edge("analyze_architecture", "identify_magic")
        workflow.add_edge("identify_magic", "build_codex")
        workflow.add_edge("build_codex", "validate_completeness")
        workflow.add_edge("validate_completeness", END)

        return workflow

    async def _analyze_architecture_node(self, state: AgentState) -> AgentState:
        """Analyze the architecture for magic dependency hints."""
        context = state.get("context", {})
        architecture = context.get("architecture", {})

        tech_stack = architecture.get("project", {}).get("tech_stack", [])
        await self._logger.ainfo(
            "Analyzing architecture for magic dependencies",
            tech_stack=tech_stack,
        )

        # Identify expected magic patterns based on tech stack
        expected_patterns = []
        for tech in tech_stack:
            tech_lower = tech.lower()
            if "fastapi" in tech_lower:
                expected_patterns.extend([
                    "dependency_injection",
                    "middleware",
                    "startup_events",
                    "background_tasks",
                ])
            if "sqlalchemy" in tech_lower:
                expected_patterns.extend([
                    "orm_sessions",
                    "relationship_loading",
                    "cascade_operations",
                ])
            if "pydantic" in tech_lower:
                expected_patterns.extend([
                    "settings_validation",
                    "env_parsing",
                ])
            if "redis" in tech_lower:
                expected_patterns.extend([
                    "cache_decorator",
                    "session_storage",
                ])
            if "jwt" in tech_lower or "auth" in tech_lower:
                expected_patterns.extend([
                    "auth_middleware",
                    "token_validation",
                ])

        return {
            **state,
            "context": {
                **context,
                "expected_magic_patterns": list(set(expected_patterns)),
            },
        }

    async def _identify_magic_node(self, state: AgentState) -> AgentState:
        """Main node for identifying magic dependencies via LLM."""
        return await self.process(state)

    async def _build_codex_node(self, state: AgentState) -> AgentState:
        """Build and validate the Codex."""
        context = state.get("context", {})
        codex = context.get("codex", {})

        if not codex:
            return {
                **state,
                "errors": state.get("errors", []) + ["Codex not built"],
                "should_stop": True,
            }

        await self._logger.ainfo(
            "Codex built",
            rule_count=len(codex.get("rules", [])),
            contract_count=len(codex.get("contracts", {})),
        )

        return state

    async def _validate_completeness_node(self, state: AgentState) -> AgentState:
        """Validate that all expected magic patterns are covered."""
        context = state.get("context", {})
        expected = set(context.get("expected_magic_patterns", []))
        magic_protocols = context.get("magic_protocols", [])

        # Extract covered patterns from magic protocols
        covered = set()
        for mp in magic_protocols:
            mp_type = mp.get("protocol_type", "")
            injection = mp.get("injection_mechanism", {}).get("pattern", "")
            covered.add(mp_type)
            covered.add(injection)

        # Check for missing patterns
        missing = expected - covered
        warnings = state.get("warnings", [])

        if missing:
            warnings.append(
                f"Potentially missing magic patterns: {', '.join(missing)}"
            )
            await self._logger.awarning(
                "Some expected magic patterns may not be covered",
                missing=list(missing),
            )

        return {
            **state,
            "warnings": warnings,
            "context": {
                **context,
                "magic_validation_complete": True,
            },
        }


# Pre-defined magic protocol factories for common patterns

def create_fastapi_auth_middleware_protocol(
    env_vars: list[str] | None = None,
) -> MagicProtocolSchema:
    """Create a standard FastAPI JWT auth middleware protocol."""
    return MagicProtocolSchema(
        id_="magic:fastapi-auth",
        name="FastAPI JWT Authentication",
        protocol_type="middleware",
        injection_mechanism=InjectionMechanismSpec(
            framework="FastAPI",
            pattern="dependency_injection",
            scope="request",
        ),
        contract=ContractSpec(
            provides={
                "current_user": ProvisionSpec(
                    name="current_user",
                    type_annotation="User | None",
                    availability="after_token_validation",
                    nullable=True,
                )
            },
            requires=RequirementSpec(
                headers=["Authorization"],
                env_vars=env_vars or ["JWT_SECRET", "JWT_ALGORITHM"],
            ),
            side_effects=[
                SideEffectSpec(
                    effect_type="exception",
                    condition="missing_or_invalid_token",
                    raises="HTTPException(401)",
                ),
                SideEffectSpec(
                    effect_type="exception",
                    condition="expired_token",
                    raises="HTTPException(401)",
                ),
            ],
        ),
        codex=CodexRuleSpec(
            rule_id="AUTH-001",
            description="Protected endpoints must declare Depends(get_current_user)",
            enforcement_level="strict",
        ),
    )


def create_sqlalchemy_session_protocol() -> MagicProtocolSchema:
    """Create a standard SQLAlchemy async session protocol."""
    return MagicProtocolSchema(
        id_="magic:sqlalchemy-session",
        name="SQLAlchemy Async Session",
        protocol_type="di",
        injection_mechanism=InjectionMechanismSpec(
            framework="SQLAlchemy",
            pattern="dependency_injection",
            scope="request",
        ),
        contract=ContractSpec(
            provides={
                "db": ProvisionSpec(
                    name="db",
                    type_annotation="AsyncSession",
                    availability="after_startup",
                )
            },
            requires=RequirementSpec(
                env_vars=["DATABASE_URL"],
            ),
            side_effects=[
                SideEffectSpec(
                    effect_type="database",
                    condition="on_success",
                    description="Auto-commits transaction",
                ),
                SideEffectSpec(
                    effect_type="database",
                    condition="on_exception",
                    description="Auto-rollbacks transaction",
                ),
            ],
        ),
        codex=CodexRuleSpec(
            rule_id="DB-001",
            description="Database operations must use injected session, never create own",
            enforcement_level="strict",
        ),
    )


def create_pydantic_settings_protocol(
    env_vars: list[str],
) -> MagicProtocolSchema:
    """Create a Pydantic BaseSettings protocol."""
    return MagicProtocolSchema(
        id_="magic:pydantic-settings",
        name="Pydantic Settings",
        protocol_type="env",
        injection_mechanism=InjectionMechanismSpec(
            framework="Pydantic",
            pattern="singleton",
            scope="application",
        ),
        contract=ContractSpec(
            provides={
                "settings": ProvisionSpec(
                    name="settings",
                    type_annotation="Settings",
                    availability="at_import",
                )
            },
            requires=RequirementSpec(
                env_vars=env_vars,
            ),
            side_effects=[
                SideEffectSpec(
                    effect_type="exception",
                    condition="missing_required_env",
                    raises="ValidationError",
                ),
            ],
        ),
        codex=CodexRuleSpec(
            rule_id="ENV-001",
            description="All env vars must be declared in Settings class",
            enforcement_level="strict",
        ),
    )


def create_background_task_protocol() -> MagicProtocolSchema:
    """Create a FastAPI BackgroundTasks protocol."""
    return MagicProtocolSchema(
        id_="magic:background-tasks",
        name="FastAPI Background Tasks",
        protocol_type="di",
        injection_mechanism=InjectionMechanismSpec(
            framework="FastAPI",
            pattern="dependency_injection",
            scope="request",
        ),
        contract=ContractSpec(
            provides={
                "background_tasks": ProvisionSpec(
                    name="background_tasks",
                    type_annotation="BackgroundTasks",
                    availability="always",
                )
            },
            requires=RequirementSpec(),
            side_effects=[
                SideEffectSpec(
                    effect_type="async",
                    description="Tasks execute after response is sent",
                ),
            ],
        ),
        codex=CodexRuleSpec(
            rule_id="BG-001",
            description="Background tasks must not depend on request context",
            enforcement_level="warn",
        ),
    )
