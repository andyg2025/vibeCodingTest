"""Demo script for Phase 3: DFS Generation Engine.

This demonstrates:
1. DFS Walker traversing the dependency graph
2. Context Aggregator collecting context from ancestors and magic protocols
3. Backtracking Engine handling violations
4. DFS Implementation Agent generating code

Usage:
    export ANTHROPIC_API_KEY="your-key"
    python examples/demo_dfs_engine.py
"""

import asyncio
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.tree import Tree

from src.graph.repository import GraphRepository
from src.models import (
    ProjectNode,
    ModuleNode,
    ModuleLayer,
    FileNode,
    MagicProtocolNode,
    MagicProtocolType,
    ContainsEdge,
    DependsOnEdge,
    InfluencedByEdge,
    InjectionPoint,
    BehaviorModification,
    DependencyStrength,
)
from src.engine import (
    DFSWalker,
    ContextAggregator,
    BacktrackingEngine,
    DFSImplementationAgent,
    AggregatedContext,
    GenerationResult,
    NodeStatus,
    DFSState,
)
from src.agents.codex import Codex, CodexBuilder

console = Console()


async def setup_demo_graph(repo: GraphRepository) -> dict[str, str]:
    """Set up a demo dependency graph for testing.

    Creates a mini e-commerce structure:
    - Project
      - Core Module (config, database)
      - API Module (deps, routes/users)
    - Magic Protocols (auth middleware, db session)
    """
    await repo.init_schema()
    await repo.clear_all()

    node_ids = {}

    # Project
    project = ProjectNode(
        id="project_demo",
        name="demo-ecommerce",
        tech_stack=["fastapi", "sqlalchemy", "pydantic"],
        description="Demo e-commerce API",
    )
    await repo.create_node(project)
    node_ids["project"] = project.id

    # Core Module
    core_module = ModuleNode(
        id="module_core",
        name="core",
        layer=ModuleLayer.INFRASTRUCTURE,
        path="src/core",
        description="Core infrastructure - config and database",
    )
    await repo.create_node(core_module)
    node_ids["core"] = core_module.id
    await repo.create_edge(ContainsEdge(source_id=project.id, target_id=core_module.id))

    # API Module
    api_module = ModuleNode(
        id="module_api",
        name="api",
        layer=ModuleLayer.PRESENTATION,
        path="src/api",
        description="API routes and dependencies",
    )
    await repo.create_node(api_module)
    node_ids["api"] = api_module.id
    await repo.create_edge(ContainsEdge(source_id=project.id, target_id=api_module.id))
    await repo.create_edge(DependsOnEdge(
        source_id=api_module.id,
        target_id=core_module.id,
        strength=DependencyStrength.HARD,
    ))

    # Files in Core
    config_file = FileNode(
        id="file_config",
        name="config.py",
        path="src/core/config.py",
        language="python",
        exports=["Settings", "get_settings"],
    )
    await repo.create_node(config_file)
    node_ids["config"] = config_file.id
    await repo.create_edge(ContainsEdge(source_id=core_module.id, target_id=config_file.id))

    database_file = FileNode(
        id="file_database",
        name="database.py",
        path="src/core/database.py",
        language="python",
        exports=["get_db", "AsyncSession", "engine"],
    )
    await repo.create_node(database_file)
    node_ids["database"] = database_file.id
    await repo.create_edge(ContainsEdge(source_id=core_module.id, target_id=database_file.id))

    # Files in API
    deps_file = FileNode(
        id="file_deps",
        name="deps.py",
        path="src/api/deps.py",
        language="python",
        exports=["get_current_user", "get_db_session"],
    )
    await repo.create_node(deps_file)
    node_ids["deps"] = deps_file.id
    await repo.create_edge(ContainsEdge(source_id=api_module.id, target_id=deps_file.id))

    users_file = FileNode(
        id="file_users",
        name="users.py",
        path="src/api/routes/users.py",
        language="python",
        exports=["router"],
    )
    await repo.create_node(users_file)
    node_ids["users"] = users_file.id
    await repo.create_edge(ContainsEdge(source_id=api_module.id, target_id=users_file.id))

    # File dependencies (IMPORTS)
    from src.models import ImportsEdge
    await repo.create_edge(ImportsEdge(
        source_id=database_file.id,
        target_id=config_file.id,
        symbols=["Settings", "get_settings"],
    ))
    await repo.create_edge(ImportsEdge(
        source_id=deps_file.id,
        target_id=database_file.id,
        symbols=["get_db"],
    ))
    await repo.create_edge(ImportsEdge(
        source_id=users_file.id,
        target_id=deps_file.id,
        symbols=["get_current_user", "get_db_session"],
    ))

    # Magic Protocols
    auth_magic = MagicProtocolNode(
        id="magic_auth",
        name="AuthMiddleware",
        protocol_type=MagicProtocolType.MIDDLEWARE,
        framework="FastAPI",
        provides={"current_user": {"type": "User | None", "availability": "after_auth"}},
        requires={"env_vars": ["JWT_SECRET", "JWT_ALGORITHM"]},
        side_effects=[{"type": "exception", "raises": "HTTPException(401)"}],
        codex_rules=["AUTH-001: Protected routes must use get_current_user"],
    )
    await repo.create_node(auth_magic)
    node_ids["auth_magic"] = auth_magic.id

    db_magic = MagicProtocolNode(
        id="magic_db",
        name="DatabaseSession",
        protocol_type=MagicProtocolType.DEPENDENCY_INJECTION,
        framework="FastAPI",
        provides={"db": {"type": "AsyncSession", "availability": "per_request"}},
        requires={"env_vars": ["DATABASE_URL"]},
        side_effects=[{"type": "transaction", "description": "Auto-commit/rollback"}],
        codex_rules=["DB-001: Use injected session, don't create directly"],
    )
    await repo.create_node(db_magic)
    node_ids["db_magic"] = db_magic.id

    # Magic influences
    await repo.create_edge(InfluencedByEdge(
        source_id=users_file.id,
        target_id=auth_magic.id,
        injection_point=InjectionPoint.METHOD,
        behavior_modifications=[BehaviorModification.INPUT],
        priority=10,
    ))
    await repo.create_edge(InfluencedByEdge(
        source_id=users_file.id,
        target_id=db_magic.id,
        injection_point=InjectionPoint.METHOD,
        behavior_modifications=[BehaviorModification.SIDE_EFFECT],
        priority=20,
    ))
    await repo.create_edge(InfluencedByEdge(
        source_id=deps_file.id,
        target_id=db_magic.id,
        injection_point=InjectionPoint.METHOD,
        behavior_modifications=[BehaviorModification.OUTPUT],
        priority=10,
    ))

    return node_ids


def display_graph_structure(node_ids: dict[str, str]) -> None:
    """Display the demo graph structure."""
    tree = Tree("📦 [bold]demo-ecommerce[/bold]")

    core = tree.add("📁 [blue]core[/blue] (infrastructure)")
    core.add("📄 config.py → Settings, get_settings")
    core.add("📄 database.py → get_db, AsyncSession")

    api = tree.add("📁 [green]api[/green] (presentation)")
    api.add("📄 deps.py → get_current_user, get_db_session")
    api.add("📄 routes/users.py → router")

    magic = tree.add("✨ [magenta]Magic Protocols[/magenta]")
    magic.add("🔐 AuthMiddleware → provides current_user")
    magic.add("🗄️ DatabaseSession → provides db session")

    console.print(Panel(tree, title="Demo Graph Structure"))


async def demo_context_aggregation(repo: GraphRepository, node_ids: dict[str, str]) -> None:
    """Demonstrate context aggregation for a node."""
    console.print("\n[bold cyan]═══ Context Aggregation Demo ═══[/bold cyan]\n")

    aggregator = ContextAggregator(repo)

    # Aggregate context for users.py
    target_node = "file_users"
    console.print(f"Aggregating context for: [yellow]{target_node}[/yellow]")

    context = await aggregator.aggregate(
        node_id=target_node,
        generated_files={
            "file_config": "# Generated config.py\nclass Settings:\n    pass",
            "file_database": "# Generated database.py\ndef get_db():\n    pass",
            "file_deps": "# Generated deps.py\ndef get_current_user():\n    pass",
        },
    )

    # Display context
    table = Table(title="Aggregated Context")
    table.add_column("Category", style="cyan")
    table.add_column("Items", style="green")

    table.add_row("Magic Protocols", str(len(context.magic_protocols)))
    table.add_row("Ancestor Chain", " → ".join(context.ancestor_chain) or "None")
    table.add_row("Generated Dependencies", str(len(context.generated_dependencies)))
    table.add_row("Total Token Estimate", str(context.total_tokens))

    console.print(table)

    if context.magic_protocols:
        console.print("\n[bold]Magic Protocol Details:[/bold]")
        for mp in context.magic_protocols:
            console.print(f"  • {mp.get('name')}: provides {list(mp.get('provides', {}).keys())}")


async def demo_dfs_walk(repo: GraphRepository, node_ids: dict[str, str]) -> None:
    """Demonstrate DFS walk order."""
    console.print("\n[bold cyan]═══ DFS Walk Order Demo ═══[/bold cyan]\n")

    walker = DFSWalker(repo)

    # Get generation order starting from API module
    order = await walker.get_generation_order(["module_api"])

    console.print("[bold]DFS Generation Order:[/bold]")
    for i, node_id in enumerate(order, 1):
        console.print(f"  {i}. {node_id}")


async def demo_code_generation(repo: GraphRepository, node_ids: dict[str, str]) -> None:
    """Demonstrate code generation (mock without actual LLM call)."""
    console.print("\n[bold cyan]═══ Code Generation Demo ═══[/bold cyan]\n")

    # Create a mock generator function
    async def mock_generator(node_id: str, context: AggregatedContext) -> GenerationResult:
        """Mock generator that creates placeholder code."""
        code_templates = {
            "file_config": '''"""Application configuration."""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment."""

    DATABASE_URL: str
    JWT_SECRET: str
    JWT_ALGORITHM: str = "HS256"

    class Config:
        env_file = ".env"


def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
''',
            "file_database": '''"""Database configuration and session management."""
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from src.core.config import get_settings

settings = get_settings()
engine = create_async_engine(settings.DATABASE_URL)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def get_db() -> AsyncSession:
    """Yield database session with auto-cleanup."""
    async with AsyncSessionLocal() as session:
        yield session
''',
            "file_deps": '''"""API dependencies."""
from typing import Annotated
from fastapi import Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.database import get_db


async def get_current_user(
    db: Annotated[AsyncSession, Depends(get_db)],
    # token validation logic here
):
    """Get current authenticated user."""
    # Implementation would validate JWT token
    pass


def get_db_session():
    """Alias for get_db."""
    return Depends(get_db)
''',
            "file_users": '''"""User routes."""
from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_current_user, get_db_session

router = APIRouter(prefix="/users", tags=["users"])


@router.get("/me")
async def get_current_user_profile(
    current_user: Annotated[dict, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db_session)],
):
    """Get the current user's profile."""
    return {"user": current_user}


@router.get("/{user_id}")
async def get_user(
    user_id: int,
    db: Annotated[AsyncSession, Depends(get_db_session)],
):
    """Get a user by ID."""
    # Query database for user
    return {"user_id": user_id}
''',
        }

        code = code_templates.get(node_id, f"# Generated code for {node_id}\npass")
        return GenerationResult(
            node_id=node_id,
            status=NodeStatus.GENERATED,
            code=code,
            token_count=len(code) // 4,
        )

    walker = DFSWalker(repo)

    # Walk and generate (using file nodes as roots)
    file_nodes = ["file_config", "file_database", "file_deps", "file_users"]
    state = await walker.walk(
        root_ids=file_nodes,
        generator=mock_generator,
    )

    # Display results
    console.print(f"[bold green]Generated {len(state.generated_code)} files[/bold green]\n")

    for node_id in state.generation_order:
        code = state.generated_code.get(node_id, "")
        console.print(f"[bold]{node_id}[/bold]")
        syntax = Syntax(code[:500] + ("..." if len(code) > 500 else ""), "python", theme="monokai", line_numbers=True)
        console.print(Panel(syntax, title=f"Generated: {node_id}", border_style="green"))
        console.print()


async def demo_backtracking(repo: GraphRepository) -> None:
    """Demonstrate backtracking engine."""
    console.print("\n[bold cyan]═══ Backtracking Demo ═══[/bold cyan]\n")

    from src.agents.codex import Violation, ViolationType, EnforcementLevel

    backtracking = BacktrackingEngine(repo)

    # Simulate violations
    violations = [
        Violation(
            rule_id="AUTH-001",
            violation_type=ViolationType.MISSING_MAGIC_IMPORT,
            message="Protected route missing authentication dependency",
            file_path="src/api/routes/users.py",
            line_number=15,
            fix_suggestion="Add 'current_user: User = Depends(get_current_user)' parameter",
        ),
    ]

    result = await backtracking.handle_violations(
        node_id="file_users",
        code="# Some generated code with violations",
        violations=violations,
    )

    table = Table(title="Backtrack Result")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="yellow")

    table.add_row("Success", str(result.success))
    table.add_row("Level", result.action.level.value)
    table.add_row("Reason", result.action.reason.value)
    table.add_row("Redesign Required", str(result.redesign_required))
    table.add_row("Message", result.message)

    console.print(table)

    if result.action.suggestions:
        console.print("\n[bold]Fix Suggestions:[/bold]")
        for s in result.action.suggestions:
            console.print(f"  • {s}")


async def run_demo():
    """Run the complete Phase 3 demo."""
    console.print(Panel.fit(
        "[bold magenta]Graph-Driven DFS Code Agent Architecture[/bold magenta]\n"
        "[cyan]Phase 3: DFS Generation Engine Demo[/cyan]",
        border_style="bright_blue",
    ))

    # Initialize repository (using mock/in-memory for demo)
    console.print("\n[dim]Initializing graph repository...[/dim]")
    repo = GraphRepository()

    try:
        # Setup demo graph
        console.print("[dim]Setting up demo graph...[/dim]")
        node_ids = await setup_demo_graph(repo)

        # Display structure
        display_graph_structure(node_ids)

        # Run demos
        await demo_context_aggregation(repo, node_ids)
        await demo_dfs_walk(repo, node_ids)
        await demo_code_generation(repo, node_ids)
        await demo_backtracking(repo)

        console.print(Panel.fit(
            "[bold green]✓ Phase 3 Demo Complete![/bold green]\n\n"
            "Components demonstrated:\n"
            "• Context Aggregator - collects magic protocols, ancestors, dependencies\n"
            "• DFS Walker - traverses graph in correct dependency order\n"
            "• Code Generation - generates code with full context\n"
            "• Backtracking Engine - handles violations and suggests fixes",
            border_style="green",
        ))

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(run_demo())
