"""Demo script for the Magic-Link Agent and full pipeline.

This demonstrates Phase 1 (Magic Dependencies):
1. Global Architect generates project architecture
2. Magic-Link Agent identifies all hidden dependencies
3. Codex is built with rules and contracts

Usage:
    export ANTHROPIC_API_KEY="your-key"
    python -m uv run python examples/demo_magic_link.py
"""

import asyncio
import json
import os
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich.tree import Tree

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents import MultiAgentOrchestrator
from src.graph import close_graph_client

console = Console()

# Example requirements for the demo FastAPI e-commerce project
DEMO_REQUIREMENTS = """
Build a FastAPI e-commerce backend with the following features:

1. User Management:
   - User registration and authentication using JWT
   - User profiles with addresses
   - Role-based access control (admin, customer)

2. Product Catalog:
   - Products with categories and tags
   - Product search and filtering
   - Product reviews and ratings

3. Shopping Cart:
   - Add/remove items
   - Persistent cart for logged-in users
   - Guest cart with session storage

4. Order Management:
   - Checkout flow with payment integration
   - Order history and status tracking
   - Email notifications for order status changes

5. Admin Features:
   - Product management (CRUD)
   - Order management
   - User management

Technical Requirements:
- Use PostgreSQL with SQLAlchemy async ORM
- Redis for caching (product listings) and session storage
- Background tasks for email notifications
- Rate limiting on API endpoints
- Proper error handling and structured logging
- Environment-based configuration with Pydantic Settings

This should follow clean architecture principles with clear separation of concerns.
"""


async def run_demo():
    """Run the full Magic-Link Agent demo."""
    console.print(Panel.fit(
        "[bold blue]Graph-Driven DFS Code Agent Architecture[/bold blue]\n"
        "[dim]Phase 1: Magic-Link Agent Demo[/dim]",
        border_style="blue"
    ))

    # Show requirements
    console.print("\n[bold]User Requirements:[/bold]")
    console.print(Panel(DEMO_REQUIREMENTS.strip()[:500] + "...", title="Input", border_style="green"))

    # Initialize orchestrator
    orchestrator = MultiAgentOrchestrator()

    console.print("\n[bold yellow]Running Multi-Agent Pipeline...[/bold yellow]")
    console.print("[dim]Phase 1: Global Architect[/dim]")
    console.print("[dim]Phase 2: Magic-Link Agent[/dim]")
    console.print("[dim]Phase 3: Build Codex[/dim]\n")

    # Run the pipeline
    try:
        result = await orchestrator.run(DEMO_REQUIREMENTS)
    except Exception as e:
        console.print(f"[red]Pipeline Error: {e}[/red]")
        import traceback
        traceback.print_exc()
        return
    finally:
        await close_graph_client()

    # Display results
    if not result.get("success"):
        console.print(f"[red]Pipeline failed at phase: {result.get('phase')}[/red]")
        for err in result.get("errors", []):
            console.print(f"  [red]- {err}[/red]")
        return

    console.print("[green]Pipeline completed successfully![/green]\n")

    # Show architecture
    architecture = result.get("architecture", {})
    project = architecture.get("project", {})

    console.print(Panel(
        f"[bold]{project.get('name', 'Unknown')}[/bold]\n\n"
        f"{project.get('description', '')[:200]}...\n\n"
        f"[dim]Tech Stack: {', '.join(project.get('tech_stack', [])[:5])}...[/dim]",
        title="Project Architecture",
        border_style="cyan"
    ))

    # Modules table
    modules = architecture.get("modules", [])
    if modules:
        table = Table(title=f"Modules ({len(modules)} total)")
        table.add_column("Layer", style="yellow")
        table.add_column("Module", style="green")
        table.add_column("Dependencies", style="dim")

        for mod in modules[:8]:  # Show first 8
            table.add_row(
                mod.get("layer", ""),
                mod.get("name", ""),
                ", ".join(mod.get("dependencies", [])[:2]) or "-"
            )
        if len(modules) > 8:
            table.add_row("...", f"({len(modules) - 8} more)", "...")

        console.print(table)

    # Magic Protocols
    magic_protocols = result.get("magic_protocols", [])
    if magic_protocols:
        console.print(f"\n[bold]Magic Protocols Identified: {len(magic_protocols)}[/bold]")

        # Group by type
        by_type: dict[str, list] = {}
        for mp in magic_protocols:
            mp_type = mp.get("protocol_type", "unknown")
            if mp_type not in by_type:
                by_type[mp_type] = []
            by_type[mp_type].append(mp)

        tree = Tree("[bold]Magic Dependencies[/bold]")
        for mp_type, protocols in by_type.items():
            type_branch = tree.add(f"[yellow]{mp_type}[/yellow] ({len(protocols)})")
            for mp in protocols[:3]:  # Show first 3 per type
                mp_node = type_branch.add(f"[green]{mp.get('name', 'Unknown')}[/green]")
                # Show contract summary
                contract = mp.get("contract", {})
                provides = contract.get("provides", {})
                requires = contract.get("requires", {})
                if provides:
                    mp_node.add(f"[dim]Provides: {', '.join(provides.keys())}[/dim]")
                if requires.get("env_vars"):
                    mp_node.add(f"[dim]Env vars: {', '.join(requires['env_vars'][:3])}[/dim]")
            if len(protocols) > 3:
                type_branch.add(f"[dim]... and {len(protocols) - 3} more[/dim]")

        console.print(tree)

    # Codex summary
    codex = result.get("codex", {})
    if codex:
        rules = codex.get("rules", [])
        contracts = codex.get("contracts", [])

        console.print(Panel(
            f"[bold]Rules:[/bold] {len(rules)}\n"
            f"[bold]Contracts:[/bold] {len(contracts)}\n\n"
            f"[dim]Enforcement Levels:[/dim]\n"
            f"  - strict: Code MUST comply\n"
            f"  - warn: Code SHOULD comply\n"
            f"  - info: Documentation only",
            title="Dependency Codex",
            border_style="magenta"
        ))

        # Show some rules
        if rules:
            table = Table(title="Sample Codex Rules")
            table.add_column("Rule ID", style="cyan")
            table.add_column("Level", style="yellow")
            table.add_column("Description", style="white")

            for rule in rules[:5]:
                table.add_row(
                    rule.get("rule_id", ""),
                    rule.get("enforcement_level", ""),
                    rule.get("description", "")[:50] + "..."
                )

            console.print(table)

    # Show warnings
    warnings = result.get("warnings", [])
    if warnings:
        console.print("\n[yellow]Warnings:[/yellow]")
        for warn in warnings:
            console.print(f"  [yellow]- {warn}[/yellow]")

    # Show sample JSON output
    console.print("\n[bold]Sample Magic Protocol (JSON):[/bold]")
    if magic_protocols:
        sample = magic_protocols[0]
        json_str = json.dumps(sample, indent=2)
        console.print(Syntax(json_str[:1000] + "\n...", "json", theme="monokai"))

    console.print("\n[bold green]Demo completed![/bold green]")
    console.print("[dim]Check Neo4j Browser at http://localhost:7474[/dim]")
    console.print("[dim]User: neo4j | Password: vibecoding123[/dim]")


if __name__ == "__main__":
    asyncio.run(run_demo())
