"""Demo script for the Global Architect Agent.

This demonstrates Phase 0 PoC:
1. Takes user requirements as input
2. Invokes the Global Architect Agent
3. Stores the architecture in Neo4j
4. Displays the results

Usage:
    uv run python examples/demo_architect.py
"""

import asyncio
import os
from pprint import pprint

from langchain_core.messages import HumanMessage
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax

# Add src to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents import GlobalArchitectAgent, AgentState
from src.graph import GraphRepository, close_graph_client

console = Console()

# Example requirements for the demo FastAPI e-commerce project
DEMO_REQUIREMENTS = """
Build a FastAPI e-commerce backend with the following features:

1. User Management:
   - User registration and authentication using JWT
   - User profiles with addresses

2. Product Catalog:
   - Products with categories
   - Product search and filtering
   - Product reviews and ratings

3. Shopping Cart:
   - Add/remove items
   - Persistent cart for logged-in users

4. Order Management:
   - Checkout flow
   - Order history
   - Email notifications for order status

Technical Requirements:
- Use PostgreSQL with SQLAlchemy ORM
- Redis for caching and session storage
- Background tasks for email notifications
- Proper error handling and logging

This should follow clean architecture principles with clear separation of concerns.
"""


async def run_demo():
    """Run the Global Architect Agent demo."""
    console.print(Panel.fit(
        "[bold blue]Graph-Driven DFS Code Agent Architecture[/bold blue]\n"
        "[dim]Phase 0 PoC: Global Architect Agent Demo[/dim]",
        border_style="blue"
    ))

    # Show requirements
    console.print("\n[bold]User Requirements:[/bold]")
    console.print(Panel(DEMO_REQUIREMENTS.strip(), title="Input", border_style="green"))

    # Initialize repository and agent
    repo = GraphRepository()
    agent = GlobalArchitectAgent(repository=repo)

    # Create initial state
    state: AgentState = {
        "messages": [],
        "context": {"user_requirements": DEMO_REQUIREMENTS},
        "errors": [],
        "iteration": 0,
        "should_stop": False,
    }

    console.print("\n[bold yellow]Running Global Architect Agent...[/bold yellow]\n")

    # Run the agent
    try:
        result = await agent.run(state)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise
    finally:
        await close_graph_client()

    # Display results
    if result.get("errors"):
        console.print("[red]Errors occurred:[/red]")
        for err in result["errors"]:
            console.print(f"  - {err}")
        return

    if result.get("warnings"):
        console.print("[yellow]Warnings:[/yellow]")
        for warn in result["warnings"]:
            console.print(f"  - {warn}")

    # Show architecture
    context = result.get("context", {})
    architecture = context.get("architecture", {})

    if architecture:
        # Project info
        project = architecture.get("project", {})
        console.print(Panel(
            f"[bold]{project.get('name', 'Unknown')}[/bold]\n\n"
            f"{project.get('description', '')}\n\n"
            f"[dim]Tech Stack: {', '.join(project.get('tech_stack', []))}[/dim]",
            title="Project",
            border_style="cyan"
        ))

        # Modules table
        modules = architecture.get("modules", [])
        if modules:
            table = Table(title="Architecture Modules")
            table.add_column("ID", style="cyan")
            table.add_column("Name", style="green")
            table.add_column("Layer", style="yellow")
            table.add_column("Dependencies", style="dim")

            for mod in modules:
                table.add_row(
                    mod.get("id", ""),
                    mod.get("name", ""),
                    mod.get("layer", ""),
                    ", ".join(mod.get("dependencies", [])) or "-"
                )

            console.print(table)

        # Potential magic dependencies
        magic = architecture.get("potential_magic", [])
        if magic:
            console.print("\n[bold]Identified Magic Dependencies:[/bold]")
            for m in magic:
                console.print(f"  - [{m.get('type', 'unknown')}] {m.get('description', '')}")
                affected = m.get("affected_modules", [])
                if affected:
                    console.print(f"    [dim]Affects: {', '.join(affected)}[/dim]")

        # Show raw JSON
        console.print("\n[bold]Raw Architecture JSON:[/bold]")
        import json
        json_str = json.dumps(architecture, indent=2)
        console.print(Syntax(json_str, "json", theme="monokai", line_numbers=True))

    console.print("\n[bold green]Demo completed successfully![/bold green]")
    console.print("[dim]Check Neo4j Browser at http://localhost:7474 to see the graph.[/dim]")


if __name__ == "__main__":
    asyncio.run(run_demo())
