"""Demo script for Phase 5: Integration & Optimization.

This demonstrates:
1. Context Window Manager - intelligent context compression
2. Incremental Update Pipeline - minimal regeneration
3. End-to-End Pipeline - complete audit workflow
4. Performance Benchmarks - throughput measurements

Usage:
    python examples/demo_phase5.py
"""

import asyncio
import time
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from src.agents.codex import CodexBuilder, CodexRule, EnforcementLevel
from src.engine.context_manager import (
    ContextWindowManager,
    ContextBuilder,
    ContextItem,
)
from src.engine.incremental import (
    IncrementalUpdatePipeline,
    ChangeDetector,
)
from src.audit import (
    CodeAnalyzer,
    ViolationDetector,
    DetectionContext,
    ReportGenerator,
    ReportFormat,
    AutoFixSuggester,
    AuditAgent,
)

console = Console()


# Sample project files for demonstration
SAMPLE_PROJECT_FILES = {
    "src/core/config.py": '''"""Application configuration."""
import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    DATABASE_URL: str
    JWT_SECRET: str
    DEBUG: bool = False

    class Config:
        env_file = ".env"


def get_settings() -> Settings:
    """Get application settings."""
    return Settings()
''',

    "src/core/database.py": '''"""Database connection."""
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from src.core.config import get_settings

settings = get_settings()
engine = create_async_engine(settings.DATABASE_URL)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession)


async def get_db() -> AsyncSession:
    """Get database session."""
    async with AsyncSessionLocal() as session:
        yield session
''',

    "src/api/deps.py": '''"""API dependencies."""
from typing import Annotated
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.database import get_db
from src.core.config import get_settings

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict:
    """Get current authenticated user."""
    settings = get_settings()
    # Verify token using JWT_SECRET
    # ... token verification logic ...
    return {"id": 1, "username": "testuser"}
''',

    "src/api/routes/users.py": '''"""User routes."""
from typing import Annotated
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_current_user, get_db

router = APIRouter(prefix="/users", tags=["users"])


@router.get("/me")
async def get_profile(
    current_user: Annotated[dict, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Get current user profile."""
    return {"user": current_user}
''',

    "src/api/routes/admin.py": '''"""Admin routes with intentional violations."""
import os
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

router = APIRouter(prefix="/admin", tags=["admin"])

# Violation: Undeclared env var
ADMIN_SECRET = os.getenv("UNDECLARED_ADMIN_SECRET")


@router.get("/users")
async def list_all_users(db: Session = Depends(get_db)):
    """List all users - MISSING AUTH DEPENDENCY."""
    return {"users": []}


@router.delete("/users/{user_id}")
async def delete_user(user_id: int):
    """Delete user - MISSING AUTH AND DB."""
    return {"deleted": user_id}
''',
}


def demo_context_compression():
    """Demonstrate the Context Window Manager."""
    console.print("\n[bold cyan]═══ Context Window Manager Demo ═══[/bold cyan]\n")

    manager = ContextWindowManager(max_tokens=10000)
    builder = ContextBuilder(manager)

    # Add magic protocols (non-compressible)
    builder.add_magic_protocol(
        "magic:auth-middleware",
        """AuthenticationMiddleware:
- Provides: current_user (User | None)
- Requires: JWT_SECRET env var
- Side effect: HTTPException(401) on invalid token""",
        metadata={"framework": "FastAPI"},
    )

    builder.add_magic_protocol(
        "magic:db-session",
        """DatabaseSession:
- Provides: db (AsyncSession)
- Requires: DATABASE_URL env var
- Lifecycle: Request-scoped""",
        metadata={"framework": "SQLAlchemy"},
    )

    # Add ancestor context
    builder.add_ancestor(
        "project:vibecoding",
        "Tech stack: FastAPI, SQLAlchemy, Pydantic",
        node_type="project",
        priority=80,
        metadata={"tech_stack": ["fastapi", "sqlalchemy"]},
    )

    builder.add_ancestor(
        "module:api",
        "API module with REST endpoints",
        node_type="module",
        priority=60,
    )

    # Add siblings
    for i, (path, code) in enumerate(list(SAMPLE_PROJECT_FILES.items())[:3]):
        builder.add_sibling(
            f"file:{path}",
            code,
            priority=40,
            metadata={"module": path.split("/")[1]},
        )

    # Add generated code
    for path, code in list(SAMPLE_PROJECT_FILES.items())[3:]:
        builder.add_generated_code(
            f"file:{path}",
            code,
            priority=30,
            metadata={"module": path.split("/")[1]},
        )

    # Check budget before compression
    budget_status = builder.check_budget()

    table = Table(title="Context Budget Analysis")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Tokens", str(budget_status["total_tokens"]))
    table.add_row("Max Tokens", str(budget_status["max_tokens"]))
    table.add_row("Over Budget", "Yes" if budget_status["over_budget"] else "No")
    table.add_row("Utilization", f"{budget_status['utilization']:.1%}")

    console.print(table)

    # Compress context
    console.print("\n[bold]Compressing context...[/bold]")
    compressed = builder.build(current_node_id="file:src/api/routes/admin.py")

    # Show compression results
    results_table = Table(title="Compression Results")
    results_table.add_column("Category", style="cyan")
    results_table.add_column("Count/Size", style="green")

    results_table.add_row("Magic Protocols", str(len(compressed.magic_protocols)))
    results_table.add_row("Ancestor Summary", f"{len(compressed.ancestor_summary)} chars")
    results_table.add_row("Sibling Interfaces", str(len(compressed.sibling_interfaces)))
    results_table.add_row("Relevant Code", str(len(compressed.relevant_code)))
    results_table.add_row("Total Tokens", str(compressed.total_tokens))
    results_table.add_row("Compression Ratio", f"{compressed.compression_ratio:.1%}")

    console.print(results_table)

    # Show sample of compressed context
    console.print("\n[bold]Sample Compressed Context:[/bold]")
    prompt = compressed.to_prompt()
    console.print(Panel(prompt[:1500] + "...", title="Context Prompt (truncated)"))


async def demo_incremental_updates():
    """Demonstrate the Incremental Update Pipeline."""
    console.print("\n[bold cyan]═══ Incremental Update Pipeline Demo ═══[/bold cyan]\n")

    pipeline = IncrementalUpdatePipeline()

    # Initial state
    console.print("[bold]1. Initial file state:[/bold]")
    initial_files = dict(SAMPLE_PROJECT_FILES)

    result = await pipeline.process_update(initial_files)
    console.print(f"   Status: {result['status']}")
    console.print(f"   Files tracked: {result['statistics']['total_files']}")

    # Make some changes
    console.print("\n[bold]2. Making changes...[/bold]")
    modified_files = dict(initial_files)

    # Modify admin routes (fix some violations)
    modified_files["src/api/routes/admin.py"] = '''"""Admin routes - partially fixed."""
from typing import Annotated
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_current_user, get_db

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/users")
async def list_all_users(
    current_user: Annotated[dict, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """List all users - NOW WITH AUTH."""
    return {"users": []}
'''

    # Add a new file
    modified_files["src/api/routes/orders.py"] = '''"""Order routes."""
from typing import Annotated
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_current_user, get_db

router = APIRouter(prefix="/orders", tags=["orders"])


@router.get("/")
async def list_orders(
    current_user: Annotated[dict, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """List user orders."""
    return {"orders": []}
'''

    # Process the update
    result = await pipeline.process_update(modified_files)

    # Show results
    table = Table(title="Incremental Update Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Status", result["status"])
    table.add_row("Files Changed", str(result["statistics"]["changed"]))
    table.add_row("Files Affected", str(result["statistics"]["affected"]))
    table.add_row("Files to Regenerate", str(len(result["plan"]["regenerate"])))
    table.add_row("Files Skipped", str(result["statistics"]["skipped"]))
    table.add_row("Savings", f"{result['statistics']['savings']:.1%}")

    console.print(table)

    # Show changes
    console.print("\n[bold]Changes detected:[/bold]")
    for change in result["changes"]:
        icon = {"created": "+", "modified": "~", "deleted": "-"}[change["type"]]
        console.print(f"  [{icon}] {change['path']} ({change['type']})")

    # Show plan
    console.print("\n[bold]Regeneration plan:[/bold]")
    console.print(f"  To regenerate: {result['plan']['regenerate']}")
    console.print(f"  To skip: {len(result['plan']['skip'])} files")


async def demo_end_to_end_pipeline():
    """Demonstrate the complete end-to-end pipeline."""
    console.print("\n[bold cyan]═══ End-to-End Pipeline Demo ═══[/bold cyan]\n")

    # Build codex
    builder = CodexBuilder()
    builder.add_standard_rules()
    codex = builder.build(freeze=True)

    # Create audit agent
    agent = AuditAgent(codex)

    # Define magic protocols
    magic_protocols = [
        {
            "id": "magic:auth-middleware",
            "name": "AuthenticationMiddleware",
            "type": "middleware",
            "provides": {"current_user": {"type": "User | None"}},
            "requires": {"env_vars": ["JWT_SECRET"]},
        },
        {
            "id": "magic:db-session",
            "name": "DatabaseSession",
            "type": "di",
            "provides": {"db": {"type": "AsyncSession"}},
            "requires": {"env_vars": ["DATABASE_URL"]},
        },
    ]

    # Audit all files
    results = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Auditing files...", total=len(SAMPLE_PROJECT_FILES))

        for path, code in SAMPLE_PROJECT_FILES.items():
            result = await agent.audit_code(code, path, magic_protocols)
            results.append(result)
            progress.advance(task)

    # Summary
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    total_violations = sum(len(r.violations) for r in results)

    summary_table = Table(title="Audit Summary")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")

    summary_table.add_row("Files Audited", str(len(results)))
    summary_table.add_row("Passed", f"[green]{passed}[/green]")
    summary_table.add_row("Failed", f"[red]{failed}[/red]")
    summary_table.add_row("Total Violations", str(total_violations))

    console.print(summary_table)

    # Show per-file results
    console.print("\n[bold]Per-file Results:[/bold]")
    for result in results:
        status = "[green]PASS[/green]" if result.passed else "[red]FAIL[/red]"
        violations = len(result.violations)
        console.print(f"  {status} {result.file_path} ({violations} violations)")

    # Generate report
    console.print("\n[bold]Generating Report...[/bold]")
    report = agent.generate_report(results, ReportFormat.MARKDOWN)
    console.print(Panel(report[:2000] + "...", title="Markdown Report (truncated)"))

    # Get fix suggestions
    fixer = AutoFixSuggester()
    all_suggestions = []
    for result in results:
        suggestions = fixer.suggest_fixes(result)
        all_suggestions.extend(suggestions)

    if all_suggestions:
        console.print(f"\n[bold]Fix Suggestions ({len(all_suggestions)} total):[/bold]")
        fix_table = Table()
        fix_table.add_column("File", style="cyan")
        fix_table.add_column("Type", style="green")
        fix_table.add_column("Description", style="yellow")
        fix_table.add_column("Auto", style="magenta")

        for s in all_suggestions[:5]:
            fix_table.add_row(
                s.violation.file_path.split("/")[-1],
                s.fix_type.value,
                s.description[:40] + "..." if len(s.description) > 40 else s.description,
                "Yes" if s.auto_applicable else "No",
            )

        console.print(fix_table)


def demo_performance_benchmarks():
    """Demonstrate performance benchmarks."""
    console.print("\n[bold cyan]═══ Performance Benchmarks Demo ═══[/bold cyan]\n")

    def measure_time(name: str, func, iterations: int = 100, **kwargs) -> dict:
        """Measure execution time."""
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            func(**kwargs)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        return {
            "name": name,
            "iterations": iterations,
            "avg_ms": sum(times) / iterations * 1000,
            "min_ms": min(times) * 1000,
            "max_ms": max(times) * 1000,
            "ops_per_sec": iterations / sum(times),
        }

    benchmarks = []

    # Benchmark 1: Context Compression
    console.print("[bold]1. Context Compression[/bold]")
    manager = ContextWindowManager(max_tokens=10000)
    items = [
        ContextItem(
            id=f"item-{i}",
            content="x" * 500,
            category="generated_code",
            priority=50,
            tokens=125,
        )
        for i in range(100)
    ]
    result = measure_time("Context Compression", manager.compress_context, items=items)
    benchmarks.append(result)
    console.print(f"   Avg: {result['avg_ms']:.2f}ms, Ops/sec: {result['ops_per_sec']:.0f}")

    # Benchmark 2: Change Detection
    console.print("[bold]2. Change Detection[/bold]")
    detector = ChangeDetector()
    files = {f"file_{i}.py": f"content_{i}" for i in range(50)}
    detector.detect_changes(files)  # Prime cache
    result = measure_time("Change Detection", detector.detect_changes, files=files)
    benchmarks.append(result)
    console.print(f"   Avg: {result['avg_ms']:.2f}ms, Ops/sec: {result['ops_per_sec']:.0f}")

    # Benchmark 3: Code Analysis
    console.print("[bold]3. Code Analysis[/bold]")
    analyzer = CodeAnalyzer()
    sample_code = SAMPLE_PROJECT_FILES["src/api/routes/admin.py"]
    result = measure_time("Code Analysis", analyzer.analyze_code, code=sample_code, file_path="test.py")
    benchmarks.append(result)
    console.print(f"   Avg: {result['avg_ms']:.2f}ms, Ops/sec: {result['ops_per_sec']:.0f}")

    # Benchmark 4: Violation Detection
    console.print("[bold]4. Violation Detection[/bold]")
    codex = CodexBuilder().add_standard_rules().build(freeze=True)
    detector = ViolationDetector(codex)
    context = DetectionContext(declared_env_vars=["DATABASE_URL", "JWT_SECRET"])
    result = measure_time(
        "Violation Detection",
        detector.detect,
        iterations=50,
        code=sample_code,
        file_path="test.py",
        context=context,
    )
    benchmarks.append(result)
    console.print(f"   Avg: {result['avg_ms']:.2f}ms, Ops/sec: {result['ops_per_sec']:.0f}")

    # Summary table
    console.print("\n[bold]Benchmark Summary:[/bold]")
    bench_table = Table()
    bench_table.add_column("Benchmark", style="cyan")
    bench_table.add_column("Avg (ms)", style="green", justify="right")
    bench_table.add_column("Min (ms)", style="yellow", justify="right")
    bench_table.add_column("Max (ms)", style="red", justify="right")
    bench_table.add_column("Ops/sec", style="magenta", justify="right")

    for b in benchmarks:
        bench_table.add_row(
            b["name"],
            f"{b['avg_ms']:.2f}",
            f"{b['min_ms']:.2f}",
            f"{b['max_ms']:.2f}",
            f"{b['ops_per_sec']:.0f}",
        )

    console.print(bench_table)


async def run_demo():
    """Run the complete Phase 5 demo."""
    console.print(Panel.fit(
        "[bold magenta]Graph-Driven DFS Code Agent Architecture[/bold magenta]\n"
        "[cyan]Phase 5: Integration & Optimization Demo[/cyan]",
        border_style="bright_blue",
    ))

    try:
        # 1. Context Window Manager
        demo_context_compression()

        # 2. Incremental Updates
        await demo_incremental_updates()

        # 3. End-to-End Pipeline
        await demo_end_to_end_pipeline()

        # 4. Performance Benchmarks
        demo_performance_benchmarks()

        console.print(Panel.fit(
            "[bold green]Phase 5 Demo Complete![/bold green]\n\n"
            "Components demonstrated:\n"
            "  Context Window Manager - intelligent compression\n"
            "  Incremental Update Pipeline - minimal regeneration\n"
            "  End-to-End Pipeline - complete audit workflow\n"
            "  Performance Benchmarks - throughput measurements\n\n"
            "[dim]Run tests with: pytest tests/ -v[/dim]",
            border_style="green",
        ))

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(run_demo())
