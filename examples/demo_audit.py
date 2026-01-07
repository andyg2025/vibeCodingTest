"""Demo script for Phase 4: Magic Dependency Audit System.

This demonstrates:
1. Code Analyzer - tree-sitter based AST analysis
2. Rule Engine - executing Codex rules
3. Violation Detector - finding issues
4. Report Generator - multiple formats
5. Auto-Fix Suggester - generating fixes
6. Audit Agent - orchestrating the pipeline

Usage:
    python examples/demo_audit.py
"""

import asyncio

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from src.agents.codex import Codex, CodexBuilder, CodexRule, EnforcementLevel
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


# Sample code with various violations
SAMPLE_CODE_WITH_VIOLATIONS = '''"""User routes with intentional violations."""
import os
from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session, sessionmaker

router = APIRouter(prefix="/admin", tags=["admin"])

# Violation: Direct env var access without declaration
SECRET_KEY = os.getenv("UNDECLARED_SECRET")

# Violation: Direct session creation (should use DI)
engine = create_engine(os.environ["DATABASE_URL"])
SessionLocal = sessionmaker(bind=engine)


@router.get("/users")
async def get_users(
    db: Annotated[Session, Depends(SessionLocal)],
):
    """Get all users - MISSING AUTH DEPENDENCY."""
    return db.query(User).all()


@router.delete("/users/{user_id}")
async def delete_user(
    user_id: int,
    db: Session = Depends(SessionLocal),
):
    """Delete user - MISSING AUTH, direct session creation."""
    # Violation: Side effect without declaration
    db.delete(db.query(User).get(user_id))
    db.commit()
    return {"deleted": user_id}
'''

SAMPLE_CODE_CLEAN = '''"""User routes following best practices."""
from typing import Annotated
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import get_settings
from src.api.deps import get_current_user, get_db

router = APIRouter(prefix="/users", tags=["users"])


@router.get("/me")
async def get_current_user_profile(
    current_user: Annotated[dict, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Get the current user's profile."""
    return {"user": current_user}
'''


def build_demo_codex() -> Codex:
    """Build a Codex with demo rules."""
    builder = CodexBuilder()

    # Add standard rules
    builder.add_standard_rules()

    # Add custom rules
    builder._codex.add_rule(CodexRule(
        rule_id="DEMO-001",
        description="All routes under /admin must have authentication",
        enforcement_level=EnforcementLevel.STRICT,
        fix_suggestion="Add 'current_user: User = Depends(get_current_user)' parameter",
    ))

    return builder.build(freeze=True)


def demo_code_analyzer():
    """Demonstrate the Code Analyzer."""
    console.print("\n[bold cyan]═══ Code Analyzer Demo ═══[/bold cyan]\n")

    analyzer = CodeAnalyzer()
    analysis = analyzer.analyze_code(SAMPLE_CODE_WITH_VIOLATIONS, "demo_violations.py")

    # Show analysis results
    table = Table(title="Code Analysis Results")
    table.add_column("Category", style="cyan")
    table.add_column("Count", style="green")
    table.add_column("Details", style="yellow")

    table.add_row(
        "Imports",
        str(len(analysis.imports)),
        ", ".join(i.name for i in analysis.imports[:3]) + "...",
    )
    table.add_row(
        "Env Accesses",
        str(len(analysis.env_accesses)),
        ", ".join(e.name for e in analysis.env_accesses),
    )
    table.add_row(
        "Decorators",
        str(len(analysis.decorators)),
        ", ".join(d.name.split("(")[0] for d in analysis.decorators),
    )
    table.add_row(
        "Dependencies",
        str(len(analysis.dependencies)),
        ", ".join(d.name for d in analysis.dependencies),
    )
    table.add_row(
        "Function Defs",
        str(len(analysis.get_by_category(analyzer.analyzer.AnalysisCategory.FUNCTION_DEFINITIONS if hasattr(analyzer, 'analyzer') else analysis.results[0].category))),
        "",
    )

    console.print(table)

    # Show env var details
    if analysis.env_accesses:
        console.print("\n[bold]Environment Variable Accesses:[/bold]")
        for env in analysis.env_accesses:
            console.print(f"  • {env.name} at line {env.location.line}: {env.raw_text}")


def demo_violation_detector():
    """Demonstrate the Violation Detector."""
    console.print("\n[bold cyan]═══ Violation Detector Demo ═══[/bold cyan]\n")

    codex = build_demo_codex()
    detector = ViolationDetector(codex)

    # Create context with declared env vars
    context = DetectionContext(
        declared_env_vars=["DATABASE_URL", "JWT_SECRET"],  # UNDECLARED_SECRET is missing
        protected_patterns=["/admin"],
    )

    # Detect violations
    result = detector.detect(SAMPLE_CODE_WITH_VIOLATIONS, "demo_violations.py", context)

    # Show results
    status = "[green]PASSED[/green]" if result.passed else "[red]FAILED[/red]"
    console.print(f"Detection Status: {status}")
    console.print(f"Total Violations: {len(result.violations)}")
    console.print(f"  Strict: {len(result.strict_violations)}")
    console.print(f"  Warnings: {len(result.warnings)}")

    # Show violations
    if result.violations:
        console.print("\n[bold]Violations Found:[/bold]")
        for v in result.violations:
            severity = "🔴" if v.severity == EnforcementLevel.STRICT else "🟡"
            console.print(f"\n{severity} [{v.rule_id}] {v.message}")
            if v.line_number:
                console.print(f"   Line {v.line_number}: {v.code_snippet[:50]}...")
            if v.fix_suggestion:
                console.print(f"   [dim]Fix: {v.fix_suggestion}[/dim]")

    return result


def demo_report_generator(result):
    """Demonstrate the Report Generator."""
    console.print("\n[bold cyan]═══ Report Generator Demo ═══[/bold cyan]\n")

    reporter = ReportGenerator()

    # Generate different formats
    console.print("[bold]Text Report (excerpt):[/bold]")
    text_report = reporter.generate([result], ReportFormat.TEXT)
    console.print(Panel(text_report[:1000] + "...", title="Text Report"))

    console.print("\n[bold]Markdown Report (excerpt):[/bold]")
    md_report = reporter.generate([result], ReportFormat.MARKDOWN)
    console.print(Panel(md_report[:800] + "...", title="Markdown Report"))


def demo_auto_fixer(result):
    """Demonstrate the Auto-Fix Suggester."""
    console.print("\n[bold cyan]═══ Auto-Fix Suggester Demo ═══[/bold cyan]\n")

    fixer = AutoFixSuggester()
    suggestions = fixer.suggest_fixes(result)

    console.print(f"Generated {len(suggestions)} fix suggestions:\n")

    table = Table(title="Fix Suggestions")
    table.add_column("Rule", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("Complexity", style="yellow")
    table.add_column("Description", style="white")
    table.add_column("Auto", style="magenta")

    for s in suggestions:
        table.add_row(
            s.violation.rule_id,
            s.fix_type.value,
            s.complexity.value,
            s.description[:40] + "..." if len(s.description) > 40 else s.description,
            "✓" if s.auto_applicable else "✗",
        )

    console.print(table)

    # Show detailed fix
    if suggestions:
        console.print("\n[bold]Detailed Fix Suggestion:[/bold]")
        fix = suggestions[0]
        console.print(f"  Rule: {fix.violation.rule_id}")
        console.print(f"  Description: {fix.description}")
        console.print(f"  Confidence: {fix.confidence:.0%}")
        if fix.patches:
            console.print("  Patches:")
            for patch in fix.patches:
                console.print(f"    - {patch.description}")


async def demo_audit_agent():
    """Demonstrate the Audit Agent."""
    console.print("\n[bold cyan]═══ Audit Agent Demo ═══[/bold cyan]\n")

    codex = build_demo_codex()
    agent = AuditAgent(codex)

    # Audit both sample codes
    magic_protocols = [
        {
            "id": "magic_auth",
            "name": "AuthMiddleware",
            "provides": {"current_user": {"type": "User"}},
            "requires": {"env_vars": ["JWT_SECRET"]},
        },
        {
            "id": "magic_db",
            "name": "DatabaseSession",
            "provides": {"db": {"type": "AsyncSession"}},
            "requires": {"env_vars": ["DATABASE_URL"]},
        },
    ]

    # Audit code with violations
    console.print("[bold]Auditing code with violations...[/bold]")
    result_bad = await agent.audit_code(
        SAMPLE_CODE_WITH_VIOLATIONS,
        "violations.py",
        magic_protocols,
    )

    console.print(f"  Status: {'[green]PASS[/green]' if result_bad.passed else '[red]FAIL[/red]'}")
    console.print(f"  Violations: {len(result_bad.violations)}")

    # Audit clean code
    console.print("\n[bold]Auditing clean code...[/bold]")
    result_good = await agent.audit_code(
        SAMPLE_CODE_CLEAN,
        "clean.py",
        magic_protocols,
    )

    console.print(f"  Status: {'[green]PASS[/green]' if result_good.passed else '[red]FAIL[/red]'}")
    console.print(f"  Violations: {len(result_good.violations)}")

    # Generate combined report
    console.print("\n[bold]Combined Audit Report:[/bold]")
    report = agent.generate_report([result_bad, result_good], ReportFormat.TEXT)
    console.print(Panel(report, title="Audit Report"))


async def run_demo():
    """Run the complete Phase 4 demo."""
    console.print(Panel.fit(
        "[bold magenta]Graph-Driven DFS Code Agent Architecture[/bold magenta]\n"
        "[cyan]Phase 4: Magic Dependency Audit System Demo[/cyan]",
        border_style="bright_blue",
    ))

    try:
        # 1. Code Analyzer
        demo_code_analyzer()

        # 2. Violation Detector
        result = demo_violation_detector()

        # 3. Report Generator
        demo_report_generator(result)

        # 4. Auto-Fix Suggester
        demo_auto_fixer(result)

        # 5. Audit Agent
        await demo_audit_agent()

        console.print(Panel.fit(
            "[bold green]✓ Phase 4 Demo Complete![/bold green]\n\n"
            "Components demonstrated:\n"
            "• Code Analyzer - tree-sitter AST analysis\n"
            "• Violation Detector - rule-based detection\n"
            "• Report Generator - text, markdown, JSON, SARIF\n"
            "• Auto-Fix Suggester - patch generation\n"
            "• Audit Agent - orchestrated pipeline",
            border_style="green",
        ))

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(run_demo())
