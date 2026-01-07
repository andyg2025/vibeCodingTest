"""Audit Agent - orchestrates the complete audit process.

The Audit Agent:
1. Loads the Codex rules
2. Analyzes code files
3. Detects violations
4. Generates reports
5. Suggests fixes
"""

from pathlib import Path
from typing import Any

import structlog
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

from src.agents.base import AgentConfig, AgentState, AgentType, BaseAgent
from src.agents.codex import Codex
from src.audit.analyzer import CodeAnalyzer
from src.audit.detector import ViolationDetector, DetectionContext, DetectionResult
from src.audit.rules import RuleEngine
from src.audit.reporter import ReportGenerator, ReportFormat
from src.audit.fixer import AutoFixSuggester, FixSuggestion

logger = structlog.get_logger()


AUDIT_SYSTEM_PROMPT = """You are the Audit Agent in a graph-driven code generation system.

Your responsibilities:
1. Analyze generated code for compliance with magic dependency rules
2. Detect violations of the Dependency Codex
3. Ensure all implicit dependencies are properly declared
4. Verify lifecycle and side effect constraints
5. Suggest fixes for detected violations

You work with:
- CodeAnalyzer: Uses tree-sitter for AST analysis
- RuleEngine: Executes Codex rules against code
- ViolationDetector: Combines analysis and rules
- AutoFixSuggester: Generates fix suggestions

When reviewing code, focus on:
- Undeclared environment variables
- Missing authentication dependencies
- Improper database session usage
- Lifecycle ordering issues
- Side effect declarations

Provide clear, actionable feedback on violations."""


class AuditAgent(BaseAgent):
    """Agent that audits generated code against the Codex.

    Orchestrates the audit pipeline:
    1. Load Codex rules
    2. Analyze files with tree-sitter
    3. Execute rules with RuleEngine
    4. Detect violations
    5. Generate reports
    6. Suggest fixes
    """

    def __init__(self, codex: Codex | None = None):
        config = AgentConfig(
            name="AuditAgent",
            agent_type=AgentType.AUDIT,
            system_prompt=AUDIT_SYSTEM_PROMPT,
            temperature=0.0,
        )
        super().__init__(config)

        self.codex = codex
        self.analyzer = CodeAnalyzer()
        self.detector = ViolationDetector(codex)
        self.reporter = ReportGenerator()
        self.fixer = AutoFixSuggester()

    def get_system_prompt(self) -> str:
        return AUDIT_SYSTEM_PROMPT

    async def process(self, state: AgentState) -> AgentState:
        """Process audit request from workflow state."""
        context = state.get("context", {})

        # Get files to audit
        files_to_audit = context.get("files_to_audit", [])
        generated_files = context.get("generated_files", {})

        if not files_to_audit and not generated_files:
            return {
                **state,
                "errors": state.get("errors", []) + ["No files to audit"],
            }

        # Build detection context from magic protocols
        magic_protocols = context.get("magic_protocols", [])
        detection_context = DetectionContext.from_magic_protocols(magic_protocols)

        # Audit each file
        results = []

        # Audit generated files (in-memory)
        for file_id, code in generated_files.items():
            result = self.detector.detect(code, file_id, detection_context)
            results.append(result)

        # Audit file paths
        for file_path in files_to_audit:
            result = self.detector.detect_file(file_path, detection_context)
            results.append(result)

        # Check overall pass/fail
        all_passed = all(r.passed for r in results)
        total_violations = sum(len(r.violations) for r in results)

        # Generate fix suggestions
        all_fixes: list[FixSuggestion] = []
        for result in results:
            fixes = self.fixer.suggest_fixes(result)
            all_fixes.extend(fixes)

        # Update state
        return {
            **state,
            "context": {
                **context,
                "audit_results": [self._result_to_dict(r) for r in results],
                "audit_passed": all_passed,
                "total_violations": total_violations,
                "fix_suggestions": [self._fix_to_dict(f) for f in all_fixes],
            },
            "audit_passed": all_passed,
            "violations": [
                v.to_dict()
                for r in results
                for v in r.violations
            ],
        }

    async def audit_code(
        self,
        code: str,
        file_path: str = "<generated>",
        magic_protocols: list[dict[str, Any]] | None = None,
    ) -> DetectionResult:
        """Audit a single piece of code.

        Args:
            code: Python source code
            file_path: Path for reporting
            magic_protocols: Magic protocols affecting this code

        Returns:
            DetectionResult with violations
        """
        context = DetectionContext.from_magic_protocols(magic_protocols or [])
        return self.detector.detect(code, file_path, context)

    async def audit_files(
        self,
        file_paths: list[str | Path],
        magic_protocols: list[dict[str, Any]] | None = None,
    ) -> list[DetectionResult]:
        """Audit multiple files.

        Args:
            file_paths: Paths to Python files
            magic_protocols: Magic protocols affecting these files

        Returns:
            List of detection results
        """
        context = DetectionContext.from_magic_protocols(magic_protocols or [])
        return [
            self.detector.detect_file(f, context)
            for f in file_paths
        ]

    async def audit_project(
        self,
        root_path: str | Path,
        magic_protocols: list[dict[str, Any]] | None = None,
        exclude_patterns: list[str] | None = None,
    ) -> list[DetectionResult]:
        """Audit all Python files in a project.

        Args:
            root_path: Project root directory
            magic_protocols: Magic protocols for the project
            exclude_patterns: Glob patterns to exclude

        Returns:
            List of detection results
        """
        root = Path(root_path)
        exclude_patterns = exclude_patterns or ["**/test_*", "**/__pycache__/*", "**/venv/*"]

        # Find all Python files
        python_files = list(root.rglob("*.py"))

        # Apply exclusions
        for pattern in exclude_patterns:
            excluded = set(root.glob(pattern))
            python_files = [f for f in python_files if f not in excluded]

        await self._logger.ainfo(
            "Auditing project",
            root=str(root),
            file_count=len(python_files),
        )

        return await self.audit_files(python_files, magic_protocols)

    def generate_report(
        self,
        results: list[DetectionResult],
        format: ReportFormat = ReportFormat.TEXT,
        output_path: str | Path | None = None,
    ) -> str:
        """Generate an audit report.

        Args:
            results: Detection results
            format: Output format
            output_path: Optional path to save report

        Returns:
            Formatted report string
        """
        report = self.reporter.generate(results, format)

        if output_path:
            Path(output_path).write_text(report)

        return report

    def suggest_fixes(
        self, results: list[DetectionResult]
    ) -> dict[str, list[FixSuggestion]]:
        """Get fix suggestions for all results."""
        return self.fixer.suggest_fixes_batch(results)

    def create_graph(self) -> StateGraph:
        """Create a LangGraph workflow for the audit agent."""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("analyze", self._analyze_node)
        workflow.add_node("detect", self._detect_node)
        workflow.add_node("report", self._report_node)

        # Set entry point
        workflow.set_entry_point("analyze")

        # Add edges
        workflow.add_edge("analyze", "detect")
        workflow.add_edge("detect", "report")
        workflow.add_edge("report", END)

        return workflow

    async def _analyze_node(self, state: AgentState) -> AgentState:
        """Analyze code files."""
        context = state.get("context", {})
        generated_files = context.get("generated_files", {})

        analyses = {}
        for file_id, code in generated_files.items():
            analysis = self.analyzer.analyze_code(code, file_id)
            analyses[file_id] = {
                "imports": len(analysis.imports),
                "env_accesses": len(analysis.env_accesses),
                "decorators": len(analysis.decorators),
                "dependencies": len(analysis.dependencies),
            }

        return {
            **state,
            "context": {
                **context,
                "analyses": analyses,
            },
        }

    async def _detect_node(self, state: AgentState) -> AgentState:
        """Detect violations."""
        return await self.process(state)

    async def _report_node(self, state: AgentState) -> AgentState:
        """Generate report."""
        context = state.get("context", {})
        audit_results = context.get("audit_results", [])

        # Generate summary for LLM
        summary_lines = ["Audit Summary:"]
        for result in audit_results:
            status = "PASS" if result.get("passed", True) else "FAIL"
            violations = len(result.get("violations", []))
            summary_lines.append(f"  {result.get('file_path', 'unknown')}: {status} ({violations} violations)")

        summary = "\n".join(summary_lines)

        # Add to messages
        messages = state.get("messages", [])
        messages.append(AIMessage(content=summary))

        return {
            **state,
            "messages": messages,
            "context": {
                **context,
                "audit_summary": summary,
            },
        }

    def _result_to_dict(self, result: DetectionResult) -> dict[str, Any]:
        """Convert detection result to dict."""
        return {
            "file_path": result.file_path,
            "passed": result.passed,
            "violations": [v.to_dict() for v in result.violations],
            "violation_count": len(result.violations),
            "strict_count": len(result.strict_violations),
            "warning_count": len(result.warnings),
        }

    def _fix_to_dict(self, fix: FixSuggestion) -> dict[str, Any]:
        """Convert fix suggestion to dict."""
        return {
            "rule_id": fix.violation.rule_id,
            "fix_type": fix.fix_type.value,
            "complexity": fix.complexity.value,
            "description": fix.description,
            "auto_applicable": fix.auto_applicable,
            "confidence": fix.confidence,
        }
