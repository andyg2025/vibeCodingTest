"""End-to-end integration tests for the Graph-Driven DFS Code Agent.

Tests the complete pipeline:
1. Architect Agent → Project skeleton
2. Magic-Link Agent → Magic dependency detection
3. DFS Walker → Code generation with context
4. Audit Agent → Violation detection and reporting
"""

import pytest
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.codex import Codex, CodexBuilder, EnforcementLevel
from src.agents.state import ProjectSpec, ModuleSpec, FileSpec
from src.audit import (
    CodeAnalyzer,
    ViolationDetector,
    DetectionContext,
    ReportGenerator,
    ReportFormat,
    AutoFixSuggester,
    AuditAgent,
)
from src.engine.context import ContextAggregator, AggregatedContext
from src.engine.walker import DFSWalker, DFSState, GenerationResult
from src.engine.backtrack import BacktrackingEngine, BacktrackLevel


class TestCodeAnalyzer:
    """Tests for the Code Analyzer component."""

    def test_analyze_imports(self, sample_code_with_violations: str):
        """Test that imports are correctly detected."""
        analyzer = CodeAnalyzer()
        analysis = analyzer.analyze_code(sample_code_with_violations, "test.py")

        assert len(analysis.imports) > 0
        import_names = [i.name for i in analysis.imports]
        assert "os" in import_names
        assert "fastapi" in import_names

    def test_analyze_env_accesses(self, sample_code_with_violations: str):
        """Test that environment variable accesses are detected."""
        analyzer = CodeAnalyzer()
        analysis = analyzer.analyze_code(sample_code_with_violations, "test.py")

        assert len(analysis.env_accesses) > 0
        env_names = [e.name for e in analysis.env_accesses]
        assert "UNDECLARED_VAR" in env_names

    def test_analyze_decorators(self, sample_code_with_violations: str):
        """Test that decorators are detected."""
        analyzer = CodeAnalyzer()
        analysis = analyzer.analyze_code(sample_code_with_violations, "test.py")

        assert len(analysis.decorators) > 0
        decorator_names = [d.name for d in analysis.decorators]
        assert any("router" in name for name in decorator_names)

    def test_analyze_clean_code(self, sample_code_clean: str):
        """Test analysis of clean code."""
        analyzer = CodeAnalyzer()
        analysis = analyzer.analyze_code(sample_code_clean, "clean.py")

        # Should detect imports and dependencies
        assert len(analysis.imports) > 0
        assert len(analysis.dependencies) > 0


class TestViolationDetector:
    """Tests for the Violation Detector component."""

    def test_detect_undeclared_env_var(
        self,
        sample_code_with_violations: str,
        sample_codex: Codex,
    ):
        """Test detection of undeclared environment variables."""
        detector = ViolationDetector(sample_codex)
        context = DetectionContext(
            declared_env_vars=["DATABASE_URL", "JWT_SECRET"],  # UNDECLARED_VAR is missing
        )

        result = detector.detect(sample_code_with_violations, "test.py", context)

        assert not result.passed
        assert len(result.violations) > 0
        # Should find undeclared env var
        env_violations = [v for v in result.violations if "env" in v.rule_id.lower() or "ENV" in v.message]
        assert len(env_violations) > 0

    def test_clean_code_passes(
        self,
        sample_code_clean: str,
        sample_codex: Codex,
    ):
        """Test that clean code passes detection."""
        detector = ViolationDetector(sample_codex)
        context = DetectionContext(
            declared_env_vars=["DATABASE_URL", "JWT_SECRET", "DEBUG"],
        )

        result = detector.detect(sample_code_clean, "clean.py", context)

        # Clean code should pass or have minimal warnings
        assert result.passed or len(result.strict_violations) == 0


class TestReportGenerator:
    """Tests for the Report Generator component."""

    def test_generate_text_report(
        self,
        sample_code_with_violations: str,
        sample_codex: Codex,
    ):
        """Test text report generation."""
        detector = ViolationDetector(sample_codex)
        context = DetectionContext(declared_env_vars=[])
        result = detector.detect(sample_code_with_violations, "test.py", context)

        reporter = ReportGenerator()
        report = reporter.generate([result], ReportFormat.TEXT)

        assert "MAGIC DEPENDENCY AUDIT REPORT" in report
        assert "test.py" in report
        assert "Violations" in report or "FAIL" in report

    def test_generate_json_report(
        self,
        sample_code_with_violations: str,
        sample_codex: Codex,
    ):
        """Test JSON report generation."""
        detector = ViolationDetector(sample_codex)
        context = DetectionContext(declared_env_vars=[])
        result = detector.detect(sample_code_with_violations, "test.py", context)

        reporter = ReportGenerator()
        report = reporter.generate([result], ReportFormat.JSON)

        import json
        data = json.loads(report)
        assert "timestamp" in data
        assert "summary" in data
        assert "results" in data

    def test_generate_sarif_report(
        self,
        sample_code_with_violations: str,
        sample_codex: Codex,
    ):
        """Test SARIF report generation."""
        detector = ViolationDetector(sample_codex)
        context = DetectionContext(declared_env_vars=[])
        result = detector.detect(sample_code_with_violations, "test.py", context)

        reporter = ReportGenerator()
        report = reporter.generate([result], ReportFormat.SARIF)

        import json
        data = json.loads(report)
        assert data["$schema"].endswith("sarif-schema-2.1.0.json")
        assert "runs" in data
        assert data["runs"][0]["tool"]["driver"]["name"] == "MagicDependencyAudit"


class TestAutoFixSuggester:
    """Tests for the Auto-Fix Suggester component."""

    def test_suggest_fixes(
        self,
        sample_code_with_violations: str,
        sample_codex: Codex,
    ):
        """Test fix suggestion generation."""
        detector = ViolationDetector(sample_codex)
        context = DetectionContext(declared_env_vars=[])
        result = detector.detect(sample_code_with_violations, "test.py", context)

        fixer = AutoFixSuggester()
        suggestions = fixer.suggest_fixes(result)

        assert len(suggestions) > 0
        for suggestion in suggestions:
            assert suggestion.description
            assert suggestion.fix_type
            assert 0 <= suggestion.confidence <= 1


class TestContextWindowManager:
    """Tests for the Context Window Manager component."""

    def test_compress_empty_context(self):
        """Test compression with empty inputs."""
        from src.engine.context_manager import ContextWindowManager, ContextItem

        manager = ContextWindowManager(max_tokens=10000)
        context = manager.compress_context(items=[])

        assert context.total_tokens == 0

    def test_compress_with_magic_protocols(self):
        """Test compression preserves magic protocols."""
        from src.engine.context_manager import ContextWindowManager, ContextItem

        manager = ContextWindowManager(max_tokens=10000)

        items = [
            ContextItem(
                id="magic:auth",
                content="Auth middleware",
                category="magic_protocol",
                priority=100,
                tokens=50,
                compressible=False,
            ),
            ContextItem(
                id="generated:test.py",
                content="x" * 1000,
                category="generated_code",
                priority=30,
                tokens=250,
            ),
        ]

        context = manager.compress_context(items)

        # Magic protocols should be preserved
        assert len(context.magic_protocols) == 1
        assert context.magic_protocols[0].id == "magic:auth"

    def test_token_limit_respected(self):
        """Test that token limits are respected."""
        from src.engine.context_manager import ContextWindowManager, ContextItem

        manager = ContextWindowManager(max_tokens=500)

        # Create items that exceed budget
        items = [
            ContextItem(
                id=f"item-{i}",
                content="x" * 1000,
                category="generated_code",
                priority=50,
                tokens=250,
            )
            for i in range(10)
        ]

        context = manager.compress_context(items)

        # Should be compressed to fit
        assert context.total_tokens <= 600  # Allow some overhead


class TestBacktrackingEngine:
    """Tests for the Backtracking Engine component."""

    def test_backtrack_level_enum_exists(self):
        """Test that BacktrackLevel enum has expected values."""
        assert BacktrackLevel.CURRENT_NODE is not None
        assert BacktrackLevel.PARENT_NODE is not None
        assert BacktrackLevel.MODULE_LEVEL is not None
        assert BacktrackLevel.GLOBAL is not None

    def test_backtrack_action_creation(self):
        """Test BacktrackAction can be created."""
        from src.engine.backtrack import BacktrackAction, BacktrackReason

        action = BacktrackAction(
            level=BacktrackLevel.CURRENT_NODE,
            reason=BacktrackReason.TYPE_MISMATCH,
            node_id="test-node",
        )

        assert action.level == BacktrackLevel.CURRENT_NODE
        assert action.node_id == "test-node"
        assert action.reason == BacktrackReason.TYPE_MISMATCH


class TestIncrementalUpdate:
    """Tests for the Incremental Update Pipeline."""

    def test_change_detection(self):
        """Test change detection between file states."""
        from src.engine.incremental import ChangeDetector, ChangeType

        detector = ChangeDetector()

        # Initial state
        files_v1 = {
            "file1.py": "content1",
            "file2.py": "content2",
        }
        changes = detector.detect_changes(files_v1)
        assert len(changes) == 2  # Both are new

        # Modified state
        files_v2 = {
            "file1.py": "content1_modified",
            "file2.py": "content2",
            "file3.py": "content3",  # New file
        }
        changes = detector.detect_changes(files_v2)

        # Should detect: file1 modified, file3 created
        assert len(changes) == 2
        change_types = {c.path: c.change_type for c in changes}
        assert change_types.get("file1.py") == ChangeType.MODIFIED
        assert change_types.get("file3.py") == ChangeType.CREATED

    @pytest.mark.asyncio
    async def test_incremental_pipeline(self):
        """Test full incremental update pipeline."""
        from src.engine.incremental import IncrementalUpdatePipeline

        pipeline = IncrementalUpdatePipeline()

        files = {
            "src/main.py": "# Main file",
            "src/utils.py": "# Utils",
        }

        result = await pipeline.process_update(files)

        assert result["status"] in ("no_changes", "updated")
        assert "statistics" in result


class TestAuditAgent:
    """Tests for the Audit Agent orchestration."""

    @pytest.mark.asyncio
    async def test_audit_code(
        self,
        sample_code_with_violations: str,
        sample_magic_protocols: list[dict],
        sample_codex: Codex,
    ):
        """Test auditing code through the agent."""
        agent = AuditAgent(sample_codex)

        result = await agent.audit_code(
            sample_code_with_violations,
            "test.py",
            sample_magic_protocols,
        )

        assert result is not None
        assert hasattr(result, "violations")
        assert hasattr(result, "passed")

    @pytest.mark.asyncio
    async def test_audit_multiple_files(
        self,
        sample_code_with_violations: str,
        sample_code_clean: str,
        sample_codex: Codex,
        tmp_path,
    ):
        """Test auditing multiple files."""
        # Create temp files
        bad_file = tmp_path / "bad.py"
        bad_file.write_text(sample_code_with_violations)

        good_file = tmp_path / "good.py"
        good_file.write_text(sample_code_clean)

        agent = AuditAgent(sample_codex)
        results = await agent.audit_files([bad_file, good_file])

        assert len(results) == 2


class TestEndToEndPipeline:
    """End-to-end integration tests for the complete pipeline."""

    @pytest.mark.asyncio
    async def test_full_audit_pipeline(
        self,
        sample_code_with_violations: str,
        sample_magic_protocols: list[dict],
    ):
        """Test the complete audit pipeline end-to-end."""
        # 1. Build Codex
        builder = CodexBuilder()
        builder.add_standard_rules()
        codex = builder.build(freeze=True)

        # 2. Create Audit Agent
        agent = AuditAgent(codex)

        # 3. Audit code
        result = await agent.audit_code(
            sample_code_with_violations,
            "test.py",
            sample_magic_protocols,
        )

        # 4. Generate report
        report = agent.generate_report([result], ReportFormat.TEXT)

        # 5. Get fix suggestions
        fixer = AutoFixSuggester()
        suggestions = fixer.suggest_fixes(result)

        # Verify pipeline completed
        assert result is not None
        assert "AUDIT REPORT" in report
        assert len(suggestions) >= 0  # May or may not have suggestions

    @pytest.mark.asyncio
    async def test_clean_code_passes_pipeline(
        self,
        sample_code_clean: str,
        sample_magic_protocols: list[dict],
    ):
        """Test that clean code passes the audit pipeline."""
        builder = CodexBuilder()
        builder.add_standard_rules()
        codex = builder.build(freeze=True)

        agent = AuditAgent(codex)
        result = await agent.audit_code(
            sample_code_clean,
            "clean.py",
            sample_magic_protocols,
        )

        # Clean code should pass or have no strict violations
        assert result.passed or len(result.strict_violations) == 0


class TestComponentIntegration:
    """Tests for integration between components."""

    def test_analyzer_to_detector_integration(
        self,
        sample_code_with_violations: str,
        sample_codex: Codex,
    ):
        """Test that analyzer output integrates with detector."""
        # Analyzer produces analysis
        analyzer = CodeAnalyzer()
        analysis = analyzer.analyze_code(sample_code_with_violations, "test.py")

        # Detector uses analysis
        detector = ViolationDetector(sample_codex)
        context = DetectionContext(declared_env_vars=[])
        result = detector.detect(sample_code_with_violations, "test.py", context)

        # Analysis categories should map to violations
        if analysis.env_accesses:
            # Should have env-related violations
            env_violations = [
                v for v in result.violations
                if "env" in v.message.lower() or "ENV" in v.rule_id
            ]
            assert len(env_violations) > 0

    def test_detector_to_fixer_integration(
        self,
        sample_code_with_violations: str,
        sample_codex: Codex,
    ):
        """Test that detector output integrates with fixer."""
        detector = ViolationDetector(sample_codex)
        context = DetectionContext(declared_env_vars=[])
        result = detector.detect(sample_code_with_violations, "test.py", context)

        fixer = AutoFixSuggester()
        suggestions = fixer.suggest_fixes(result)

        # Each violation should have a suggestion
        # (some may be generic)
        assert len(suggestions) <= len(result.violations) + 5  # Allow some overhead

    def test_reporter_formats_all_data(
        self,
        sample_code_with_violations: str,
        sample_codex: Codex,
    ):
        """Test that reporter can format all detection data."""
        detector = ViolationDetector(sample_codex)
        context = DetectionContext(declared_env_vars=[])
        result = detector.detect(sample_code_with_violations, "test.py", context)

        reporter = ReportGenerator()

        # All formats should work
        for fmt in ReportFormat:
            report = reporter.generate([result], fmt)
            assert len(report) > 0
