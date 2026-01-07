"""Violation Detector - combines analysis and rule execution.

The Detector:
1. Analyzes code using the CodeAnalyzer
2. Executes rules using the RuleEngine
3. Produces structured violations
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from src.agents.codex import Codex, EnforcementLevel, Violation, ViolationType
from src.audit.analyzer import CodeAnalyzer, FileAnalysis
from src.audit.rules import RuleEngine, RuleMatch

logger = structlog.get_logger()


@dataclass
class DetectionContext:
    """Context for violation detection."""

    # Declared environment variables (from Settings class or magic protocols)
    declared_env_vars: list[str] = field(default_factory=list)

    # Declared side effects (from magic protocols)
    declared_side_effects: list[str] = field(default_factory=list)

    # Protected route patterns
    protected_patterns: list[str] = field(default_factory=lambda: ["/admin", "/user", "/protected"])

    # Files allowed to create DB sessions
    db_config_files: list[str] = field(default_factory=lambda: ["database.py", "config.py"])

    # Magic protocols affecting this file
    magic_protocols: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for rule engine."""
        return {
            "declared_env_vars": self.declared_env_vars,
            "declared_side_effects": self.declared_side_effects,
            "protected_patterns": self.protected_patterns,
            "db_config_files": self.db_config_files,
            "magic_protocols": self.magic_protocols,
        }

    @classmethod
    def from_magic_protocols(cls, protocols: list[dict[str, Any]]) -> "DetectionContext":
        """Build context from magic protocols."""
        ctx = cls()

        for protocol in protocols:
            ctx.magic_protocols.append(protocol)

            # Extract env vars from requires
            requires = protocol.get("requires", {})
            if isinstance(requires, dict):
                ctx.declared_env_vars.extend(requires.get("env_vars", []))

            # Extract side effects
            side_effects = protocol.get("side_effects", [])
            for se in side_effects:
                if isinstance(se, dict):
                    ctx.declared_side_effects.append(se.get("type", ""))

        return ctx


@dataclass
class DetectionResult:
    """Result of detecting violations in code."""

    file_path: str
    analysis: FileAnalysis
    rule_matches: list[RuleMatch]
    violations: list[Violation]

    @property
    def passed(self) -> bool:
        """True if no strict violations."""
        return not any(
            v.severity == EnforcementLevel.STRICT
            for v in self.violations
        )

    @property
    def strict_violations(self) -> list[Violation]:
        """Get only strict violations."""
        return [v for v in self.violations if v.severity == EnforcementLevel.STRICT]

    @property
    def warnings(self) -> list[Violation]:
        """Get warnings (non-strict violations)."""
        return [v for v in self.violations if v.severity == EnforcementLevel.WARN]


class ViolationDetector:
    """Detects code violations against the Codex rules.

    Combines the CodeAnalyzer and RuleEngine to produce
    actionable violation reports.
    """

    def __init__(self, codex: Codex | None = None):
        self.codex = codex
        self.analyzer = CodeAnalyzer()
        self.rule_engine = RuleEngine(codex)
        self._logger = logger.bind(component="ViolationDetector")

    def detect(
        self,
        code: str,
        file_path: str = "<string>",
        context: DetectionContext | None = None,
    ) -> DetectionResult:
        """Detect violations in Python code.

        Args:
            code: Python source code
            file_path: Path for reporting
            context: Detection context with declared vars, etc

        Returns:
            DetectionResult with analysis, matches, and violations
        """
        context = context or DetectionContext()

        self._logger.info("Detecting violations", file=file_path)

        # Step 1: Analyze code
        analysis = self.analyzer.analyze_code(code, file_path)

        if analysis.errors:
            self._logger.warning(
                "Analysis had errors",
                file=file_path,
                errors=analysis.errors,
            )

        # Step 2: Execute rules
        rule_matches = self.rule_engine.execute(analysis, context.to_dict())

        # Step 3: Convert failed matches to violations
        violations = self._matches_to_violations(rule_matches, file_path)

        # Step 4: Add any analysis-based violations (undeclared env vars, etc)
        violations.extend(self._detect_undeclared_env_vars(analysis, context))
        violations.extend(self._detect_missing_imports(analysis, context))

        result = DetectionResult(
            file_path=file_path,
            analysis=analysis,
            rule_matches=rule_matches,
            violations=violations,
        )

        self._logger.info(
            "Detection complete",
            file=file_path,
            violations=len(violations),
            passed=result.passed,
        )

        return result

    def detect_file(
        self,
        file_path: str | Path,
        context: DetectionContext | None = None,
    ) -> DetectionResult:
        """Detect violations in a Python file."""
        path = Path(file_path)
        if not path.exists():
            return DetectionResult(
                file_path=str(path),
                analysis=FileAnalysis(str(path), errors=[f"File not found: {path}"]),
                rule_matches=[],
                violations=[Violation(
                    rule_id="FILE-001",
                    violation_type=ViolationType.CRITICAL,
                    message=f"File not found: {path}",
                    file_path=str(path),
                )],
            )

        code = path.read_text()
        return self.detect(code, str(path), context)

    def detect_multiple(
        self,
        files: list[str | Path],
        context: DetectionContext | None = None,
    ) -> list[DetectionResult]:
        """Detect violations in multiple files."""
        return [self.detect_file(f, context) for f in files]

    def _matches_to_violations(
        self,
        matches: list[RuleMatch],
        file_path: str,
    ) -> list[Violation]:
        """Convert failed rule matches to violations."""
        violations = []

        for match in matches:
            if not match.matched:
                # Determine violation type from match
                violation_type = self._infer_violation_type(match)

                # Get location from analysis result
                line_number = None
                code_snippet = ""
                if match.analysis_result:
                    line_number = match.analysis_result.location.line
                    code_snippet = match.analysis_result.raw_text[:100]

                violations.append(Violation(
                    rule_id=match.rule.rule_id,
                    violation_type=violation_type,
                    message=match.message,
                    file_path=file_path,
                    line_number=line_number,
                    code_snippet=code_snippet,
                    fix_suggestion=match.rule.fix_suggestion,
                    severity=match.rule.enforcement_level,
                ))

        return violations

    def _infer_violation_type(self, match: RuleMatch) -> ViolationType:
        """Infer violation type from rule match."""
        match match.match_type.value:
            case "env_var_declared":
                return ViolationType.UNDECLARED_ENV_VAR
            case "dependency_declared":
                return ViolationType.MISSING_DEPENDENCY
            case "decorator_present":
                return ViolationType.MISSING_MAGIC_IMPORT
            case "side_effect_allowed":
                return ViolationType.SIDE_EFFECT_MISMATCH
            case "lifecycle_order":
                return ViolationType.LIFECYCLE_ORDER
            case _:
                return ViolationType.CONTRACT_VIOLATION

    def _detect_undeclared_env_vars(
        self,
        analysis: FileAnalysis,
        context: DetectionContext,
    ) -> list[Violation]:
        """Detect undeclared environment variable accesses."""
        violations = []
        declared = set(context.declared_env_vars)

        for env_access in analysis.env_accesses:
            var_name = env_access.details.get("var_name", "")
            if var_name and var_name not in declared:
                violations.append(Violation(
                    rule_id="ENV-AUTO",
                    violation_type=ViolationType.UNDECLARED_ENV_VAR,
                    message=f"Environment variable '{var_name}' accessed but not declared",
                    file_path=analysis.file_path,
                    line_number=env_access.location.line,
                    code_snippet=env_access.raw_text,
                    fix_suggestion=f"Add '{var_name}' to Settings class or magic protocol",
                    severity=EnforcementLevel.STRICT,
                ))

        return violations

    def _detect_missing_imports(
        self,
        analysis: FileAnalysis,
        context: DetectionContext,
    ) -> list[Violation]:
        """Detect missing magic protocol imports."""
        violations = []

        # Check if file uses magic-provided symbols without proper imports
        for dep in analysis.dependencies:
            dep_func = dep.details.get("dependency_function", "")

            # Check if this dependency is provided by a magic protocol
            is_magic_provided = False
            for protocol in context.magic_protocols:
                provides = protocol.get("provides", {})
                if dep_func in str(provides):
                    is_magic_provided = True
                    break

            if is_magic_provided:
                # Check if proper import exists
                has_import = any(
                    dep_func in imp.details.get("names", [])
                    for imp in analysis.imports
                )

                if not has_import:
                    violations.append(Violation(
                        rule_id="MAGIC-IMPORT",
                        violation_type=ViolationType.MISSING_MAGIC_IMPORT,
                        message=f"Magic dependency '{dep_func}' used without proper import",
                        file_path=analysis.file_path,
                        line_number=dep.location.line,
                        code_snippet=dep.raw_text,
                        fix_suggestion=f"Import '{dep_func}' from the appropriate module",
                        severity=EnforcementLevel.WARN,
                    ))

        return violations
