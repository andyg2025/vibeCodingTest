"""Auto-Fix Suggester for code violations.

Generates actionable fix suggestions and code patches
for detected violations.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

from src.agents.codex import Violation, ViolationType
from src.audit.detector import DetectionResult

logger = structlog.get_logger()


class FixType(str, Enum):
    """Types of fixes that can be suggested."""

    ADD_IMPORT = "add_import"
    ADD_PARAMETER = "add_parameter"
    ADD_DECORATOR = "add_decorator"
    ADD_ENV_VAR = "add_env_var"
    REPLACE_CODE = "replace_code"
    ADD_CODE = "add_code"
    REMOVE_CODE = "remove_code"
    REFACTOR = "refactor"


class FixComplexity(str, Enum):
    """Complexity of applying a fix."""

    TRIVIAL = "trivial"  # Single line change
    SIMPLE = "simple"  # Few lines, localized
    MODERATE = "moderate"  # Multiple locations
    COMPLEX = "complex"  # Requires redesign


@dataclass
class CodePatch:
    """A patch to apply to code."""

    file_path: str
    line_start: int
    line_end: int | None = None
    original: str = ""
    replacement: str = ""
    description: str = ""


@dataclass
class FixSuggestion:
    """A suggested fix for a violation."""

    violation: Violation
    fix_type: FixType
    complexity: FixComplexity
    description: str
    patches: list[CodePatch] = field(default_factory=list)
    auto_applicable: bool = False
    confidence: float = 0.8
    metadata: dict[str, Any] = field(default_factory=dict)


class AutoFixSuggester:
    """Generates fix suggestions for violations.

    The suggester analyzes violations and generates:
    1. Human-readable fix descriptions
    2. Code patches that can be auto-applied
    3. Confidence scores for suggestions
    """

    def __init__(self):
        self._logger = logger.bind(component="AutoFixSuggester")
        self._fix_generators: dict[ViolationType, callable] = {
            ViolationType.UNDECLARED_ENV_VAR: self._fix_undeclared_env_var,
            ViolationType.MISSING_DEPENDENCY: self._fix_missing_dependency,
            ViolationType.MISSING_MAGIC_IMPORT: self._fix_missing_import,
            ViolationType.LIFECYCLE_ORDER: self._fix_lifecycle_order,
            ViolationType.SIDE_EFFECT_MISMATCH: self._fix_side_effect,
            ViolationType.CONTRACT_VIOLATION: self._fix_contract_violation,
        }

    def suggest_fixes(self, result: DetectionResult) -> list[FixSuggestion]:
        """Generate fix suggestions for a detection result.

        Args:
            result: Detection result with violations

        Returns:
            List of fix suggestions
        """
        suggestions = []

        for violation in result.violations:
            generator = self._fix_generators.get(violation.violation_type)
            if generator:
                suggestion = generator(violation, result)
                if suggestion:
                    suggestions.append(suggestion)
            else:
                # Generate generic suggestion
                suggestions.append(self._generic_fix(violation))

        self._logger.info(
            "Generated fix suggestions",
            file=result.file_path,
            violations=len(result.violations),
            suggestions=len(suggestions),
        )

        return suggestions

    def suggest_fixes_batch(
        self, results: list[DetectionResult]
    ) -> dict[str, list[FixSuggestion]]:
        """Generate fix suggestions for multiple files.

        Returns:
            Dict mapping file paths to their fix suggestions
        """
        return {
            result.file_path: self.suggest_fixes(result)
            for result in results
        }

    def _fix_undeclared_env_var(
        self, violation: Violation, result: DetectionResult
    ) -> FixSuggestion:
        """Generate fix for undeclared environment variable."""
        var_name = violation.code_snippet.split('"')[1] if '"' in violation.code_snippet else "VAR"

        # Suggest adding to Settings class
        settings_patch = f'    {var_name}: str = Field(..., env="{var_name}")'

        return FixSuggestion(
            violation=violation,
            fix_type=FixType.ADD_ENV_VAR,
            complexity=FixComplexity.SIMPLE,
            description=f"Add '{var_name}' to the Settings class",
            patches=[
                CodePatch(
                    file_path="src/core/config.py",
                    line_start=0,  # Will be determined by context
                    original="",
                    replacement=settings_patch,
                    description=f"Add {var_name} field to Settings",
                )
            ],
            auto_applicable=False,  # Requires finding Settings class
            confidence=0.9,
            metadata={"var_name": var_name},
        )

    def _fix_missing_dependency(
        self, violation: Violation, result: DetectionResult
    ) -> FixSuggestion:
        """Generate fix for missing dependency (auth, etc)."""
        # Extract context from violation
        is_auth = "auth" in violation.message.lower() or "current_user" in violation.message.lower()

        if is_auth:
            param_code = "current_user: Annotated[User, Depends(get_current_user)]"
            import_code = "from src.api.deps import get_current_user"

            return FixSuggestion(
                violation=violation,
                fix_type=FixType.ADD_PARAMETER,
                complexity=FixComplexity.SIMPLE,
                description="Add authentication dependency to route",
                patches=[
                    CodePatch(
                        file_path=violation.file_path,
                        line_start=violation.line_number or 1,
                        original="",
                        replacement=param_code,
                        description="Add current_user parameter",
                    ),
                    CodePatch(
                        file_path=violation.file_path,
                        line_start=1,
                        original="",
                        replacement=import_code,
                        description="Add get_current_user import",
                    ),
                ],
                auto_applicable=False,
                confidence=0.85,
                metadata={"dependency_type": "auth"},
            )

        return self._generic_fix(violation)

    def _fix_missing_import(
        self, violation: Violation, result: DetectionResult
    ) -> FixSuggestion:
        """Generate fix for missing import."""
        # Try to determine what needs to be imported
        missing_item = violation.code_snippet.split("(")[0] if "(" in violation.code_snippet else ""

        return FixSuggestion(
            violation=violation,
            fix_type=FixType.ADD_IMPORT,
            complexity=FixComplexity.TRIVIAL,
            description=f"Add import for '{missing_item}'",
            patches=[
                CodePatch(
                    file_path=violation.file_path,
                    line_start=1,
                    original="",
                    replacement=f"from src.api.deps import {missing_item}",
                    description=f"Import {missing_item}",
                )
            ],
            auto_applicable=True,
            confidence=0.7,
            metadata={"import_name": missing_item},
        )

    def _fix_lifecycle_order(
        self, violation: Violation, result: DetectionResult
    ) -> FixSuggestion:
        """Generate fix for lifecycle ordering issues."""
        startup_code = '''
@app.on_event("startup")
async def startup():
    """Initialize resources on startup."""
    # Initialize database connection
    pass

@app.on_event("shutdown")
async def shutdown():
    """Cleanup resources on shutdown."""
    # Close database connection
    pass
'''

        return FixSuggestion(
            violation=violation,
            fix_type=FixType.ADD_CODE,
            complexity=FixComplexity.MODERATE,
            description="Add startup/shutdown lifecycle handlers",
            patches=[
                CodePatch(
                    file_path=violation.file_path,
                    line_start=0,  # Append to file
                    original="",
                    replacement=startup_code,
                    description="Add lifecycle event handlers",
                )
            ],
            auto_applicable=False,
            confidence=0.75,
        )

    def _fix_side_effect(
        self, violation: Violation, result: DetectionResult
    ) -> FixSuggestion:
        """Generate fix for side effect mismatch."""
        return FixSuggestion(
            violation=violation,
            fix_type=FixType.REFACTOR,
            complexity=FixComplexity.COMPLEX,
            description="Declare side effect in magic protocol or remove the operation",
            patches=[],
            auto_applicable=False,
            confidence=0.6,
            metadata={
                "options": [
                    "Add side effect declaration to magic protocol",
                    "Move the operation to a function that declares side effects",
                    "Remove the side effect if not needed",
                ]
            },
        )

    def _fix_contract_violation(
        self, violation: Violation, result: DetectionResult
    ) -> FixSuggestion:
        """Generate fix for contract violation."""
        return FixSuggestion(
            violation=violation,
            fix_type=FixType.REFACTOR,
            complexity=FixComplexity.MODERATE,
            description=violation.fix_suggestion or "Review and fix the contract violation",
            patches=[],
            auto_applicable=False,
            confidence=0.5,
        )

    def _generic_fix(self, violation: Violation) -> FixSuggestion:
        """Generate a generic fix suggestion."""
        return FixSuggestion(
            violation=violation,
            fix_type=FixType.REFACTOR,
            complexity=FixComplexity.MODERATE,
            description=violation.fix_suggestion or f"Fix {violation.violation_type.value}",
            patches=[],
            auto_applicable=False,
            confidence=0.5,
        )

    def apply_patches(
        self,
        suggestions: list[FixSuggestion],
        dry_run: bool = True,
    ) -> dict[str, str]:
        """Apply auto-applicable patches.

        Args:
            suggestions: Fix suggestions with patches
            dry_run: If True, return patched content without writing

        Returns:
            Dict mapping file paths to patched content
        """
        # Group patches by file
        patches_by_file: dict[str, list[CodePatch]] = {}
        for suggestion in suggestions:
            if suggestion.auto_applicable:
                for patch in suggestion.patches:
                    if patch.file_path not in patches_by_file:
                        patches_by_file[patch.file_path] = []
                    patches_by_file[patch.file_path].append(patch)

        results = {}

        for file_path, patches in patches_by_file.items():
            try:
                # Read file
                from pathlib import Path
                path = Path(file_path)
                if not path.exists():
                    continue

                content = path.read_text()
                lines = content.split("\n")

                # Apply patches (in reverse order to preserve line numbers)
                patches.sort(key=lambda p: p.line_start, reverse=True)

                for patch in patches:
                    if patch.line_start == 1 and not patch.original:
                        # Insert at beginning (after imports)
                        import_end = 0
                        for i, line in enumerate(lines):
                            if line.startswith("import ") or line.startswith("from "):
                                import_end = i + 1
                        lines.insert(import_end, patch.replacement)
                    elif patch.original:
                        # Replace existing code
                        for i, line in enumerate(lines):
                            if patch.original in line:
                                lines[i] = line.replace(patch.original, patch.replacement)
                                break

                patched_content = "\n".join(lines)
                results[file_path] = patched_content

                if not dry_run:
                    path.write_text(patched_content)
                    self._logger.info("Applied patches", file=file_path, patches=len(patches))

            except Exception as e:
                self._logger.error("Failed to apply patches", file=file_path, error=str(e))

        return results

    def format_suggestions(self, suggestions: list[FixSuggestion]) -> str:
        """Format suggestions as human-readable text."""
        if not suggestions:
            return "No fix suggestions available."

        lines = ["Fix Suggestions:", "=" * 40, ""]

        for i, s in enumerate(suggestions, 1):
            lines.append(f"{i}. [{s.complexity.value.upper()}] {s.description}")
            lines.append(f"   Rule: {s.violation.rule_id}")
            lines.append(f"   Type: {s.fix_type.value}")
            lines.append(f"   Confidence: {s.confidence:.0%}")
            lines.append(f"   Auto-applicable: {'Yes' if s.auto_applicable else 'No'}")

            if s.patches:
                lines.append("   Patches:")
                for patch in s.patches:
                    lines.append(f"     - {patch.file_path}: {patch.description}")

            if s.metadata.get("options"):
                lines.append("   Options:")
                for opt in s.metadata["options"]:
                    lines.append(f"     • {opt}")

            lines.append("")

        return "\n".join(lines)
