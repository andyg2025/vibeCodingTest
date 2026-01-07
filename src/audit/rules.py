"""Rule Engine for executing Codex rules against analyzed code.

The Rule Engine:
1. Loads rules from the Codex
2. Matches rules against code analysis results
3. Produces violations when rules are breached
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Protocol

import structlog

from src.agents.codex import Codex, CodexRule, EnforcementLevel, ViolationType
from src.audit.analyzer import FileAnalysis, AnalysisCategory, AnalysisResult

logger = structlog.get_logger()


class RuleMatchType(str, Enum):
    """How a rule matches against code."""

    ENV_VAR_DECLARED = "env_var_declared"
    DEPENDENCY_DECLARED = "dependency_declared"
    DECORATOR_PRESENT = "decorator_present"
    IMPORT_PRESENT = "import_present"
    SIDE_EFFECT_ALLOWED = "side_effect_allowed"
    LIFECYCLE_ORDER = "lifecycle_order"
    PATTERN_MATCH = "pattern_match"


@dataclass
class RuleMatch:
    """A rule that matched (or failed to match) against code."""

    rule: CodexRule
    matched: bool
    match_type: RuleMatchType
    analysis_result: AnalysisResult | None = None
    context: dict[str, Any] = field(default_factory=dict)
    message: str = ""


class RuleMatcher(Protocol):
    """Protocol for rule matching functions."""

    def __call__(
        self,
        rule: CodexRule,
        analysis: FileAnalysis,
        context: dict[str, Any],
    ) -> list[RuleMatch]:
        ...


@dataclass
class RuleDefinition:
    """A rule definition with its matcher."""

    rule_id_prefix: str
    description: str
    matcher: RuleMatcher
    default_enforcement: EnforcementLevel = EnforcementLevel.STRICT


class RuleEngine:
    """Executes Codex rules against analyzed code.

    The engine supports both:
    1. Built-in rules (registered matchers)
    2. Dynamic rules from the Codex
    """

    def __init__(self, codex: Codex | None = None):
        self.codex = codex
        self._matchers: dict[str, RuleMatcher] = {}
        self._logger = logger.bind(component="RuleEngine")

        # Register built-in matchers
        self._register_builtin_matchers()

    def _register_builtin_matchers(self) -> None:
        """Register the built-in rule matchers."""
        self._matchers["ENV"] = self._match_env_rules
        self._matchers["AUTH"] = self._match_auth_rules
        self._matchers["DB"] = self._match_db_rules
        self._matchers["LIFECYCLE"] = self._match_lifecycle_rules
        self._matchers["SIDE_EFFECT"] = self._match_side_effect_rules
        self._matchers["GLOBAL"] = self._match_global_rules

    def register_matcher(self, prefix: str, matcher: RuleMatcher) -> None:
        """Register a custom rule matcher."""
        self._matchers[prefix] = matcher

    def execute(
        self,
        analysis: FileAnalysis,
        context: dict[str, Any] | None = None,
    ) -> list[RuleMatch]:
        """Execute all applicable rules against the analyzed code.

        Args:
            analysis: The analyzed file
            context: Additional context (declared env vars, magic protocols, etc)

        Returns:
            List of rule matches (both passed and failed)
        """
        context = context or {}
        matches: list[RuleMatch] = []

        # Get rules from codex if available
        rules = self.codex.get_all_rules() if self.codex else []

        self._logger.info(
            "Executing rules",
            file=analysis.file_path,
            rule_count=len(rules),
        )

        for rule in rules:
            # Find the appropriate matcher based on rule ID prefix
            prefix = rule.rule_id.split("-")[0] if "-" in rule.rule_id else rule.rule_id
            matcher = self._matchers.get(prefix)

            if matcher:
                rule_matches = matcher(rule, analysis, context)
                matches.extend(rule_matches)
            else:
                # Try generic pattern matching
                generic_matches = self._generic_match(rule, analysis, context)
                matches.extend(generic_matches)

        # Log summary
        failed = [m for m in matches if not m.matched]
        self._logger.info(
            "Rule execution complete",
            total_matches=len(matches),
            failures=len(failed),
        )

        return matches

    def _match_env_rules(
        self,
        rule: CodexRule,
        analysis: FileAnalysis,
        context: dict[str, Any],
    ) -> list[RuleMatch]:
        """Match environment variable rules."""
        matches = []
        declared_env_vars = set(context.get("declared_env_vars", []))

        for env_access in analysis.env_accesses:
            var_name = env_access.details.get("var_name", "")
            is_declared = var_name in declared_env_vars

            matches.append(RuleMatch(
                rule=rule,
                matched=is_declared,
                match_type=RuleMatchType.ENV_VAR_DECLARED,
                analysis_result=env_access,
                context={"var_name": var_name, "declared": list(declared_env_vars)},
                message=f"Environment variable '{var_name}' {'is' if is_declared else 'is NOT'} declared",
            ))

        return matches

    def _match_auth_rules(
        self,
        rule: CodexRule,
        analysis: FileAnalysis,
        context: dict[str, Any],
    ) -> list[RuleMatch]:
        """Match authentication rules."""
        matches = []
        protected_patterns = context.get("protected_patterns", ["/admin", "/user", "/protected"])

        # Check decorators for route definitions
        for decorator in analysis.decorators:
            decorator_text = decorator.details.get("decorator", "")

            # Check if this is a route decorator with a protected path
            is_protected_route = False
            for pattern in protected_patterns:
                if pattern in decorator_text:
                    is_protected_route = True
                    break

            if is_protected_route:
                # Check if the decorated function has auth dependency
                decorated_name = decorator.details.get("decorates", "")
                has_auth = self._function_has_auth_dependency(decorated_name, analysis)

                matches.append(RuleMatch(
                    rule=rule,
                    matched=has_auth,
                    match_type=RuleMatchType.DEPENDENCY_DECLARED,
                    analysis_result=decorator,
                    context={"route": decorator_text, "has_auth": has_auth},
                    message=f"Protected route '{decorator_text}' {'has' if has_auth else 'MISSING'} auth dependency",
                ))

        return matches

    def _function_has_auth_dependency(self, func_name: str, analysis: FileAnalysis) -> bool:
        """Check if a function has authentication dependency."""
        # Look for get_current_user, current_user in function parameters
        for func_def in analysis.get_by_category(AnalysisCategory.FUNCTION_DEFINITIONS):
            if func_def.name == func_name:
                params = func_def.details.get("parameters", [])
                for param in params:
                    param_name = param.get("name", "")
                    param_type = param.get("type", "") or ""
                    if "current_user" in param_name.lower() or "current_user" in param_type.lower():
                        return True
                    if "get_current_user" in param_type:
                        return True
        return False

    def _match_db_rules(
        self,
        rule: CodexRule,
        analysis: FileAnalysis,
        context: dict[str, Any],
    ) -> list[RuleMatch]:
        """Match database session rules."""
        matches = []
        db_config_files = context.get("db_config_files", ["database.py", "config.py"])

        # Check if this is a config file (allowed to create sessions)
        is_config_file = any(f in analysis.file_path for f in db_config_files)

        if not is_config_file:
            # Check for direct session creation
            for func_call in analysis.function_calls:
                func_name = func_call.details.get("function", "")
                bad_patterns = ["Session()", "sessionmaker(", "create_engine("]

                is_violation = any(p in func_name for p in bad_patterns)

                if is_violation:
                    matches.append(RuleMatch(
                        rule=rule,
                        matched=False,
                        match_type=RuleMatchType.PATTERN_MATCH,
                        analysis_result=func_call,
                        context={"function": func_name},
                        message=f"Direct database session creation detected: {func_name}",
                    ))

        return matches

    def _match_lifecycle_rules(
        self,
        rule: CodexRule,
        analysis: FileAnalysis,
        context: dict[str, Any],
    ) -> list[RuleMatch]:
        """Match lifecycle ordering rules."""
        matches = []

        # Check for proper startup/shutdown patterns
        has_startup = False
        has_shutdown = False

        for decorator in analysis.decorators:
            dec_text = decorator.details.get("decorator", "")
            if "on_event" in dec_text:
                if "startup" in dec_text:
                    has_startup = True
                if "shutdown" in dec_text:
                    has_shutdown = True

        # If file uses DB, check for proper lifecycle
        uses_db = any(
            "database" in imp.name.lower() or "session" in imp.name.lower()
            for imp in analysis.imports
        )

        if uses_db and "main.py" in analysis.file_path:
            matches.append(RuleMatch(
                rule=rule,
                matched=has_startup,
                match_type=RuleMatchType.LIFECYCLE_ORDER,
                analysis_result=None,
                context={"has_startup": has_startup, "has_shutdown": has_shutdown},
                message=f"Database lifecycle: startup={'✓' if has_startup else '✗'}, shutdown={'✓' if has_shutdown else '✗'}",
            ))

        return matches

    def _match_side_effect_rules(
        self,
        rule: CodexRule,
        analysis: FileAnalysis,
        context: dict[str, Any],
    ) -> list[RuleMatch]:
        """Match side effect declaration rules."""
        matches = []
        declared_side_effects = context.get("declared_side_effects", [])

        for side_effect in analysis.get_by_category(AnalysisCategory.SIDE_EFFECTS):
            func_name = side_effect.details.get("function", "")
            is_declared = func_name in declared_side_effects

            matches.append(RuleMatch(
                rule=rule,
                matched=is_declared,
                match_type=RuleMatchType.SIDE_EFFECT_ALLOWED,
                analysis_result=side_effect,
                context={"function": func_name},
                message=f"Side effect '{func_name}' {'is' if is_declared else 'is NOT'} declared in magic protocol",
            ))

        return matches

    def _match_global_rules(
        self,
        rule: CodexRule,
        analysis: FileAnalysis,
        context: dict[str, Any],
    ) -> list[RuleMatch]:
        """Match global/catch-all rules."""
        # Delegate to specific matchers based on rule content
        if "env" in rule.rule_id.lower() or "environment" in rule.description.lower():
            return self._match_env_rules(rule, analysis, context)
        if "auth" in rule.rule_id.lower() or "authentication" in rule.description.lower():
            return self._match_auth_rules(rule, analysis, context)
        if "db" in rule.rule_id.lower() or "database" in rule.description.lower():
            return self._match_db_rules(rule, analysis, context)

        return []

    def _generic_match(
        self,
        rule: CodexRule,
        analysis: FileAnalysis,
        context: dict[str, Any],
    ) -> list[RuleMatch]:
        """Generic pattern matching for rules without specific matchers."""
        # Use the rule's validator if available
        if rule.validator:
            # Reconstruct code from analysis (simplified)
            code = analysis.results[0].raw_text if analysis.results else ""
            violations = rule.validator(code, context)
            return [
                RuleMatch(
                    rule=rule,
                    matched=len(violations) == 0,
                    match_type=RuleMatchType.PATTERN_MATCH,
                    message=v.message if violations else "Rule passed",
                )
                for v in violations
            ] if violations else []

        return []
