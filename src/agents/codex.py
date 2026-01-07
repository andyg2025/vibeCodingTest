"""Dependency Codex - The Law Book for Magic Dependencies.

The Codex is the central authority that governs how all generated code
must handle magic dependencies. It contains:

1. **Rules**: Enforceable constraints on code generation
2. **Contracts**: What each magic protocol provides and requires
3. **Validators**: Functions to check code compliance
4. **Fixers**: Suggested fixes for violations

The Codex is consulted by:
- DFS Implementation Agent: To ensure generated code is compliant
- Audit Agent: To verify final code against rules
- Backtracking Engine: To determine if violations require redesign
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import structlog

logger = structlog.get_logger()


class EnforcementLevel(str, Enum):
    """How strictly a rule is enforced."""

    STRICT = "strict"  # Violations block code generation
    WARN = "warn"  # Violations generate warnings but allow generation
    INFO = "info"  # For documentation, no enforcement


class ViolationType(str, Enum):
    """Types of codex violations."""

    MISSING_DEPENDENCY = "missing_dependency"
    UNDECLARED_ENV_VAR = "undeclared_env_var"
    CONTRACT_VIOLATION = "contract_violation"
    LIFECYCLE_ORDER = "lifecycle_order"
    SIDE_EFFECT_MISMATCH = "side_effect_mismatch"
    MISSING_ERROR_HANDLING = "missing_error_handling"
    # Additional types for backtracking
    CRITICAL = "critical"
    CIRCULAR_DEPENDENCY = "circular_dependency"
    TYPE_MISMATCH = "type_mismatch"
    MISSING_MAGIC_IMPORT = "missing_magic_import"


@dataclass
class CodexRule:
    """A single rule in the Codex."""

    rule_id: str
    description: str
    enforcement_level: EnforcementLevel
    magic_protocol_id: str | None = None
    validator: Callable[[str, dict[str, Any]], list["Violation"]] | None = None
    fix_suggestion: str = ""
    auto_fixable: bool = False


@dataclass
class Violation:
    """A violation of a Codex rule."""

    rule_id: str
    violation_type: ViolationType
    message: str
    file_path: str
    line_number: int | None = None
    code_snippet: str = ""
    fix_suggestion: str = ""
    severity: EnforcementLevel = EnforcementLevel.STRICT

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "violation_type": self.violation_type.value,
            "message": self.message,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "code_snippet": self.code_snippet,
            "fix_suggestion": self.fix_suggestion,
            "severity": self.severity.value,
        }

    def model_dump(self) -> dict[str, Any]:
        """Pydantic-compatible serialization."""
        return self.to_dict()

    @property
    def suggestion(self) -> str:
        """Alias for fix_suggestion for API compatibility."""
        return self.fix_suggestion

    @property
    def details(self) -> dict[str, Any]:
        """Additional details for the violation."""
        return {
            "code_snippet": self.code_snippet,
            "line_number": self.line_number,
        }


@dataclass
class MagicContract:
    """Contract specification for a magic protocol."""

    protocol_id: str
    name: str
    provides: dict[str, dict[str, Any]] = field(default_factory=dict)
    requires: dict[str, Any] = field(default_factory=dict)
    side_effects: list[dict[str, Any]] = field(default_factory=list)


class Codex:
    """The Dependency Codex - central rule authority.

    The Codex is immutable once built. All agents must consult it
    before generating code that involves magic dependencies.
    """

    def __init__(self):
        self._rules: dict[str, CodexRule] = {}
        self._contracts: dict[str, MagicContract] = {}
        self._frozen = False

    def add_rule(self, rule: CodexRule) -> None:
        """Add a rule to the Codex."""
        if self._frozen:
            raise RuntimeError("Cannot modify frozen Codex")
        self._rules[rule.rule_id] = rule

    def add_contract(self, contract: MagicContract) -> None:
        """Add a magic contract to the Codex."""
        if self._frozen:
            raise RuntimeError("Cannot modify frozen Codex")
        self._contracts[contract.protocol_id] = contract

    def freeze(self) -> None:
        """Freeze the Codex, preventing further modifications."""
        self._frozen = True
        logger.info(
            "Codex frozen",
            rule_count=len(self._rules),
            contract_count=len(self._contracts),
        )

    def get_rule(self, rule_id: str) -> CodexRule | None:
        """Get a rule by ID."""
        return self._rules.get(rule_id)

    def get_contract(self, protocol_id: str) -> MagicContract | None:
        """Get a contract by protocol ID."""
        return self._contracts.get(protocol_id)

    def get_rules_for_protocol(self, protocol_id: str) -> list[CodexRule]:
        """Get all rules associated with a magic protocol."""
        return [
            rule for rule in self._rules.values()
            if rule.magic_protocol_id == protocol_id
        ]

    def get_all_rules(self) -> list[CodexRule]:
        """Get all rules in the Codex."""
        return list(self._rules.values())

    def get_all_contracts(self) -> list[MagicContract]:
        """Get all contracts in the Codex."""
        return list(self._contracts.values())

    def get_required_env_vars(self) -> set[str]:
        """Get all environment variables required by contracts."""
        env_vars = set()
        for contract in self._contracts.values():
            requires = contract.requires
            if isinstance(requires, dict):
                env_vars.update(requires.get("env_vars", []))
        return env_vars

    def validate_code(
        self,
        code: str,
        context: dict[str, Any],
        file_path: str = "<generated>",
    ) -> list[Violation]:
        """Validate code against all applicable Codex rules."""
        violations = []

        for rule in self._rules.values():
            if rule.validator:
                rule_violations = rule.validator(code, context)
                for v in rule_violations:
                    v.file_path = file_path
                violations.extend(rule_violations)

        return violations

    def check_compliance(
        self,
        code: str,
        magic_protocols: list[dict[str, Any]],
        file_path: str = "<generated>",
    ) -> list[Violation]:
        """Check code compliance against magic protocols.

        This is the main entry point for the DFS agent to validate generated code.
        """
        violations = []

        # Check against general rules
        context = {"magic_protocols": magic_protocols}
        violations.extend(self.validate_code(code, context, file_path))

        # Check against each magic protocol's contract
        for protocol in magic_protocols:
            protocol_id = protocol.get("id", "")
            if protocol_id:
                protocol_violations = self.check_contract_compliance(
                    code, protocol_id, context
                )
                violations.extend(protocol_violations)

        return violations

    def check_contract_compliance(
        self,
        code: str,
        protocol_id: str,
        context: dict[str, Any],
    ) -> list[Violation]:
        """Check if code complies with a specific magic contract."""
        contract = self._contracts.get(protocol_id)
        if not contract:
            return []

        violations = []

        # Check required dependencies are declared
        for req_key, req_value in contract.requires.items():
            if req_key == "env_vars":
                for env_var in req_value:
                    if env_var not in context.get("declared_env_vars", []):
                        violations.append(Violation(
                            rule_id=f"{protocol_id}-ENV",
                            violation_type=ViolationType.UNDECLARED_ENV_VAR,
                            message=f"Environment variable '{env_var}' required by {contract.name} but not declared",
                            file_path="",
                            fix_suggestion=f"Add {env_var} to Settings class",
                        ))

            elif req_key == "headers":
                for header in req_value:
                    if header not in context.get("declared_headers", []):
                        violations.append(Violation(
                            rule_id=f"{protocol_id}-HEADER",
                            violation_type=ViolationType.CONTRACT_VIOLATION,
                            message=f"Header '{header}' required by {contract.name} but not declared",
                            file_path="",
                        ))

            elif req_key == "dependencies":
                for dep in req_value:
                    if dep not in context.get("declared_dependencies", []):
                        violations.append(Violation(
                            rule_id=f"{protocol_id}-DEP",
                            violation_type=ViolationType.MISSING_DEPENDENCY,
                            message=f"Dependency '{dep}' required by {contract.name}",
                            file_path="",
                        ))

        return violations

    def to_dict(self) -> dict[str, Any]:
        """Serialize the Codex to a dictionary."""
        return {
            "version": "1.0",
            "frozen": self._frozen,
            "rules": [
                {
                    "rule_id": r.rule_id,
                    "description": r.description,
                    "enforcement_level": r.enforcement_level.value,
                    "magic_protocol_id": r.magic_protocol_id,
                    "fix_suggestion": r.fix_suggestion,
                    "auto_fixable": r.auto_fixable,
                }
                for r in self._rules.values()
            ],
            "contracts": [
                {
                    "protocol_id": c.protocol_id,
                    "name": c.name,
                    "provides": c.provides,
                    "requires": c.requires,
                    "side_effects": c.side_effects,
                }
                for c in self._contracts.values()
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Codex":
        """Deserialize a Codex from a dictionary."""
        codex = cls()

        for rule_data in data.get("rules", []):
            rule = CodexRule(
                rule_id=rule_data["rule_id"],
                description=rule_data["description"],
                enforcement_level=EnforcementLevel(rule_data["enforcement_level"]),
                magic_protocol_id=rule_data.get("magic_protocol_id"),
                fix_suggestion=rule_data.get("fix_suggestion", ""),
                auto_fixable=rule_data.get("auto_fixable", False),
            )
            codex.add_rule(rule)

        for contract_data in data.get("contracts", []):
            contract = MagicContract(
                protocol_id=contract_data["protocol_id"],
                name=contract_data["name"],
                provides=contract_data.get("provides", {}),
                requires=contract_data.get("requires", {}),
                side_effects=contract_data.get("side_effects", []),
            )
            codex.add_contract(contract)

        if data.get("frozen", False):
            codex.freeze()

        return codex


class CodexBuilder:
    """Builder for constructing a Codex from magic protocols."""

    def __init__(self):
        self._codex = Codex()

    def add_magic_protocol(self, protocol: dict[str, Any]) -> "CodexBuilder":
        """Add a magic protocol and its associated rules/contracts."""
        protocol_id = protocol.get("id", "")
        name = protocol.get("name", "")
        contract_data = protocol.get("contract", {})

        # Create contract
        contract = MagicContract(
            protocol_id=protocol_id,
            name=name,
            provides=contract_data.get("provides", {}),
            requires=contract_data.get("requires", {}),
            side_effects=contract_data.get("side_effects", []),
        )
        self._codex.add_contract(contract)

        # Create rules from codex_rules
        for rule_data in protocol.get("codex_rules", []):
            rule = CodexRule(
                rule_id=rule_data.get("rule_id", f"{protocol_id}-DEFAULT"),
                description=rule_data.get("description", ""),
                enforcement_level=EnforcementLevel(
                    rule_data.get("enforcement_level", "strict")
                ),
                magic_protocol_id=protocol_id,
                fix_suggestion=rule_data.get("fix_suggestion", ""),
                auto_fixable=rule_data.get("auto_fixable", False),
            )
            self._codex.add_rule(rule)

        return self

    def add_standard_rules(self) -> "CodexBuilder":
        """Add standard rules that apply to all projects."""
        standard_rules = [
            CodexRule(
                rule_id="GLOBAL-001",
                description="All environment variables must be declared in Settings",
                enforcement_level=EnforcementLevel.STRICT,
                validator=_validate_env_vars_declared,
            ),
            CodexRule(
                rule_id="GLOBAL-002",
                description="Protected routes must have authentication dependency",
                enforcement_level=EnforcementLevel.STRICT,
                validator=_validate_auth_on_protected_routes,
            ),
            CodexRule(
                rule_id="GLOBAL-003",
                description="Database operations must use injected session",
                enforcement_level=EnforcementLevel.STRICT,
                validator=_validate_db_session_usage,
            ),
            CodexRule(
                rule_id="GLOBAL-004",
                description="Background tasks must not access request context",
                enforcement_level=EnforcementLevel.WARN,
            ),
            CodexRule(
                rule_id="GLOBAL-005",
                description="All exceptions from magic protocols must be handled",
                enforcement_level=EnforcementLevel.WARN,
            ),
        ]

        for rule in standard_rules:
            self._codex.add_rule(rule)

        return self

    def build(self, freeze: bool = True) -> Codex:
        """Build and optionally freeze the Codex."""
        if freeze:
            self._codex.freeze()
        return self._codex


# Standard validators

def _validate_env_vars_declared(code: str, context: dict[str, Any]) -> list[Violation]:
    """Check that all env var accesses are declared."""
    import re
    violations = []

    declared = set(context.get("declared_env_vars", []))

    # Find os.getenv calls
    patterns = [
        r'os\.getenv\s*\(\s*["\'](\w+)["\']',
        r'os\.environ\s*\[\s*["\'](\w+)["\']',
        r'os\.environ\.get\s*\(\s*["\'](\w+)["\']',
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, code):
            env_var = match.group(1)
            if env_var not in declared:
                violations.append(Violation(
                    rule_id="GLOBAL-001",
                    violation_type=ViolationType.UNDECLARED_ENV_VAR,
                    message=f"Environment variable '{env_var}' accessed but not declared in Settings",
                    file_path="",
                    code_snippet=match.group(0),
                    fix_suggestion=f"Add '{env_var}: str' to Settings class",
                ))

    return violations


def _validate_auth_on_protected_routes(code: str, context: dict[str, Any]) -> list[Violation]:
    """Check that protected routes have auth dependency."""
    import re
    violations = []

    protected_patterns = context.get("protected_route_patterns", [
        r"@\w+\.(get|post|put|delete|patch)\s*\(\s*[\"']/admin",
        r"@\w+\.(get|post|put|delete|patch)\s*\(\s*[\"']/user",
        r"@\w+\.(get|post|put|delete|patch)\s*\(\s*[\"']/api/v\d+/protected",
    ])

    for pattern in protected_patterns:
        for match in re.finditer(pattern, code, re.MULTILINE):
            # Check if next function has auth dependency
            start = match.end()
            next_func = code[start:start + 500]  # Look ahead
            if "current_user" not in next_func and "get_current_user" not in next_func:
                violations.append(Violation(
                    rule_id="GLOBAL-002",
                    violation_type=ViolationType.MISSING_DEPENDENCY,
                    message="Protected route missing authentication dependency",
                    file_path="",
                    code_snippet=match.group(0),
                    fix_suggestion="Add 'current_user: User = Depends(get_current_user)' parameter",
                ))

    return violations


def _validate_db_session_usage(code: str, context: dict[str, Any]) -> list[Violation]:
    """Check that database operations use injected session."""
    import re
    violations = []

    # Check for direct session creation (bad pattern)
    bad_patterns = [
        r"Session\s*\(\s*\)",
        r"sessionmaker\s*\(",
        r"create_engine\s*\(",  # Should only be in config
    ]

    allowed_files = context.get("db_config_files", ["database.py", "config.py"])
    current_file = context.get("current_file", "")

    # Skip if this is an allowed file
    if any(allowed in current_file for allowed in allowed_files):
        return violations

    for pattern in bad_patterns:
        for match in re.finditer(pattern, code):
            violations.append(Violation(
                rule_id="GLOBAL-003",
                violation_type=ViolationType.CONTRACT_VIOLATION,
                message="Direct database session creation detected - use injected session",
                file_path="",
                code_snippet=match.group(0),
                fix_suggestion="Use 'db: AsyncSession = Depends(get_db)' parameter instead",
            ))

    return violations
