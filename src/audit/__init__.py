"""Magic Dependency Audit System for the Graph-Driven Code Agent Architecture.

This module implements Phase 4 of the plan:
- Code Analyzer: Tree-sitter based AST analysis
- Rule Engine: Executes Codex rules against code
- Violation Detector: Combines analysis and rules
- Report Generator: Multiple output formats
- Auto-Fix Suggester: Generates fix suggestions
- Audit Agent: Orchestrates the audit pipeline
"""

from .analyzer import (
    AnalysisCategory,
    AnalysisResult,
    CodeAnalyzer,
    CodeLocation,
    FileAnalysis,
)
from .rules import (
    RuleEngine,
    RuleMatch,
    RuleMatchType,
)
from .detector import (
    DetectionContext,
    DetectionResult,
    ViolationDetector,
)
from .reporter import (
    AuditReport,
    AuditSummary,
    ReportFormat,
    ReportGenerator,
)
from .fixer import (
    AutoFixSuggester,
    CodePatch,
    FixComplexity,
    FixSuggestion,
    FixType,
)
from .agent import AuditAgent

__all__ = [
    # Analyzer
    "AnalysisCategory",
    "AnalysisResult",
    "CodeAnalyzer",
    "CodeLocation",
    "FileAnalysis",
    # Rules
    "RuleEngine",
    "RuleMatch",
    "RuleMatchType",
    # Detector
    "DetectionContext",
    "DetectionResult",
    "ViolationDetector",
    # Reporter
    "AuditReport",
    "AuditSummary",
    "ReportFormat",
    "ReportGenerator",
    # Fixer
    "AutoFixSuggester",
    "CodePatch",
    "FixComplexity",
    "FixSuggestion",
    "FixType",
    # Agent
    "AuditAgent",
]
