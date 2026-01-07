"""Report Generator for audit results.

Generates human-readable and machine-readable reports
from violation detection results.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import structlog

from src.agents.codex import EnforcementLevel, Violation
from src.audit.detector import DetectionResult

logger = structlog.get_logger()


class ReportFormat(str, Enum):
    """Output format for reports."""

    TEXT = "text"
    JSON = "json"
    MARKDOWN = "markdown"
    SARIF = "sarif"  # Static Analysis Results Interchange Format


@dataclass
class AuditSummary:
    """Summary statistics for an audit."""

    total_files: int = 0
    files_passed: int = 0
    files_failed: int = 0
    total_violations: int = 0
    strict_violations: int = 0
    warnings: int = 0
    by_rule: dict[str, int] = field(default_factory=dict)
    by_type: dict[str, int] = field(default_factory=dict)


@dataclass
class AuditReport:
    """Complete audit report."""

    timestamp: datetime
    summary: AuditSummary
    results: list[DetectionResult]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "summary": {
                "total_files": self.summary.total_files,
                "files_passed": self.summary.files_passed,
                "files_failed": self.summary.files_failed,
                "total_violations": self.summary.total_violations,
                "strict_violations": self.summary.strict_violations,
                "warnings": self.summary.warnings,
                "by_rule": self.summary.by_rule,
                "by_type": self.summary.by_type,
            },
            "results": [
                {
                    "file": r.file_path,
                    "passed": r.passed,
                    "violations": [v.to_dict() for v in r.violations],
                }
                for r in self.results
            ],
            "metadata": self.metadata,
        }


class ReportGenerator:
    """Generates audit reports in various formats."""

    def __init__(self):
        self._logger = logger.bind(component="ReportGenerator")

    def generate(
        self,
        results: list[DetectionResult],
        format: ReportFormat = ReportFormat.TEXT,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Generate a report from detection results.

        Args:
            results: List of detection results
            format: Output format
            metadata: Additional metadata to include

        Returns:
            Formatted report string
        """
        # Build summary
        summary = self._build_summary(results)

        # Create report
        report = AuditReport(
            timestamp=datetime.utcnow(),
            summary=summary,
            results=results,
            metadata=metadata or {},
        )

        # Format output
        match format:
            case ReportFormat.TEXT:
                return self._format_text(report)
            case ReportFormat.JSON:
                return self._format_json(report)
            case ReportFormat.MARKDOWN:
                return self._format_markdown(report)
            case ReportFormat.SARIF:
                return self._format_sarif(report)
            case _:
                return self._format_text(report)

    def _build_summary(self, results: list[DetectionResult]) -> AuditSummary:
        """Build summary statistics from results."""
        summary = AuditSummary()

        for result in results:
            summary.total_files += 1

            if result.passed:
                summary.files_passed += 1
            else:
                summary.files_failed += 1

            for violation in result.violations:
                summary.total_violations += 1

                if violation.severity == EnforcementLevel.STRICT:
                    summary.strict_violations += 1
                else:
                    summary.warnings += 1

                # Count by rule
                rule_id = violation.rule_id
                summary.by_rule[rule_id] = summary.by_rule.get(rule_id, 0) + 1

                # Count by type
                vtype = violation.violation_type.value
                summary.by_type[vtype] = summary.by_type.get(vtype, 0) + 1

        return summary

    def _format_text(self, report: AuditReport) -> str:
        """Format as plain text."""
        lines = []
        s = report.summary

        # Header
        lines.append("=" * 60)
        lines.append("MAGIC DEPENDENCY AUDIT REPORT")
        lines.append("=" * 60)
        lines.append(f"Timestamp: {report.timestamp.isoformat()}")
        lines.append("")

        # Summary
        lines.append("SUMMARY")
        lines.append("-" * 40)
        lines.append(f"Total Files:       {s.total_files}")
        lines.append(f"Files Passed:      {s.files_passed}")
        lines.append(f"Files Failed:      {s.files_failed}")
        lines.append(f"Total Violations:  {s.total_violations}")
        lines.append(f"  - Strict:        {s.strict_violations}")
        lines.append(f"  - Warnings:      {s.warnings}")
        lines.append("")

        # Violations by rule
        if s.by_rule:
            lines.append("VIOLATIONS BY RULE")
            lines.append("-" * 40)
            for rule_id, count in sorted(s.by_rule.items()):
                lines.append(f"  {rule_id}: {count}")
            lines.append("")

        # Details
        lines.append("DETAILED RESULTS")
        lines.append("-" * 40)

        for result in report.results:
            status = "✓ PASS" if result.passed else "✗ FAIL"
            lines.append(f"\n{status} {result.file_path}")

            if result.violations:
                for v in result.violations:
                    severity = "ERROR" if v.severity == EnforcementLevel.STRICT else "WARN"
                    location = f":{v.line_number}" if v.line_number else ""
                    lines.append(f"  [{severity}] {v.rule_id}{location}: {v.message}")
                    if v.fix_suggestion:
                        lines.append(f"         Fix: {v.fix_suggestion}")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)

    def _format_json(self, report: AuditReport) -> str:
        """Format as JSON."""
        return json.dumps(report.to_dict(), indent=2)

    def _format_markdown(self, report: AuditReport) -> str:
        """Format as Markdown."""
        lines = []
        s = report.summary

        # Header
        lines.append("# Magic Dependency Audit Report")
        lines.append("")
        lines.append(f"**Generated:** {report.timestamp.isoformat()}")
        lines.append("")

        # Summary
        lines.append("## Summary")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Total Files | {s.total_files} |")
        lines.append(f"| Files Passed | {s.files_passed} |")
        lines.append(f"| Files Failed | {s.files_failed} |")
        lines.append(f"| Total Violations | {s.total_violations} |")
        lines.append(f"| Strict Violations | {s.strict_violations} |")
        lines.append(f"| Warnings | {s.warnings} |")
        lines.append("")

        # Violations by type
        if s.by_type:
            lines.append("### Violations by Type")
            lines.append("")
            for vtype, count in sorted(s.by_type.items()):
                lines.append(f"- **{vtype}**: {count}")
            lines.append("")

        # Details
        lines.append("## Detailed Results")
        lines.append("")

        for result in report.results:
            status = "✅" if result.passed else "❌"
            lines.append(f"### {status} `{result.file_path}`")
            lines.append("")

            if result.violations:
                lines.append("| Severity | Rule | Line | Message |")
                lines.append("|----------|------|------|---------|")
                for v in result.violations:
                    severity = "🔴 ERROR" if v.severity == EnforcementLevel.STRICT else "🟡 WARN"
                    line = v.line_number or "-"
                    lines.append(f"| {severity} | {v.rule_id} | {line} | {v.message} |")
                lines.append("")

                # Fix suggestions
                fixes = [v for v in result.violations if v.fix_suggestion]
                if fixes:
                    lines.append("**Suggested Fixes:**")
                    lines.append("")
                    for v in fixes:
                        lines.append(f"- `{v.rule_id}`: {v.fix_suggestion}")
                    lines.append("")
            else:
                lines.append("No violations found.")
                lines.append("")

        return "\n".join(lines)

    def _format_sarif(self, report: AuditReport) -> str:
        """Format as SARIF (Static Analysis Results Interchange Format).

        SARIF is a standard format for static analysis tools,
        supported by GitHub, Azure DevOps, etc.
        """
        sarif = {
            "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
            "version": "2.1.0",
            "runs": [
                {
                    "tool": {
                        "driver": {
                            "name": "MagicDependencyAudit",
                            "version": "1.0.0",
                            "informationUri": "https://vibecoding.dev",
                            "rules": self._sarif_rules(report),
                        }
                    },
                    "results": self._sarif_results(report),
                }
            ],
        }

        return json.dumps(sarif, indent=2)

    def _sarif_rules(self, report: AuditReport) -> list[dict[str, Any]]:
        """Generate SARIF rule definitions."""
        rules = {}

        for result in report.results:
            for v in result.violations:
                if v.rule_id not in rules:
                    rules[v.rule_id] = {
                        "id": v.rule_id,
                        "name": v.rule_id,
                        "shortDescription": {"text": v.violation_type.value},
                        "fullDescription": {"text": v.message},
                        "defaultConfiguration": {
                            "level": "error" if v.severity == EnforcementLevel.STRICT else "warning"
                        },
                    }

        return list(rules.values())

    def _sarif_results(self, report: AuditReport) -> list[dict[str, Any]]:
        """Generate SARIF results."""
        results = []

        for detection in report.results:
            for v in detection.violations:
                result = {
                    "ruleId": v.rule_id,
                    "level": "error" if v.severity == EnforcementLevel.STRICT else "warning",
                    "message": {"text": v.message},
                    "locations": [
                        {
                            "physicalLocation": {
                                "artifactLocation": {"uri": v.file_path},
                                "region": {
                                    "startLine": v.line_number or 1,
                                },
                            }
                        }
                    ],
                }

                if v.fix_suggestion:
                    result["fixes"] = [
                        {
                            "description": {"text": v.fix_suggestion},
                        }
                    ]

                results.append(result)

        return results

    def save_report(
        self,
        results: list[DetectionResult],
        output_path: str | Path,
        format: ReportFormat | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Generate and save report to file.

        Args:
            results: Detection results
            output_path: Where to save
            format: Output format (inferred from extension if None)
            metadata: Additional metadata
        """
        path = Path(output_path)

        # Infer format from extension if not specified
        if format is None:
            ext = path.suffix.lower()
            format = {
                ".txt": ReportFormat.TEXT,
                ".json": ReportFormat.JSON,
                ".md": ReportFormat.MARKDOWN,
                ".sarif": ReportFormat.SARIF,
            }.get(ext, ReportFormat.TEXT)

        report = self.generate(results, format, metadata)
        path.write_text(report)

        self._logger.info(
            "Report saved",
            path=str(path),
            format=format.value,
        )
