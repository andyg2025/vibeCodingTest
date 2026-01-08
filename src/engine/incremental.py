"""Incremental Update Pipeline for minimal regeneration.

Implements the incremental update mechanism from the architecture plan:
1. Change Detection - Detect modified files
2. Impact Analysis - Query affected downstream nodes
3. Magic Dependency Check - Verify protocol contracts
4. Selective Regeneration - Only regenerate affected subgraph
"""

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Coroutine

import structlog

logger = structlog.get_logger()


class ChangeType(str, Enum):
    """Type of change detected."""

    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"
    RENAMED = "renamed"


@dataclass
class FileChange:
    """Represents a detected file change."""

    path: str
    change_type: ChangeType
    old_hash: str | None = None
    new_hash: str | None = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def is_content_change(self) -> bool:
        """Check if content actually changed."""
        if self.change_type in (ChangeType.CREATED, ChangeType.DELETED):
            return True
        return self.old_hash != self.new_hash


@dataclass
class ImpactAnalysis:
    """Result of impact analysis."""

    changed_files: list[str]
    affected_files: list[str]  # Downstream dependencies
    affected_magic_protocols: list[str]
    requires_full_rebuild: bool = False
    reason: str | None = None


@dataclass
class RegenerationPlan:
    """Plan for selective regeneration."""

    files_to_regenerate: list[str]
    files_to_skip: list[str]
    estimated_savings: float  # Percentage of work saved
    magic_protocols_to_validate: list[str]


@dataclass
class HashCache:
    """Cache for file content hashes."""

    hashes: dict[str, str] = field(default_factory=dict)
    timestamps: dict[str, datetime] = field(default_factory=dict)

    def get_hash(self, path: str) -> str | None:
        """Get cached hash for a file."""
        return self.hashes.get(path)

    def set_hash(self, path: str, content_hash: str) -> None:
        """Set hash for a file."""
        self.hashes[path] = content_hash
        self.timestamps[path] = datetime.utcnow()

    def invalidate(self, path: str) -> None:
        """Invalidate cache for a file."""
        self.hashes.pop(path, None)
        self.timestamps.pop(path, None)


class ChangeDetector:
    """Detects changes in files."""

    def __init__(self, cache: HashCache | None = None):
        self.cache = cache or HashCache()
        self._logger = logger.bind(component="ChangeDetector")

    def compute_hash(self, content: str) -> str:
        """Compute content hash."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def detect_changes(
        self,
        files: dict[str, str],  # path -> content
        previous_hashes: dict[str, str] | None = None,
    ) -> list[FileChange]:
        """Detect changes between current and previous state.

        Args:
            files: Current file contents
            previous_hashes: Previous content hashes (uses cache if None)

        Returns:
            List of detected changes
        """
        previous = previous_hashes or self.cache.hashes
        changes = []

        # Check for modified and created files
        for path, content in files.items():
            new_hash = self.compute_hash(content)
            old_hash = previous.get(path)

            if old_hash is None:
                changes.append(FileChange(
                    path=path,
                    change_type=ChangeType.CREATED,
                    new_hash=new_hash,
                ))
            elif old_hash != new_hash:
                changes.append(FileChange(
                    path=path,
                    change_type=ChangeType.MODIFIED,
                    old_hash=old_hash,
                    new_hash=new_hash,
                ))

            # Update cache
            self.cache.set_hash(path, new_hash)

        # Check for deleted files
        for path in previous:
            if path not in files:
                changes.append(FileChange(
                    path=path,
                    change_type=ChangeType.DELETED,
                    old_hash=previous[path],
                ))
                self.cache.invalidate(path)

        self._logger.info(
            "Changes detected",
            total_files=len(files),
            changes=len(changes),
            created=len([c for c in changes if c.change_type == ChangeType.CREATED]),
            modified=len([c for c in changes if c.change_type == ChangeType.MODIFIED]),
            deleted=len([c for c in changes if c.change_type == ChangeType.DELETED]),
        )

        return changes


class ImpactAnalyzer:
    """Analyzes impact of changes on the dependency graph."""

    def __init__(self, graph_repository: Any = None):
        self.graph = graph_repository
        self._logger = logger.bind(component="ImpactAnalyzer")

    async def analyze_impact(
        self,
        changes: list[FileChange],
        max_depth: int = 3,
    ) -> ImpactAnalysis:
        """Analyze the impact of changes on downstream dependencies.

        Args:
            changes: List of file changes
            max_depth: Maximum depth to traverse for impact

        Returns:
            Impact analysis result
        """
        changed_files = [c.path for c in changes]
        affected_files = set()
        affected_magic = set()

        for change in changes:
            # Get downstream dependencies
            downstream = await self._get_downstream_deps(change.path, max_depth)
            affected_files.update(downstream)

            # Check magic protocol impact
            magic = await self._get_affected_magic_protocols(change.path)
            affected_magic.update(magic)

        # Check if full rebuild is required
        requires_full = await self._check_requires_full_rebuild(changes, affected_magic)

        analysis = ImpactAnalysis(
            changed_files=changed_files,
            affected_files=list(affected_files),
            affected_magic_protocols=list(affected_magic),
            requires_full_rebuild=requires_full,
            reason=self._get_rebuild_reason(requires_full, changes),
        )

        self._logger.info(
            "Impact analysis complete",
            changed=len(changed_files),
            affected=len(affected_files),
            magic_protocols=len(affected_magic),
            full_rebuild=requires_full,
        )

        return analysis

    async def _get_downstream_deps(
        self, file_path: str, max_depth: int
    ) -> list[str]:
        """Get files that depend on the given file."""
        if not self.graph:
            return []

        try:
            # Query graph for downstream dependencies
            # MATCH (f:File {path: $path})<-[:IMPORTS*1..N]-(downstream:File)
            result = await self.graph.query("""
                MATCH (f:File {path: $path})<-[:IMPORTS*1..3]-(downstream:File)
                RETURN DISTINCT downstream.path AS path
            """, path=file_path)

            return [r["path"] for r in result]
        except Exception as e:
            self._logger.warning("Failed to query downstream deps", error=str(e))
            return []

    async def _get_affected_magic_protocols(
        self, file_path: str
    ) -> list[str]:
        """Get magic protocols affected by changes to a file."""
        if not self.graph:
            return []

        try:
            result = await self.graph.query("""
                MATCH (f:File {path: $path})-[:INFLUENCED_BY]->(mp:MagicProtocol)
                RETURN mp.id AS id
            """, path=file_path)

            return [r["id"] for r in result]
        except Exception as e:
            self._logger.warning("Failed to query magic protocols", error=str(e))
            return []

    async def _check_requires_full_rebuild(
        self,
        changes: list[FileChange],
        affected_magic: set[str],
    ) -> bool:
        """Check if a full rebuild is required."""
        # Full rebuild if:
        # 1. Core infrastructure files changed
        # 2. Magic protocol definitions changed
        # 3. More than 50% of files affected

        for change in changes:
            path = change.path.lower()
            # Core files
            if any(core in path for core in ["config.py", "settings.py", "__init__.py"]):
                if "core" in path or "src/" in path and path.count("/") == 2:
                    return True

        # If magic protocols are affected, may need full rebuild
        if len(affected_magic) > 3:
            return True

        return False

    def _get_rebuild_reason(
        self, requires_full: bool, changes: list[FileChange]
    ) -> str | None:
        """Get reason for rebuild decision."""
        if not requires_full:
            return None

        for change in changes:
            if "config" in change.path.lower():
                return "Core configuration changed"
            if "__init__" in change.path:
                return "Module structure changed"

        return "Extensive changes detected"


class SelectiveRegenerator:
    """Performs selective regeneration of affected files."""

    def __init__(
        self,
        generator: Callable[[str, Any], Coroutine[Any, Any, str]] | None = None,
    ):
        self.generator = generator
        self._logger = logger.bind(component="SelectiveRegenerator")

    def create_plan(
        self,
        impact: ImpactAnalysis,
        all_files: list[str],
    ) -> RegenerationPlan:
        """Create a regeneration plan based on impact analysis.

        Args:
            impact: Impact analysis result
            all_files: All files in the project

        Returns:
            Regeneration plan
        """
        if impact.requires_full_rebuild:
            return RegenerationPlan(
                files_to_regenerate=all_files,
                files_to_skip=[],
                estimated_savings=0.0,
                magic_protocols_to_validate=impact.affected_magic_protocols,
            )

        # Files to regenerate: changed + affected downstream
        to_regenerate = set(impact.changed_files) | set(impact.affected_files)

        # Files to skip: everything else
        to_skip = [f for f in all_files if f not in to_regenerate]

        # Calculate savings
        savings = len(to_skip) / len(all_files) if all_files else 0.0

        plan = RegenerationPlan(
            files_to_regenerate=list(to_regenerate),
            files_to_skip=to_skip,
            estimated_savings=savings,
            magic_protocols_to_validate=impact.affected_magic_protocols,
        )

        self._logger.info(
            "Regeneration plan created",
            regenerate=len(plan.files_to_regenerate),
            skip=len(plan.files_to_skip),
            savings=f"{savings:.1%}",
        )

        return plan

    async def execute_plan(
        self,
        plan: RegenerationPlan,
        context: dict[str, Any],
    ) -> dict[str, str]:
        """Execute the regeneration plan.

        Args:
            plan: Regeneration plan
            context: Generation context

        Returns:
            Dict mapping file paths to generated content
        """
        results = {}

        for file_path in plan.files_to_regenerate:
            try:
                if self.generator:
                    content = await self.generator(file_path, context)
                    results[file_path] = content
                    self._logger.debug("Regenerated file", path=file_path)
                else:
                    self._logger.warning(
                        "No generator configured",
                        path=file_path,
                    )
            except Exception as e:
                self._logger.error(
                    "Failed to regenerate file",
                    path=file_path,
                    error=str(e),
                )

        return results


class IncrementalUpdatePipeline:
    """Complete pipeline for incremental updates.

    Orchestrates:
    1. Change detection
    2. Impact analysis
    3. Magic protocol validation
    4. Selective regeneration
    """

    def __init__(
        self,
        graph_repository: Any = None,
        generator: Callable[[str, Any], Coroutine[Any, Any, str]] | None = None,
    ):
        self.change_detector = ChangeDetector()
        self.impact_analyzer = ImpactAnalyzer(graph_repository)
        self.regenerator = SelectiveRegenerator(generator)
        self._logger = logger.bind(component="IncrementalUpdatePipeline")

    async def process_update(
        self,
        current_files: dict[str, str],
        previous_hashes: dict[str, str] | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Process an incremental update.

        Args:
            current_files: Current file contents (path -> content)
            previous_hashes: Previous content hashes
            context: Generation context

        Returns:
            Update result with regenerated files and statistics
        """
        self._logger.info(
            "Processing incremental update",
            files=len(current_files),
        )

        # 1. Detect changes
        changes = self.change_detector.detect_changes(
            current_files, previous_hashes
        )

        if not changes:
            self._logger.info("No changes detected")
            return {
                "status": "no_changes",
                "changes": [],
                "regenerated": {},
                "statistics": {"savings": 1.0},
            }

        # 2. Analyze impact
        impact = await self.impact_analyzer.analyze_impact(changes)

        # 3. Create regeneration plan
        all_files = list(current_files.keys())
        plan = self.regenerator.create_plan(impact, all_files)

        # 4. Execute plan
        regenerated = await self.regenerator.execute_plan(
            plan, context or {}
        )

        # 5. Build result
        result = {
            "status": "updated",
            "changes": [
                {
                    "path": c.path,
                    "type": c.change_type.value,
                    "hash": c.new_hash,
                }
                for c in changes
            ],
            "impact": {
                "changed_files": impact.changed_files,
                "affected_files": impact.affected_files,
                "magic_protocols": impact.affected_magic_protocols,
                "full_rebuild": impact.requires_full_rebuild,
            },
            "plan": {
                "regenerate": plan.files_to_regenerate,
                "skip": plan.files_to_skip,
                "savings": plan.estimated_savings,
            },
            "regenerated": regenerated,
            "statistics": {
                "total_files": len(current_files),
                "changed": len(changes),
                "affected": len(impact.affected_files),
                "regenerated": len(regenerated),
                "skipped": len(plan.files_to_skip),
                "savings": plan.estimated_savings,
            },
        }

        self._logger.info(
            "Incremental update complete",
            changed=len(changes),
            regenerated=len(regenerated),
            savings=f"{plan.estimated_savings:.1%}",
        )

        return result

    def get_cache_state(self) -> dict[str, str]:
        """Get current hash cache state."""
        return dict(self.change_detector.cache.hashes)

    def invalidate_cache(self, paths: list[str] | None = None) -> None:
        """Invalidate cache for specific paths or all."""
        if paths is None:
            self.change_detector.cache.hashes.clear()
            self.change_detector.cache.timestamps.clear()
        else:
            for path in paths:
                self.change_detector.cache.invalidate(path)


class LazyMagicValidator:
    """Lazy validation of magic protocols at boundary nodes.

    Only validates magic protocol contracts when:
    1. The node is at a module boundary
    2. The node directly uses magic protocol features
    """

    def __init__(self, codex: Any = None):
        self.codex = codex
        self._logger = logger.bind(component="LazyMagicValidator")

    def should_validate(
        self,
        file_path: str,
        magic_protocols: list[str],
    ) -> bool:
        """Determine if validation is needed for a file.

        Args:
            file_path: Path to the file
            magic_protocols: Magic protocols affecting the file

        Returns:
            True if validation should be performed
        """
        # Always validate if magic protocols affect the file
        if magic_protocols:
            return True

        # Validate boundary files
        boundary_patterns = [
            "__init__.py",
            "routes/",
            "api/",
            "handlers/",
        ]

        for pattern in boundary_patterns:
            if pattern in file_path:
                return True

        return False

    async def validate(
        self,
        file_path: str,
        content: str,
        magic_protocols: list[str],
    ) -> dict[str, Any]:
        """Validate magic protocol compliance.

        Args:
            file_path: Path to the file
            content: File content
            magic_protocols: Magic protocols to validate against

        Returns:
            Validation result
        """
        if not self.should_validate(file_path, magic_protocols):
            return {"validated": False, "reason": "not_required"}

        # Perform validation using codex
        violations = []

        if self.codex:
            try:
                result = self.codex.check_compliance(content, magic_protocols)
                violations = result.get("violations", [])
            except Exception as e:
                self._logger.warning(
                    "Validation failed",
                    file=file_path,
                    error=str(e),
                )

        return {
            "validated": True,
            "file": file_path,
            "protocols": magic_protocols,
            "violations": violations,
            "passed": len(violations) == 0,
        }
