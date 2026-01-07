"""Magic Dependency Detector.

Uses AST analysis (tree-sitter) to detect magic dependencies in Python code.
This is used to:
1. Analyze existing code to discover magic patterns
2. Validate generated code against the Codex
3. Build influence graphs from actual code

Detectable patterns:
- os.getenv / os.environ usage
- FastAPI Depends() calls
- Decorator applications
- SQLAlchemy relationship definitions
- Pydantic BaseSettings subclasses
- @app.on_event decorators
- BackgroundTasks usage
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import structlog

try:
    import tree_sitter_python as tspython
    from tree_sitter import Language, Parser, Node
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False

logger = structlog.get_logger()


class MagicPatternType(str, Enum):
    """Types of magic patterns that can be detected."""

    ENV_ACCESS = "env_access"
    DEPENDENCY_INJECTION = "dependency_injection"
    DECORATOR = "decorator"
    ORM_RELATIONSHIP = "orm_relationship"
    LIFECYCLE_HOOK = "lifecycle_hook"
    BACKGROUND_TASK = "background_task"
    MIDDLEWARE = "middleware"
    SETTINGS_CLASS = "settings_class"


@dataclass
class DetectedMagic:
    """A detected magic dependency pattern."""

    pattern_type: MagicPatternType
    name: str
    file_path: str
    line_number: int
    column: int
    code_snippet: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern_type": self.pattern_type.value,
            "name": self.name,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "column": self.column,
            "code_snippet": self.code_snippet,
            "details": self.details,
        }


class MagicDetector:
    """Detects magic dependencies in Python source code."""

    def __init__(self):
        if TREE_SITTER_AVAILABLE:
            self.parser = Parser(Language(tspython.language()))
        else:
            self.parser = None
            logger.warning("tree-sitter not available, using regex fallback")

        # Patterns for regex-based detection (fallback)
        self._patterns = {
            MagicPatternType.ENV_ACCESS: [
                r'os\.getenv\s*\(\s*["\'](\w+)["\']',
                r'os\.environ\s*\[\s*["\'](\w+)["\']',
                r'os\.environ\.get\s*\(\s*["\'](\w+)["\']',
            ],
            MagicPatternType.DEPENDENCY_INJECTION: [
                r'Depends\s*\(\s*(\w+)',
                r'=\s*Depends\s*\(',
            ],
            MagicPatternType.LIFECYCLE_HOOK: [
                r'@\s*app\.on_event\s*\(\s*["\'](\w+)["\']',
                r'@\s*\w+\.on_event\s*\(',
            ],
            MagicPatternType.BACKGROUND_TASK: [
                r'BackgroundTasks',
                r'background_tasks\s*:\s*BackgroundTasks',
            ],
            MagicPatternType.ORM_RELATIONSHIP: [
                r'relationship\s*\(\s*["\'](\w+)["\']',
                r'=\s*relationship\s*\(',
            ],
            MagicPatternType.SETTINGS_CLASS: [
                r'class\s+(\w+)\s*\(\s*BaseSettings\s*\)',
            ],
            MagicPatternType.MIDDLEWARE: [
                r'@\s*app\.middleware\s*\(',
                r'add_middleware\s*\(',
            ],
        }

    def detect_in_file(self, file_path: str | Path) -> list[DetectedMagic]:
        """Detect magic dependencies in a single file."""
        file_path = Path(file_path)
        if not file_path.exists():
            return []

        content = file_path.read_text()
        return self.detect_in_code(content, str(file_path))

    def detect_in_code(self, code: str, file_path: str = "<string>") -> list[DetectedMagic]:
        """Detect magic dependencies in code string."""
        results = []

        if self.parser and TREE_SITTER_AVAILABLE:
            results.extend(self._detect_with_tree_sitter(code, file_path))
        else:
            results.extend(self._detect_with_regex(code, file_path))

        return results

    def _detect_with_tree_sitter(self, code: str, file_path: str) -> list[DetectedMagic]:
        """Use tree-sitter for accurate AST-based detection."""
        results = []
        tree = self.parser.parse(bytes(code, "utf8"))
        root = tree.root_node

        # Detect various patterns
        results.extend(self._detect_env_access_ast(root, code, file_path))
        results.extend(self._detect_depends_ast(root, code, file_path))
        results.extend(self._detect_decorators_ast(root, code, file_path))
        results.extend(self._detect_relationships_ast(root, code, file_path))
        results.extend(self._detect_settings_class_ast(root, code, file_path))

        return results

    def _detect_env_access_ast(
        self, root: "Node", code: str, file_path: str
    ) -> list[DetectedMagic]:
        """Detect os.getenv and os.environ access."""
        results = []

        # Query for call expressions like os.getenv(...)
        def find_env_calls(node: "Node"):
            if node.type == "call":
                func = node.child_by_field_name("function")
                if func and func.type == "attribute":
                    obj = func.child_by_field_name("object")
                    attr = func.child_by_field_name("attribute")
                    if obj and attr:
                        obj_text = code[obj.start_byte:obj.end_byte]
                        attr_text = code[attr.start_byte:attr.end_byte]
                        if obj_text == "os" and attr_text in ("getenv", "environ"):
                            args = node.child_by_field_name("arguments")
                            env_var = ""
                            if args and args.child_count > 1:
                                first_arg = args.child(1)
                                if first_arg and first_arg.type == "string":
                                    env_var = code[first_arg.start_byte:first_arg.end_byte].strip("'\"")

                            results.append(DetectedMagic(
                                pattern_type=MagicPatternType.ENV_ACCESS,
                                name=env_var or "unknown",
                                file_path=file_path,
                                line_number=node.start_point[0] + 1,
                                column=node.start_point[1],
                                code_snippet=code[node.start_byte:node.end_byte],
                                details={"env_var": env_var},
                            ))

            for child in node.children:
                find_env_calls(child)

        find_env_calls(root)
        return results

    def _detect_depends_ast(
        self, root: "Node", code: str, file_path: str
    ) -> list[DetectedMagic]:
        """Detect FastAPI Depends() usage."""
        results = []

        def find_depends(node: "Node"):
            if node.type == "call":
                func = node.child_by_field_name("function")
                if func:
                    func_text = code[func.start_byte:func.end_byte]
                    if func_text == "Depends":
                        args = node.child_by_field_name("arguments")
                        dep_func = ""
                        if args and args.child_count > 1:
                            first_arg = args.child(1)
                            if first_arg:
                                dep_func = code[first_arg.start_byte:first_arg.end_byte]

                        results.append(DetectedMagic(
                            pattern_type=MagicPatternType.DEPENDENCY_INJECTION,
                            name=dep_func or "unknown",
                            file_path=file_path,
                            line_number=node.start_point[0] + 1,
                            column=node.start_point[1],
                            code_snippet=code[node.start_byte:node.end_byte],
                            details={"dependency_function": dep_func},
                        ))

            for child in node.children:
                find_depends(child)

        find_depends(root)
        return results

    def _detect_decorators_ast(
        self, root: "Node", code: str, file_path: str
    ) -> list[DetectedMagic]:
        """Detect decorators that may have side effects."""
        results = []
        magic_decorators = {
            "on_event": MagicPatternType.LIFECYCLE_HOOK,
            "middleware": MagicPatternType.MIDDLEWARE,
            "cached": MagicPatternType.DECORATOR,
            "retry": MagicPatternType.DECORATOR,
            "transaction": MagicPatternType.DECORATOR,
        }

        def find_decorators(node: "Node"):
            if node.type == "decorator":
                decorator_text = code[node.start_byte:node.end_byte]

                for pattern, pattern_type in magic_decorators.items():
                    if pattern in decorator_text:
                        results.append(DetectedMagic(
                            pattern_type=pattern_type,
                            name=pattern,
                            file_path=file_path,
                            line_number=node.start_point[0] + 1,
                            column=node.start_point[1],
                            code_snippet=decorator_text,
                            details={"decorator": decorator_text},
                        ))
                        break

            for child in node.children:
                find_decorators(child)

        find_decorators(root)
        return results

    def _detect_relationships_ast(
        self, root: "Node", code: str, file_path: str
    ) -> list[DetectedMagic]:
        """Detect SQLAlchemy relationship definitions."""
        results = []

        def find_relationships(node: "Node"):
            if node.type == "call":
                func = node.child_by_field_name("function")
                if func:
                    func_text = code[func.start_byte:func.end_byte]
                    if func_text == "relationship":
                        args = node.child_by_field_name("arguments")
                        target_model = ""
                        if args and args.child_count > 1:
                            first_arg = args.child(1)
                            if first_arg:
                                target_model = code[first_arg.start_byte:first_arg.end_byte].strip("'\"")

                        results.append(DetectedMagic(
                            pattern_type=MagicPatternType.ORM_RELATIONSHIP,
                            name=target_model or "unknown",
                            file_path=file_path,
                            line_number=node.start_point[0] + 1,
                            column=node.start_point[1],
                            code_snippet=code[node.start_byte:node.end_byte],
                            details={"target_model": target_model},
                        ))

            for child in node.children:
                find_relationships(child)

        find_relationships(root)
        return results

    def _detect_settings_class_ast(
        self, root: "Node", code: str, file_path: str
    ) -> list[DetectedMagic]:
        """Detect Pydantic BaseSettings subclasses."""
        results = []

        def find_settings_classes(node: "Node"):
            if node.type == "class_definition":
                name_node = node.child_by_field_name("name")
                superclasses = node.child_by_field_name("superclasses")

                if superclasses:
                    superclass_text = code[superclasses.start_byte:superclasses.end_byte]
                    if "BaseSettings" in superclass_text:
                        class_name = ""
                        if name_node:
                            class_name = code[name_node.start_byte:name_node.end_byte]

                        # Extract field definitions
                        fields = []
                        body = node.child_by_field_name("body")
                        if body:
                            for child in body.children:
                                if child.type == "expression_statement":
                                    child_text = code[child.start_byte:child.end_byte]
                                    if ":" in child_text:  # Type annotation
                                        field_name = child_text.split(":")[0].strip()
                                        fields.append(field_name)

                        results.append(DetectedMagic(
                            pattern_type=MagicPatternType.SETTINGS_CLASS,
                            name=class_name,
                            file_path=file_path,
                            line_number=node.start_point[0] + 1,
                            column=node.start_point[1],
                            code_snippet=code[node.start_byte:min(node.end_byte, node.start_byte + 200)],
                            details={
                                "class_name": class_name,
                                "fields": fields,
                            },
                        ))

            for child in node.children:
                find_settings_classes(child)

        find_settings_classes(root)
        return results

    def _detect_with_regex(self, code: str, file_path: str) -> list[DetectedMagic]:
        """Fallback regex-based detection."""
        results = []
        lines = code.split("\n")

        for pattern_type, patterns in self._patterns.items():
            for pattern in patterns:
                for i, line in enumerate(lines):
                    for match in re.finditer(pattern, line):
                        name = match.group(1) if match.lastindex else match.group(0)
                        results.append(DetectedMagic(
                            pattern_type=pattern_type,
                            name=name,
                            file_path=file_path,
                            line_number=i + 1,
                            column=match.start(),
                            code_snippet=line.strip(),
                            details={"match": match.group(0)},
                        ))

        return results

    def detect_in_directory(
        self, directory: str | Path, pattern: str = "**/*.py"
    ) -> dict[str, list[DetectedMagic]]:
        """Detect magic dependencies in all Python files in a directory."""
        directory = Path(directory)
        results: dict[str, list[DetectedMagic]] = {}

        for file_path in directory.glob(pattern):
            if "__pycache__" in str(file_path):
                continue
            file_results = self.detect_in_file(file_path)
            if file_results:
                results[str(file_path)] = file_results

        return results

    def summarize_detections(
        self, detections: list[DetectedMagic]
    ) -> dict[str, Any]:
        """Summarize detected magic dependencies."""
        summary = {
            "total_count": len(detections),
            "by_type": {},
            "env_vars": set(),
            "dependencies": set(),
            "decorators": set(),
        }

        for det in detections:
            type_key = det.pattern_type.value
            if type_key not in summary["by_type"]:
                summary["by_type"][type_key] = []
            summary["by_type"][type_key].append(det.name)

            if det.pattern_type == MagicPatternType.ENV_ACCESS:
                summary["env_vars"].add(det.details.get("env_var", det.name))
            elif det.pattern_type == MagicPatternType.DEPENDENCY_INJECTION:
                summary["dependencies"].add(det.details.get("dependency_function", det.name))
            elif det.pattern_type == MagicPatternType.DECORATOR:
                summary["decorators"].add(det.details.get("decorator", det.name))

        # Convert sets to lists for JSON serialization
        summary["env_vars"] = list(summary["env_vars"])
        summary["dependencies"] = list(summary["dependencies"])
        summary["decorators"] = list(summary["decorators"])

        return summary
