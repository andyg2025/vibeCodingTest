"""Code Analyzer using tree-sitter for AST analysis.

Provides semantic analysis of Python code to detect:
- Environment variable access
- Dependency injection patterns
- Function signatures and decorators
- Import relationships
- Side effect patterns
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()


class AnalysisCategory(str, Enum):
    """Categories of code analysis."""

    IMPORTS = "imports"
    ENV_ACCESS = "env_access"
    DECORATORS = "decorators"
    FUNCTION_CALLS = "function_calls"
    CLASS_DEFINITIONS = "class_definitions"
    FUNCTION_DEFINITIONS = "function_definitions"
    DEPENDENCIES = "dependencies"  # FastAPI Depends() calls
    SIDE_EFFECTS = "side_effects"


@dataclass
class CodeLocation:
    """Location of code in a file."""

    file_path: str
    line: int
    column: int
    end_line: int | None = None
    end_column: int | None = None

    def __str__(self) -> str:
        return f"{self.file_path}:{self.line}:{self.column}"


@dataclass
class AnalysisResult:
    """Result of analyzing a code element."""

    category: AnalysisCategory
    name: str
    location: CodeLocation
    details: dict[str, Any] = field(default_factory=dict)
    raw_text: str = ""


@dataclass
class FileAnalysis:
    """Complete analysis of a single file."""

    file_path: str
    results: list[AnalysisResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def get_by_category(self, category: AnalysisCategory) -> list[AnalysisResult]:
        """Get all results of a specific category."""
        return [r for r in self.results if r.category == category]

    @property
    def imports(self) -> list[AnalysisResult]:
        return self.get_by_category(AnalysisCategory.IMPORTS)

    @property
    def env_accesses(self) -> list[AnalysisResult]:
        return self.get_by_category(AnalysisCategory.ENV_ACCESS)

    @property
    def decorators(self) -> list[AnalysisResult]:
        return self.get_by_category(AnalysisCategory.DECORATORS)

    @property
    def function_calls(self) -> list[AnalysisResult]:
        return self.get_by_category(AnalysisCategory.FUNCTION_CALLS)

    @property
    def dependencies(self) -> list[AnalysisResult]:
        return self.get_by_category(AnalysisCategory.DEPENDENCIES)


class CodeAnalyzer:
    """Analyzes Python code using tree-sitter for AST parsing."""

    def __init__(self):
        self._logger = logger.bind(component="CodeAnalyzer")
        self._parser = None
        self._language = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazy initialization of tree-sitter."""
        if self._initialized:
            return

        try:
            import tree_sitter_python as tspython
            from tree_sitter import Language, Parser

            self._language = Language(tspython.language())
            self._parser = Parser(self._language)
            self._initialized = True
            self._logger.info("Tree-sitter initialized")
        except Exception as e:
            self._logger.error("Failed to initialize tree-sitter", error=str(e))
            raise RuntimeError(f"tree-sitter initialization failed: {e}")

    def analyze_code(self, code: str, file_path: str = "<string>") -> FileAnalysis:
        """Analyze Python code and extract semantic information.

        Args:
            code: Python source code
            file_path: Path for error reporting

        Returns:
            FileAnalysis with all detected patterns
        """
        self._ensure_initialized()

        analysis = FileAnalysis(file_path=file_path)

        try:
            tree = self._parser.parse(bytes(code, "utf-8"))
            root = tree.root_node

            # Run all analyzers
            self._analyze_imports(root, code, analysis)
            self._analyze_env_access(root, code, analysis)
            self._analyze_decorators(root, code, analysis)
            self._analyze_function_defs(root, code, analysis)
            self._analyze_class_defs(root, code, analysis)
            self._analyze_dependencies(root, code, analysis)
            self._analyze_function_calls(root, code, analysis)

        except Exception as e:
            analysis.errors.append(f"Parse error: {e}")
            self._logger.error("Code analysis failed", file=file_path, error=str(e))

        return analysis

    def analyze_file(self, file_path: str | Path) -> FileAnalysis:
        """Analyze a Python file."""
        path = Path(file_path)
        if not path.exists():
            return FileAnalysis(
                file_path=str(path),
                errors=[f"File not found: {path}"],
            )

        code = path.read_text()
        return self.analyze_code(code, str(path))

    def _analyze_imports(
        self, root, code: str, analysis: FileAnalysis
    ) -> None:
        """Extract import statements."""
        # import x
        for node in self._find_nodes(root, "import_statement"):
            names = self._get_child_text(node, "dotted_name", code)
            analysis.results.append(AnalysisResult(
                category=AnalysisCategory.IMPORTS,
                name=names,
                location=self._node_location(node, analysis.file_path),
                details={"type": "import", "module": names},
                raw_text=self._node_text(node, code),
            ))

        # from x import y
        for node in self._find_nodes(root, "import_from_statement"):
            module = self._get_child_text(node, "dotted_name", code)
            # Get imported names
            imported = []
            for child in node.children:
                if child.type == "dotted_name" and child != node.child_by_field_name("module"):
                    imported.append(self._node_text(child, code))
                elif child.type == "aliased_import":
                    name_node = child.child_by_field_name("name")
                    if name_node:
                        imported.append(self._node_text(name_node, code))

            analysis.results.append(AnalysisResult(
                category=AnalysisCategory.IMPORTS,
                name=module,
                location=self._node_location(node, analysis.file_path),
                details={
                    "type": "from_import",
                    "module": module,
                    "names": imported,
                },
                raw_text=self._node_text(node, code),
            ))

    def _analyze_env_access(
        self, root, code: str, analysis: FileAnalysis
    ) -> None:
        """Detect environment variable access patterns."""
        # Patterns: os.getenv(), os.environ[], os.environ.get()
        for node in self._find_nodes(root, "call"):
            func_text = ""
            func_node = node.child_by_field_name("function")
            if func_node:
                func_text = self._node_text(func_node, code)

            # os.getenv("VAR") or os.environ.get("VAR")
            if func_text in ("os.getenv", "os.environ.get"):
                args_node = node.child_by_field_name("arguments")
                if args_node and args_node.child_count > 0:
                    first_arg = args_node.children[1] if args_node.child_count > 1 else None
                    if first_arg and first_arg.type == "string":
                        var_name = self._node_text(first_arg, code).strip("\"'")
                        analysis.results.append(AnalysisResult(
                            category=AnalysisCategory.ENV_ACCESS,
                            name=var_name,
                            location=self._node_location(node, analysis.file_path),
                            details={"pattern": func_text, "var_name": var_name},
                            raw_text=self._node_text(node, code),
                        ))

        # os.environ["VAR"]
        for node in self._find_nodes(root, "subscript"):
            value_node = node.child_by_field_name("value")
            if value_node:
                value_text = self._node_text(value_node, code)
                if value_text == "os.environ":
                    subscript_node = node.child_by_field_name("subscript")
                    if subscript_node and subscript_node.type == "string":
                        var_name = self._node_text(subscript_node, code).strip("\"'")
                        analysis.results.append(AnalysisResult(
                            category=AnalysisCategory.ENV_ACCESS,
                            name=var_name,
                            location=self._node_location(node, analysis.file_path),
                            details={"pattern": "os.environ[]", "var_name": var_name},
                            raw_text=self._node_text(node, code),
                        ))

    def _analyze_decorators(
        self, root, code: str, analysis: FileAnalysis
    ) -> None:
        """Extract decorator usage."""
        for node in self._find_nodes(root, "decorator"):
            decorator_text = self._node_text(node, code).lstrip("@")
            # Get the decorated function/class name
            parent = node.parent
            decorated_name = ""
            if parent:
                name_node = parent.child_by_field_name("name")
                if name_node:
                    decorated_name = self._node_text(name_node, code)

            analysis.results.append(AnalysisResult(
                category=AnalysisCategory.DECORATORS,
                name=decorator_text,
                location=self._node_location(node, analysis.file_path),
                details={
                    "decorator": decorator_text,
                    "decorates": decorated_name,
                },
                raw_text=self._node_text(node, code),
            ))

    def _analyze_function_defs(
        self, root, code: str, analysis: FileAnalysis
    ) -> None:
        """Extract function definitions with parameters."""
        for node in self._find_nodes(root, "function_definition"):
            name_node = node.child_by_field_name("name")
            params_node = node.child_by_field_name("parameters")
            return_node = node.child_by_field_name("return_type")

            name = self._node_text(name_node, code) if name_node else ""
            params = self._parse_parameters(params_node, code) if params_node else []
            return_type = self._node_text(return_node, code) if return_node else ""

            # Check if async
            is_async = any(
                child.type == "async" for child in (node.parent.children if node.parent else [])
            )

            analysis.results.append(AnalysisResult(
                category=AnalysisCategory.FUNCTION_DEFINITIONS,
                name=name,
                location=self._node_location(node, analysis.file_path),
                details={
                    "name": name,
                    "parameters": params,
                    "return_type": return_type,
                    "is_async": is_async,
                },
                raw_text=self._node_text(node, code)[:200],  # Truncate
            ))

    def _analyze_class_defs(
        self, root, code: str, analysis: FileAnalysis
    ) -> None:
        """Extract class definitions."""
        for node in self._find_nodes(root, "class_definition"):
            name_node = node.child_by_field_name("name")
            name = self._node_text(name_node, code) if name_node else ""

            # Get base classes
            bases = []
            superclasses_node = node.child_by_field_name("superclasses")
            if superclasses_node:
                for child in superclasses_node.children:
                    if child.type not in ("(", ")", ","):
                        bases.append(self._node_text(child, code))

            analysis.results.append(AnalysisResult(
                category=AnalysisCategory.CLASS_DEFINITIONS,
                name=name,
                location=self._node_location(node, analysis.file_path),
                details={
                    "name": name,
                    "bases": bases,
                },
                raw_text=self._node_text(node, code)[:200],
            ))

    def _analyze_dependencies(
        self, root, code: str, analysis: FileAnalysis
    ) -> None:
        """Detect FastAPI Depends() patterns."""
        for node in self._find_nodes(root, "call"):
            func_node = node.child_by_field_name("function")
            if func_node:
                func_text = self._node_text(func_node, code)
                if func_text == "Depends":
                    args_node = node.child_by_field_name("arguments")
                    dep_func = ""
                    if args_node and args_node.child_count > 1:
                        first_arg = args_node.children[1]
                        dep_func = self._node_text(first_arg, code)

                    analysis.results.append(AnalysisResult(
                        category=AnalysisCategory.DEPENDENCIES,
                        name=dep_func,
                        location=self._node_location(node, analysis.file_path),
                        details={
                            "dependency_function": dep_func,
                        },
                        raw_text=self._node_text(node, code),
                    ))

    def _analyze_function_calls(
        self, root, code: str, analysis: FileAnalysis
    ) -> None:
        """Extract significant function calls (db operations, etc)."""
        # Patterns that indicate side effects
        side_effect_patterns = [
            "session.commit", "session.add", "session.delete",
            "db.commit", "db.add", "db.delete", "db.execute",
            "redis.set", "redis.delete", "redis.publish",
            "send_email", "send_notification",
        ]

        for node in self._find_nodes(root, "call"):
            func_node = node.child_by_field_name("function")
            if func_node:
                func_text = self._node_text(func_node, code)

                # Check for side effect patterns
                is_side_effect = any(p in func_text for p in side_effect_patterns)

                if is_side_effect:
                    analysis.results.append(AnalysisResult(
                        category=AnalysisCategory.SIDE_EFFECTS,
                        name=func_text,
                        location=self._node_location(node, analysis.file_path),
                        details={
                            "function": func_text,
                            "is_side_effect": True,
                        },
                        raw_text=self._node_text(node, code),
                    ))

    def _parse_parameters(self, params_node, code: str) -> list[dict[str, Any]]:
        """Parse function parameters."""
        params = []
        for child in params_node.children:
            if child.type in ("identifier", "typed_parameter", "default_parameter", "typed_default_parameter"):
                param: dict[str, Any] = {"name": "", "type": None, "default": None}

                if child.type == "identifier":
                    param["name"] = self._node_text(child, code)
                elif child.type == "typed_parameter":
                    name_node = child.child_by_field_name("name") or child.children[0]
                    type_node = child.child_by_field_name("type")
                    param["name"] = self._node_text(name_node, code) if name_node else ""
                    param["type"] = self._node_text(type_node, code) if type_node else None
                elif child.type in ("default_parameter", "typed_default_parameter"):
                    name_node = child.child_by_field_name("name")
                    param["name"] = self._node_text(name_node, code) if name_node else ""
                    type_node = child.child_by_field_name("type")
                    if type_node:
                        param["type"] = self._node_text(type_node, code)
                    value_node = child.child_by_field_name("value")
                    if value_node:
                        param["default"] = self._node_text(value_node, code)

                if param["name"] and param["name"] not in ("self", "cls"):
                    params.append(param)

        return params

    def _find_nodes(self, root, node_type: str):
        """Recursively find all nodes of a given type."""
        nodes = []
        if root.type == node_type:
            nodes.append(root)
        for child in root.children:
            nodes.extend(self._find_nodes(child, node_type))
        return nodes

    def _node_text(self, node, code: str) -> str:
        """Get the text of a node."""
        if node is None:
            return ""
        return code[node.start_byte:node.end_byte]

    def _get_child_text(self, node, child_type: str, code: str) -> str:
        """Get text of first child of given type."""
        for child in node.children:
            if child.type == child_type:
                return self._node_text(child, code)
        return ""

    def _node_location(self, node, file_path: str) -> CodeLocation:
        """Get location of a node."""
        return CodeLocation(
            file_path=file_path,
            line=node.start_point[0] + 1,  # 1-indexed
            column=node.start_point[1],
            end_line=node.end_point[0] + 1,
            end_column=node.end_point[1],
        )
