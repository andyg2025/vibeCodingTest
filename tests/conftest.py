"""Pytest configuration and shared fixtures."""

import asyncio
import os
from typing import Any, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.codex import Codex, CodexBuilder, CodexRule, EnforcementLevel
from src.agents.state import ProjectSpec, ModuleSpec, FileSpec
from src.models.graph_nodes import (
    ProjectNode,
    ModuleNode,
    FileNode,
    MagicProtocolNode,
    NodeType,
)


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_project_spec() -> ProjectSpec:
    """Create a sample project specification."""
    return ProjectSpec(
        name="test-project",
        description="A test project for integration testing",
        tech_stack=["fastapi", "sqlalchemy", "pydantic"],
        requirements=[
            "User authentication with JWT",
            "CRUD operations for users",
            "Database with SQLAlchemy ORM",
        ],
    )


@pytest.fixture
def sample_modules() -> list[ModuleSpec]:
    """Create sample module specifications."""
    return [
        ModuleSpec(
            name="core",
            description="Core configuration and utilities",
            layer="infrastructure",
            dependencies=[],
        ),
        ModuleSpec(
            name="models",
            description="Database models",
            layer="domain",
            dependencies=["core"],
        ),
        ModuleSpec(
            name="api",
            description="API routes and endpoints",
            layer="application",
            dependencies=["core", "models"],
        ),
    ]


@pytest.fixture
def sample_files() -> list[FileSpec]:
    """Create sample file specifications."""
    return [
        FileSpec(
            path="src/core/config.py",
            description="Application configuration",
            module="core",
            dependencies=[],
        ),
        FileSpec(
            path="src/core/database.py",
            description="Database connection",
            module="core",
            dependencies=["src/core/config.py"],
        ),
        FileSpec(
            path="src/models/user.py",
            description="User model",
            module="models",
            dependencies=["src/core/database.py"],
        ),
        FileSpec(
            path="src/api/deps.py",
            description="Dependency injection",
            module="api",
            dependencies=["src/core/database.py"],
        ),
        FileSpec(
            path="src/api/routes/users.py",
            description="User routes",
            module="api",
            dependencies=["src/models/user.py", "src/api/deps.py"],
        ),
    ]


@pytest.fixture
def sample_magic_protocols() -> list[dict[str, Any]]:
    """Create sample magic protocol definitions."""
    return [
        {
            "id": "magic:auth-middleware",
            "name": "AuthenticationMiddleware",
            "type": "middleware",
            "provides": {"current_user": {"type": "User | None"}},
            "requires": {"env_vars": ["JWT_SECRET"]},
            "side_effects": [{"type": "exception", "raises": "HTTPException(401)"}],
        },
        {
            "id": "magic:db-session",
            "name": "DatabaseSession",
            "type": "di",
            "provides": {"db": {"type": "AsyncSession"}},
            "requires": {"env_vars": ["DATABASE_URL"]},
            "side_effects": [],
        },
        {
            "id": "magic:settings",
            "name": "SettingsProvider",
            "type": "env",
            "provides": {"settings": {"type": "Settings"}},
            "requires": {"env_vars": ["DATABASE_URL", "JWT_SECRET", "DEBUG"]},
            "side_effects": [],
        },
    ]


@pytest.fixture
def sample_codex() -> Codex:
    """Create a sample Codex with rules."""
    builder = CodexBuilder()
    builder.add_standard_rules()

    # Add custom rules for testing
    builder._codex.add_rule(CodexRule(
        rule_id="TEST-001",
        description="All API routes must have authentication",
        enforcement_level=EnforcementLevel.STRICT,
        fix_suggestion="Add 'current_user: User = Depends(get_current_user)' parameter",
    ))

    return builder.build(freeze=True)


@pytest.fixture
def sample_project_node() -> ProjectNode:
    """Create a sample project node."""
    return ProjectNode(
        id="proj:test-project",
        name="test-project",
        tech_stack=["fastapi", "sqlalchemy"],
    )


@pytest.fixture
def sample_module_nodes() -> list[ModuleNode]:
    """Create sample module nodes."""
    return [
        ModuleNode(
            id="mod:core",
            name="core",
            module_type="infrastructure",
            layer=0,
        ),
        ModuleNode(
            id="mod:models",
            name="models",
            module_type="domain",
            layer=1,
        ),
        ModuleNode(
            id="mod:api",
            name="api",
            module_type="application",
            layer=2,
        ),
    ]


@pytest.fixture
def sample_file_nodes() -> list[FileNode]:
    """Create sample file nodes."""
    return [
        FileNode(
            id="file:src/core/config.py",
            path="src/core/config.py",
            language="python",
        ),
        FileNode(
            id="file:src/core/database.py",
            path="src/core/database.py",
            language="python",
        ),
        FileNode(
            id="file:src/models/user.py",
            path="src/models/user.py",
            language="python",
        ),
    ]


@pytest.fixture
def sample_magic_protocol_nodes() -> list[MagicProtocolNode]:
    """Create sample magic protocol nodes."""
    return [
        MagicProtocolNode(
            id="magic:auth-middleware",
            name="AuthenticationMiddleware",
            protocol_type="middleware",
            provides={"current_user": {"type": "User | None"}},
            requires={"env_vars": ["JWT_SECRET"]},
        ),
        MagicProtocolNode(
            id="magic:db-session",
            name="DatabaseSession",
            protocol_type="di",
            provides={"db": {"type": "AsyncSession"}},
            requires={"env_vars": ["DATABASE_URL"]},
        ),
    ]


@pytest.fixture
def sample_code_with_violations() -> str:
    """Sample code that contains violations."""
    return '''"""Routes with violations."""
import os
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

router = APIRouter(prefix="/admin")

# Violation: Undeclared env var
SECRET = os.getenv("UNDECLARED_VAR")

@router.get("/users")
async def get_users(db: Session = Depends(get_db)):
    """Get users - missing auth."""
    return db.query(User).all()
'''


@pytest.fixture
def sample_code_clean() -> str:
    """Sample code that follows best practices."""
    return '''"""Clean routes following best practices."""
from typing import Annotated
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import get_settings
from src.api.deps import get_current_user, get_db

router = APIRouter(prefix="/users", tags=["users"])


@router.get("/me")
async def get_profile(
    current_user: Annotated[dict, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Get current user profile."""
    return {"user": current_user}
'''


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing without API calls."""
    async def _mock_response(prompt: str) -> str:
        return '''"""Generated module."""

def example_function():
    """Example function."""
    return "generated"
'''
    return _mock_response


@pytest.fixture
def mock_graph_repository():
    """Create a mock graph repository."""
    mock = MagicMock()
    mock.get_node = AsyncMock(return_value=None)
    mock.create_node = AsyncMock(return_value="node-id")
    mock.get_dependencies = AsyncMock(return_value=[])
    mock.get_ancestors = AsyncMock(return_value=[])
    mock.get_magic_protocols = AsyncMock(return_value=[])
    return mock
