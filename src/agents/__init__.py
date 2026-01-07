"""Multi-Agent system for the DFS Code Agent Architecture."""

from .base import (
    AgentConfig,
    AgentMessage,
    AgentRegistry,
    AgentState,
    AgentType,
    BaseAgent,
    agent_registry,
)
from .state import (
    ArchitectureGraph,
    FileSpec,
    MagicProtocolSpec,
    ModuleSpec,
    ProjectSpec,
    WorkflowState,
    create_initial_state,
)
from .architect import GlobalArchitectAgent
from .magic_link import MagicLinkAgent
from .magic_detector import MagicDetector, DetectedMagic, MagicPatternType
from .codex import (
    Codex,
    CodexBuilder,
    CodexRule,
    EnforcementLevel,
    MagicContract,
    Violation,
    ViolationType,
)
from .magic_templates import (
    FastAPITemplates,
    SQLAlchemyTemplates,
    PydanticTemplates,
    RedisTemplates,
    CeleryTemplates,
    get_all_templates,
    get_templates_for_stack,
)
from .orchestrator import MultiAgentOrchestrator, PipelineResult

__all__ = [
    # Base
    "AgentConfig",
    "AgentMessage",
    "AgentRegistry",
    "AgentState",
    "AgentType",
    "BaseAgent",
    "agent_registry",
    # State
    "ArchitectureGraph",
    "FileSpec",
    "MagicProtocolSpec",
    "ModuleSpec",
    "ProjectSpec",
    "WorkflowState",
    "create_initial_state",
    # Agents
    "GlobalArchitectAgent",
    "MagicLinkAgent",
    # Magic Detector
    "MagicDetector",
    "DetectedMagic",
    "MagicPatternType",
    # Codex
    "Codex",
    "CodexBuilder",
    "CodexRule",
    "EnforcementLevel",
    "MagicContract",
    "Violation",
    "ViolationType",
    # Templates
    "FastAPITemplates",
    "SQLAlchemyTemplates",
    "PydanticTemplates",
    "RedisTemplates",
    "CeleryTemplates",
    "get_all_templates",
    "get_templates_for_stack",
    # Orchestrator
    "MultiAgentOrchestrator",
    "PipelineResult",
]
