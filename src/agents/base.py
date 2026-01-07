"""Base Agent class and common utilities for the Multi-Agent system.

Provides the foundation for all specialized agents in the DFS Code Agent Architecture:
- Global Architect Agent
- Governance & Magic-Link Agent
- Logical Designer Agent
- DFS Implementation Agent
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, TypedDict

import structlog
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

logger = structlog.get_logger()


class AgentType(str, Enum):
    """Types of agents in the system."""

    GLOBAL_ARCHITECT = "global_architect"
    GOVERNANCE_MAGIC_LINK = "governance_magic_link"
    LOGICAL_DESIGNER = "logical_designer"
    DFS_IMPLEMENTATION = "dfs_implementation"
    AUDIT = "audit"


class AgentState(TypedDict, total=False):
    """Base state shared across agents in the workflow."""

    messages: list[BaseMessage]
    current_node_id: str | None
    context: dict[str, Any]
    errors: list[str]
    iteration: int
    should_stop: bool


class AgentConfig(BaseModel):
    """Configuration for an agent."""

    name: str
    agent_type: AgentType
    model: str = "claude-sonnet-4-20250514"
    temperature: float = 0.0
    max_tokens: int = 8192
    system_prompt: str = ""


class BaseAgent(ABC):
    """Abstract base class for all agents.

    Provides common functionality:
    - LLM initialization
    - State management
    - Logging
    - Error handling
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self.llm = self._init_llm()
        self._logger = logger.bind(agent=config.name, agent_type=config.agent_type.value)

    def _init_llm(self) -> ChatAnthropic:
        """Initialize the Claude LLM."""
        return ChatAnthropic(
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the system prompt for this agent."""
        pass

    @abstractmethod
    async def process(self, state: AgentState) -> AgentState:
        """Process the current state and return updated state.

        This is the main entry point for agent logic.
        """
        pass

    async def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        """Invoke the LLM with messages."""
        system_prompt = self.get_system_prompt()

        if system_prompt:
            full_messages = [SystemMessage(content=system_prompt)] + messages
        else:
            full_messages = messages

        await self._logger.ainfo(
            "Invoking LLM",
            message_count=len(full_messages),
            model=self.config.model,
        )

        response = await self.llm.ainvoke(full_messages)
        return response

    def create_graph(self) -> StateGraph:
        """Create a LangGraph workflow for this agent.

        Override in subclasses for more complex workflows.
        """
        workflow = StateGraph(AgentState)

        # Add the main processing node
        workflow.add_node("process", self._process_node)

        # Set entry point
        workflow.set_entry_point("process")

        # Add edge to end
        workflow.add_edge("process", END)

        return workflow

    async def _process_node(self, state: AgentState) -> AgentState:
        """LangGraph node wrapper for process method."""
        try:
            return await self.process(state)
        except Exception as e:
            await self._logger.aerror("Agent processing failed", error=str(e))
            errors = state.get("errors", [])
            errors.append(f"{self.config.name}: {str(e)}")
            return {**state, "errors": errors, "should_stop": True}

    async def run(self, initial_state: AgentState) -> AgentState:
        """Run the agent workflow."""
        graph = self.create_graph()
        app = graph.compile()

        await self._logger.ainfo("Starting agent workflow")
        result = await app.ainvoke(initial_state)
        await self._logger.ainfo("Agent workflow completed")

        return result


class AgentMessage(BaseModel):
    """Structured message for inter-agent communication."""

    sender: AgentType
    receiver: AgentType | None = None  # None = broadcast
    content: dict[str, Any]
    correlation_id: str = ""
    priority: int = Field(default=5, ge=1, le=10)


class AgentRegistry:
    """Registry for managing agent instances."""

    def __init__(self):
        self._agents: dict[AgentType, BaseAgent] = {}

    def register(self, agent: BaseAgent) -> None:
        """Register an agent."""
        self._agents[agent.config.agent_type] = agent
        logger.info("Registered agent", agent=agent.config.name)

    def get(self, agent_type: AgentType) -> BaseAgent | None:
        """Get an agent by type."""
        return self._agents.get(agent_type)

    def all(self) -> list[BaseAgent]:
        """Get all registered agents."""
        return list(self._agents.values())


# Global registry
agent_registry = AgentRegistry()
