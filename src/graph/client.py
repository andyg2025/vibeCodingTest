"""Neo4j Graph Database Client.

Provides connection management and basic operations for the
graph-driven code agent architecture.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import structlog
from neo4j import AsyncDriver, AsyncGraphDatabase, AsyncSession
from pydantic import BaseModel
from pydantic_settings import BaseSettings

logger = structlog.get_logger()


class Neo4jSettings(BaseSettings):
    """Neo4j connection settings."""

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "vibecoding123"
    neo4j_database: str = "neo4j"

    class Config:
        env_prefix = ""
        case_sensitive = False


class GraphClient:
    """Async Neo4j client wrapper with connection pooling."""

    def __init__(self, settings: Neo4jSettings | None = None):
        self.settings = settings or Neo4jSettings()
        self._driver: AsyncDriver | None = None

    async def connect(self) -> None:
        """Establish connection to Neo4j."""
        if self._driver is not None:
            return

        self._driver = AsyncGraphDatabase.driver(
            self.settings.neo4j_uri,
            auth=(self.settings.neo4j_user, self.settings.neo4j_password),
        )

        # Verify connectivity
        async with self._driver.session() as session:
            result = await session.run("RETURN 1 AS n")
            await result.consume()

        await logger.ainfo(
            "Connected to Neo4j",
            uri=self.settings.neo4j_uri,
            database=self.settings.neo4j_database,
        )

    async def close(self) -> None:
        """Close the driver connection."""
        if self._driver:
            await self._driver.close()
            self._driver = None
            await logger.ainfo("Disconnected from Neo4j")

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get a session context manager."""
        if self._driver is None:
            await self.connect()

        assert self._driver is not None
        async with self._driver.session(database=self.settings.neo4j_database) as session:
            yield session

    async def execute_query(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a Cypher query and return results."""
        async with self.session() as session:
            result = await session.run(query, parameters or {})
            records = await result.data()
            return records

    async def execute_write(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute a write query and return summary."""
        async with self.session() as session:
            result = await session.run(query, parameters or {})
            summary = await result.consume()
            return {
                "nodes_created": summary.counters.nodes_created,
                "nodes_deleted": summary.counters.nodes_deleted,
                "relationships_created": summary.counters.relationships_created,
                "relationships_deleted": summary.counters.relationships_deleted,
                "properties_set": summary.counters.properties_set,
            }


class QueryResult(BaseModel):
    """Structured query result."""

    data: list[dict[str, Any]]
    summary: dict[str, Any] | None = None


# Global client instance
_client: GraphClient | None = None


async def get_graph_client() -> GraphClient:
    """Get or create the global graph client."""
    global _client
    if _client is None:
        _client = GraphClient()
        await _client.connect()
    return _client


async def close_graph_client() -> None:
    """Close the global graph client."""
    global _client
    if _client is not None:
        await _client.close()
        _client = None
