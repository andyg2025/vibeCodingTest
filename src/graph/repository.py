"""Graph Repository for Node and Edge Operations.

Provides high-level CRUD operations for graph nodes and edges,
including specialized queries for magic dependency traversal.
"""

from typing import Any, TypeVar

import structlog

from src.models.graph_edges import (
    BaseEdge,
    ContainsEdge,
    DependsOnEdge,
    EdgeType,
    ImplementsEdge,
    ImportsEdge,
    InfluencedByEdge,
)
from src.models.graph_nodes import (
    BaseNode,
    FileNode,
    InterfaceNode,
    MagicProtocolNode,
    ModuleNode,
    NodeType,
    ProjectNode,
)

from .client import GraphClient, get_graph_client

logger = structlog.get_logger()

N = TypeVar("N", bound=BaseNode)
E = TypeVar("E", bound=BaseEdge)


class GraphRepository:
    """Repository for graph database operations."""

    def __init__(self, client: GraphClient | None = None):
        self._client = client

    async def _get_client(self) -> GraphClient:
        """Get the graph client, initializing if needed."""
        if self._client is None:
            self._client = await get_graph_client()
        return self._client

    # ==================== Node Operations ====================

    async def create_node(self, node: BaseNode) -> str:
        """Create a node in the graph."""
        client = await self._get_client()
        props = node.to_cypher_properties()
        label = node.node_type.value

        query = f"""
        CREATE (n:{label} $props)
        RETURN n.id AS id
        """

        result = await client.execute_query(query, {"props": props})
        await logger.ainfo("Created node", node_type=label, node_id=node.id)
        return result[0]["id"]

    async def get_node(self, node_id: str, node_type: NodeType | None = None) -> dict[str, Any] | None:
        """Get a node by ID."""
        client = await self._get_client()

        if node_type:
            query = f"""
            MATCH (n:{node_type.value} {{id: $id}})
            RETURN n
            """
        else:
            query = """
            MATCH (n {id: $id})
            RETURN n, labels(n) AS labels
            """

        result = await client.execute_query(query, {"id": node_id})
        return result[0] if result else None

    async def update_node(self, node_id: str, updates: dict[str, Any]) -> bool:
        """Update node properties."""
        client = await self._get_client()

        query = """
        MATCH (n {id: $id})
        SET n += $updates
        RETURN n.id AS id
        """

        result = await client.execute_query(query, {"id": node_id, "updates": updates})
        return len(result) > 0

    async def delete_node(self, node_id: str) -> bool:
        """Delete a node and its relationships."""
        client = await self._get_client()

        query = """
        MATCH (n {id: $id})
        DETACH DELETE n
        """

        summary = await client.execute_write(query, {"id": node_id})
        return summary["nodes_deleted"] > 0

    # ==================== Edge Operations ====================

    async def create_edge(self, edge: BaseEdge) -> bool:
        """Create a relationship between nodes."""
        client = await self._get_client()
        props = edge.to_cypher_properties()
        edge_type = edge.edge_type.value

        query = f"""
        MATCH (source {{id: $source_id}})
        MATCH (target {{id: $target_id}})
        CREATE (source)-[r:{edge_type} $props]->(target)
        RETURN type(r) AS rel_type
        """

        result = await client.execute_query(
            query,
            {
                "source_id": edge.source_id,
                "target_id": edge.target_id,
                "props": props,
            },
        )

        if result:
            await logger.ainfo(
                "Created edge",
                edge_type=edge_type,
                source=edge.source_id,
                target=edge.target_id,
            )
        return len(result) > 0

    async def get_edges(
        self,
        source_id: str | None = None,
        target_id: str | None = None,
        edge_type: EdgeType | None = None,
    ) -> list[dict[str, Any]]:
        """Query edges with optional filters."""
        client = await self._get_client()

        conditions = []
        params: dict[str, Any] = {}

        if source_id:
            conditions.append("source.id = $source_id")
            params["source_id"] = source_id
        if target_id:
            conditions.append("target.id = $target_id")
            params["target_id"] = target_id

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        rel_type = f":{edge_type.value}" if edge_type else ""

        query = f"""
        MATCH (source)-[r{rel_type}]->(target)
        {where_clause}
        RETURN source.id AS source_id, target.id AS target_id,
               type(r) AS edge_type, properties(r) AS props
        """

        return await client.execute_query(query, params)

    # ==================== Magic Dependency Queries ====================

    async def get_magic_influences(self, file_id: str) -> list[dict[str, Any]]:
        """Get all magic protocols that influence a file.

        This is a core query for the DFS context aggregation.
        """
        client = await self._get_client()

        query = """
        MATCH (f:File {id: $file_id})-[r:INFLUENCED_BY]->(mp:MagicProtocol)
        RETURN mp, r
        ORDER BY r.priority ASC
        """

        return await client.execute_query(query, {"file_id": file_id})

    async def get_ancestor_context(self, node_id: str, max_depth: int = 5) -> list[dict[str, Any]]:
        """Get ancestor chain for context aggregation.

        Traverses CONTAINS relationships upward to collect
        parent context for DFS generation.
        """
        client = await self._get_client()

        # Use a fixed max depth since Neo4j doesn't allow parameters in path length
        # Filter by depth in the WHERE clause instead
        query = """
        MATCH path = (n {id: $node_id})<-[:CONTAINS*1..10]-(ancestor)
        WHERE length(path) <= $max_depth
        RETURN ancestor, length(path) AS depth
        ORDER BY depth ASC
        """

        return await client.execute_query(
            query, {"node_id": node_id, "max_depth": max_depth}
        )

    async def get_dependencies_ordered(self, node_id: str) -> list[dict[str, Any]]:
        """Get dependencies in topological order for DFS traversal.

        Returns child nodes that must be generated before this node.
        """
        client = await self._get_client()

        query = """
        MATCH (n {id: $node_id})-[:IMPORTS|DEPENDS_ON]->(dep)
        RETURN dep.id AS dep_id, dep.name AS dep_name, labels(dep) AS labels
        """

        return await client.execute_query(query, {"node_id": node_id})

    async def detect_cycles(self) -> list[list[str]]:
        """Detect circular dependencies using Tarjan's algorithm.

        Returns list of strongly connected components with size > 1.
        """
        client = await self._get_client()

        # Use Neo4j's built-in cycle detection
        query = """
        CALL gds.cycles.stream('project_graph')
        YIELD path
        RETURN [node IN nodes(path) | node.id] AS cycle
        """

        try:
            result = await client.execute_query(query)
            return [r["cycle"] for r in result]
        except Exception:
            # GDS not installed, use simpler approach
            query = """
            MATCH path = (n)-[:IMPORTS*2..10]->(n)
            RETURN [node IN nodes(path) | node.id] AS cycle
            LIMIT 100
            """
            result = await client.execute_query(query)
            return [r["cycle"] for r in result]

    async def get_downstream_impact(
        self, changed_node_id: str, max_depth: int = 3
    ) -> list[dict[str, Any]]:
        """Get all nodes affected by a change (for incremental updates).

        Traverses import dependencies to find all files that may need
        regeneration when a file changes.
        """
        client = await self._get_client()

        query = """
        MATCH path = (changed {id: $node_id})<-[:IMPORTS*1..10]-(downstream)
        WHERE length(path) <= $max_depth
        RETURN downstream.id AS node_id, downstream.name AS name,
               length(path) AS distance
        ORDER BY distance ASC
        """

        return await client.execute_query(
            query, {"node_id": changed_node_id, "max_depth": max_depth}
        )

    # ==================== Schema Initialization ====================

    async def init_schema(self) -> None:
        """Initialize graph schema with constraints and indexes."""
        client = await self._get_client()

        constraints = [
            "CREATE CONSTRAINT project_id IF NOT EXISTS FOR (p:Project) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT module_id IF NOT EXISTS FOR (m:Module) REQUIRE m.id IS UNIQUE",
            "CREATE CONSTRAINT file_id IF NOT EXISTS FOR (f:File) REQUIRE f.id IS UNIQUE",
            "CREATE CONSTRAINT magic_id IF NOT EXISTS FOR (mp:MagicProtocol) REQUIRE mp.id IS UNIQUE",
            "CREATE CONSTRAINT interface_id IF NOT EXISTS FOR (i:Interface) REQUIRE i.id IS UNIQUE",
        ]

        indexes = [
            "CREATE INDEX file_path IF NOT EXISTS FOR (f:File) ON (f.path)",
            "CREATE INDEX module_layer IF NOT EXISTS FOR (m:Module) ON (m.layer)",
            "CREATE INDEX magic_type IF NOT EXISTS FOR (mp:MagicProtocol) ON (mp.protocol_type)",
        ]

        for stmt in constraints + indexes:
            try:
                await client.execute_write(stmt)
            except Exception as e:
                await logger.awarning("Schema statement failed", statement=stmt, error=str(e))

        await logger.ainfo("Graph schema initialized")

    async def clear_all(self) -> dict[str, int]:
        """Clear all nodes and relationships (use with caution!)."""
        client = await self._get_client()

        result = await client.execute_write("MATCH (n) DETACH DELETE n")
        await logger.awarning("Cleared all graph data", **result)
        return result
