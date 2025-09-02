"""
Schema indexer for LanceDB RAG
"""

import logging
import json
from typing import Dict, List, Any, Optional
from .vector_store import LanceDBVectorStore
from .embedding_handler import EmbeddingHandler

logger = logging.getLogger(__name__)


class SchemaIndexer:
    """Index database schema in LanceDB for retrieval"""

    def __init__(self, db_path: Optional[str] = None):
        """Initialize the schema indexer"""
        self.vector_store = LanceDBVectorStore(db_path)
        self.embedding_handler = EmbeddingHandler()

    def index_schema(self, schema_info: Dict[str, Any]):
        """Index a database schema"""
        try:
            # Make sure schema table is initialized
            self.vector_store.initialize_schema_table()

            # Process and index each table
            for table_name, table_info in schema_info.get("tables", {}).items():
                self._index_table(table_name, table_info)

            # Index relationships if available
            if "relationships" in schema_info:
                self._index_relationships(schema_info["relationships"])

            logger.info(
                f"Indexed schema with {len(schema_info.get('tables', {}))} tables"
            )

        except Exception as e:
            logger.error(f"Error indexing schema: {str(e)}")
            raise

    def _index_table(self, table_name: str, table_info: Dict[str, Any]):
        """Index a single table"""
        # Create table description
        columns_text = "\n".join(
            [
                f"- {col_name}: {col_info.get('type', 'unknown')} {' (PRIMARY KEY)' if col_info.get('primary_key', False) else ''}"
                for col_name, col_info in table_info.get("columns", {}).items()
            ]
        )

        table_description = f"Table: {table_name}\nColumns:\n{columns_text}"

        # Generate embedding
        embeddings = self.embedding_handler.get_embeddings(table_description)
        if embeddings and len(embeddings) > 0:
            # Add to vector store
            self.vector_store.add_schema_embedding(
                text=table_description,
                embedding=embeddings[0],
                metadata={
                    "type": "table",
                    "name": table_name,
                    "table_info": json.dumps(table_info),
                },
            )

            # Also index each column separately for better retrieval
            for col_name, col_info in table_info.get("columns", {}).items():
                col_description = f"Column {col_name} in table {table_name}: {col_info.get('type', 'unknown')}"
                col_embeddings = self.embedding_handler.get_embeddings(col_description)

                if col_embeddings and len(col_embeddings) > 0:
                    self.vector_store.add_schema_embedding(
                        text=col_description,
                        embedding=col_embeddings[0],
                        metadata={
                            "type": "column",
                            "name": col_name,
                            "table": table_name,
                            "column_info": json.dumps(col_info),
                        },
                    )

    def _index_relationships(self, relationships: List[Dict[str, Any]]):
        """Index relationships between tables"""
        for rel in relationships:
            source_table = rel.get("source_table")
            target_table = rel.get("target_table")
            source_column = rel.get("source_column")
            target_column = rel.get("target_column")
            rel_type = rel.get("type", "foreign_key")

            if source_table and target_table and source_column and target_column:
                rel_description = f"Relationship: {source_table}.{source_column} -> {target_table}.{target_column} ({rel_type})"

                embeddings = self.embedding_handler.get_embeddings(rel_description)
                if embeddings and len(embeddings) > 0:
                    self.vector_store.add_schema_embedding(
                        text=rel_description,
                        embedding=embeddings[0],
                        metadata={
                            "type": "relationship",
                            "source_table": source_table,
                            "target_table": target_table,
                            "source_column": source_column,
                            "target_column": target_column,
                            "relationship_type": rel_type,
                        },
                    )

    def search_schema(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search the schema based on a natural language query"""
        try:
            # Generate embedding for the query
            query_embeddings = self.embedding_handler.get_embeddings(query)

            if not query_embeddings or len(query_embeddings) == 0:
                logger.error("Failed to generate query embeddings")
                return []

            # Search for relevant schema information
            results = self.vector_store.search_similar(query_embeddings[0], limit=limit)

            # Extract and format results
            formatted_results = []
            for result in results:
                formatted_results.append(
                    {
                        "text": result.get("text", ""),
                        "score": float(result.get("_distance", 0.0)),
                        "metadata": result.get("metadata", {}),
                    }
                )

            return formatted_results

        except Exception as e:
            logger.error(f"Error searching schema: {str(e)}")
            return []
