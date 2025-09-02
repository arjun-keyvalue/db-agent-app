"""
Vector store implementation using LanceDB for RAG
"""

import os
import logging
import lancedb
from typing import List, Dict, Any, Optional
from config import Config

logger = logging.getLogger(__name__)


class LanceDBVectorStore:
    """
    Vector store using LanceDB for schema and query embedding storage
    """

    def __init__(self, db_path: str = None):
        """Initialize the LanceDB vector store"""
        self.db_path = db_path or Config.LANCEDB_PATH
        self.connection = None
        self.table = None
        self._connect()

    def _connect(self):
        """Connect to LanceDB"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

            # Connect to LanceDB
            self.connection = lancedb.connect(self.db_path)
            logger.info(f"Connected to LanceDB at {self.db_path}")

        except Exception as e:
            logger.error(f"Failed to connect to LanceDB: {str(e)}")
            raise

    def initialize_schema_table(self):
        """Initialize the schema table if not exists"""
        try:
            table_names = self.connection.table_names()

            if "schema_embeddings" not in table_names:
                # Create schema table with initial schema
                self.connection.create_table(
                    "schema_embeddings",
                    data=[
                        {
                            "id": "sample",
                            "text": "Sample schema entry",
                            "embedding": [0.0] * 768,  # Default embedding dimension
                            "metadata": {},
                        }
                    ],
                    mode="overwrite",
                )
                logger.info("Created schema_embeddings table in LanceDB")

            # Open the table
            self.table = self.connection.open_table("schema_embeddings")

        except Exception as e:
            logger.error(f"Failed to initialize schema table: {str(e)}")
            raise

    def add_schema_embedding(
        self, text: str, embedding: List[float], metadata: Dict[str, Any] = None
    ):
        """Add a schema embedding to the table"""
        try:
            if self.table is None:
                self.initialize_schema_table()

            # Add the embedding
            self.table.add(
                [
                    {
                        "id": f"schema_{len(embedding)}_{hash(text) % 10000}",
                        "text": text,
                        "embedding": embedding,
                        "metadata": metadata or {},
                    }
                ]
            )

            logger.info(f"Added schema embedding for: {text[:50]}...")

        except Exception as e:
            logger.error(f"Failed to add schema embedding: {str(e)}")
            raise

    def search_similar(
        self, query_embedding: List[float], limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for similar schema entries"""
        try:
            if self.table is None:
                self.initialize_schema_table()

            # Search for similar embeddings
            results = self.table.search(query_embedding).limit(limit).to_list()

            logger.info(f"Found {len(results)} similar schema entries")
            return results

        except Exception as e:
            logger.error(f"Failed to search similar embeddings: {str(e)}")
            raise

    def close(self):
        """Close the connection to LanceDB"""
        # LanceDB handles connections internally,
        # but we can set references to None
        self.table = None
        self.connection = None
