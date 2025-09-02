"""
Context retrieval node using LanceDB for RAG
"""

import logging
from typing import Dict, Any, List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ..states import AgentState

logger = logging.getLogger(__name__)


class ContextRetrieverNode:
    """Retrieve relevant context using LanceDB vector storage"""

    def __init__(self, schema_indexer):
        self.schema_indexer = schema_indexer
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )

    def __call__(self, state: AgentState) -> AgentState:
        """Retrieve relevant context for the user query"""

        user_query = state.get("user_query", "")

        if not user_query:
            state["relevant_context"] = []
            state["next_action"] = "query_generator"
            return state

        try:
            # Use schema indexer to search for relevant context
            relevant_chunks = []

            if self.schema_indexer and self.schema_indexer.db is not None:
                try:
                    results = self.schema_indexer.search_schema(user_query, k=5)
                    relevant_chunks = [
                        result.get("text", "")
                        for result in results
                        if result.get("text")
                    ]
                    logger.info(
                        f"📄 Retrieved {len(relevant_chunks)} context chunks from vector search"
                    )
                except Exception as search_error:
                    logger.warning(f"LanceDB search failed: {search_error}")
                    # Fall back to basic schema retrieval
                    relevant_chunks = self._get_fallback_context(user_query)
            else:
                logger.debug(
                    "Vector search not available, using fallback context retrieval"
                )
                relevant_chunks = self._get_fallback_context(user_query)

            state["relevant_context"] = relevant_chunks
            
            # Also provide structured schema context for backward compatibility
            # Convert chunks to a structured database_schema format
            if relevant_chunks:
                database_schema = self._format_schema_from_chunks(relevant_chunks)
                state["database_schema"] = database_schema
                logger.info(
                    f"📋 Using {len(relevant_chunks)} context chunks for query generation"
                )
            else:
                state["database_schema"] = "Schema information not available via vector search."
                logger.info(
                    "⚠️  No context chunks available - proceeding without schema context"
                )
            
            state["next_action"] = "query_generator"

        except Exception as e:
            logger.warning(f"Context retrieval failed: {str(e)}")
            # Continue without context
            state["relevant_context"] = []
            state["database_schema"] = f"Context retrieval failed: {str(e)}"
            state["next_action"] = "query_generator"

        return state

    def _format_schema_from_chunks(self, chunks: List[str]) -> str:
        """Convert retrieved chunks into a structured database schema format"""
        try:
            # Combine all chunks into a comprehensive schema description
            schema_parts = ["DATABASE SCHEMA (from vector search):\n"]
            
            for i, chunk in enumerate(chunks, 1):
                if chunk.strip():
                    schema_parts.append(f"\n--- Relevant Schema Section {i} ---")
                    schema_parts.append(chunk.strip())
            
            return "\n".join(schema_parts)
        except Exception as e:
            logger.warning(f"Failed to format schema from chunks: {e}")
            return f"Schema chunks available but formatting failed: {', '.join(chunks[:2])}..."

    def _get_fallback_context(self, user_query: str) -> List[str]:
        """Get basic context when LanceDB is not available"""
        try:
            # Simple keyword-based context retrieval
            # This is a basic fallback that provides some database schema context
            fallback_context = [
                "Database schema information:",
                "Available tables and basic structure will be retrieved from the database directly.",
                f"User query: {user_query}",
                "Note: Advanced vector search not available, using basic schema retrieval.",
            ]
            return fallback_context
        except Exception as e:
            logger.warning(f"Fallback context retrieval failed: {e}")
            return []
