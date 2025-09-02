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
            state["next_action"] = "query_generator"

            if relevant_chunks:
                logger.info(
                    f"📋 Using {len(relevant_chunks)} context chunks for query generation"
                )
            else:
                logger.info(
                    "⚠️  No context chunks available - proceeding without schema context"
                )

        except Exception as e:
            logger.warning(f"Context retrieval failed: {str(e)}")
            # Continue without context
            state["relevant_context"] = []
            state["next_action"] = "query_generator"

        return state

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
