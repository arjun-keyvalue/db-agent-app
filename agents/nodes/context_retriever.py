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
            chunk_size=1000,
            chunk_overlap=200
        )
    

    
    def __call__(self, state: AgentState) -> AgentState:
        """Retrieve relevant context for the user query"""
        
        user_query = state.get('user_query', '')
        
        if not user_query:
            state['relevant_context'] = []
            state['next_action'] = 'query_generator'
            return state
        
        try:
            # Use schema indexer to search for relevant context
            if self.schema_indexer and self.schema_indexer.schema_table:
                results = self.schema_indexer.search_schema(user_query, k=5)
                relevant_chunks = [result.get('text', '') for result in results if result.get('text')]
            else:
                logger.warning("Schema indexer not available, using empty context")
                relevant_chunks = []
            
            state['relevant_context'] = relevant_chunks
            state['next_action'] = 'query_generator'
            
            logger.info(f"Retrieved {len(relevant_chunks)} relevant context chunks")
            
        except Exception as e:
            logger.warning(f"Context retrieval failed: {str(e)}")
            # Continue without context
            state['relevant_context'] = []
            state['next_action'] = 'query_generator'
        
        return state
    
