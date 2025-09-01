"""
Context retrieval node using LanceDB for RAG
"""

import logging
import os
from typing import Dict, Any, List
import lancedb
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ..states import AgentState

logger = logging.getLogger(__name__)


class ContextRetrieverNode:
    """Retrieve relevant context using LanceDB vector storage"""
    
    def __init__(self, openai_api_key: str, lancedb_path: str = "./lancedb_rag"):
        self.openai_api_key = openai_api_key
        self.lancedb_path = lancedb_path
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=openai_api_key
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.db = None
        self.table = None
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize LanceDB connection and table"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(self.lancedb_path, exist_ok=True)
            
            # Connect to LanceDB
            self.db = lancedb.connect(self.lancedb_path)
            
            # Check if table exists, create if not
            table_name = "schema_context"
            if table_name not in self.db.table_names():
                # Create empty table with schema
                self.table = self.db.create_table(
                    table_name,
                    data=[{
                        "id": "init",
                        "text": "initialization",
                        "vector": self.embeddings.embed_query("initialization"),
                        "metadata": {"type": "init"}
                    }]
                )
                logger.info(f"Created new LanceDB table: {table_name}")
            else:
                self.table = self.db.open_table(table_name)
                logger.info(f"Opened existing LanceDB table: {table_name}")
                
        except Exception as e:
            logger.error(f"Failed to initialize LanceDB: {str(e)}")
            self.db = None
            self.table = None
    
    def __call__(self, state: AgentState) -> AgentState:
        """Retrieve relevant context for the user query"""
        
        user_query = state.get('user_query', '')
        database_schema = state.get('database_schema', '')
        
        if not user_query:
            state['relevant_context'] = []
            state['next_action'] = 'query_generator'
            return state
        
        try:
            # First, ensure schema is indexed
            self._index_schema_if_needed(database_schema)
            
            # Retrieve relevant context
            relevant_chunks = self._retrieve_context(user_query, k=5)
            
            state['relevant_context'] = relevant_chunks
            state['next_action'] = 'query_generator'
            
            logger.info(f"Retrieved {len(relevant_chunks)} relevant context chunks")
            
        except Exception as e:
            logger.warning(f"Context retrieval failed: {str(e)}")
            # Continue without context
            state['relevant_context'] = []
            state['next_action'] = 'query_generator'
        
        return state
    
    def _index_schema_if_needed(self, database_schema: str):
        """Index database schema if not already done"""
        
        if not self.table or not database_schema:
            return
        
        try:
            # Check if schema is already indexed (simple check)
            existing_count = len(self.table.search().limit(1).to_list())
            
            if existing_count <= 1:  # Only init record exists
                # Split schema into chunks
                schema_chunks = self._create_schema_chunks(database_schema)
                
                # Create embeddings and store
                documents = []
                for i, chunk in enumerate(schema_chunks):
                    embedding = self.embeddings.embed_query(chunk)
                    documents.append({
                        "id": f"schema_{i}",
                        "text": chunk,
                        "vector": embedding,
                        "metadata": {"type": "schema", "chunk_id": i}
                    })
                
                # Add to table
                if documents:
                    self.table.add(documents)
                    logger.info(f"Indexed {len(documents)} schema chunks")
        
        except Exception as e:
            logger.error(f"Failed to index schema: {str(e)}")
    
    def _create_schema_chunks(self, database_schema: str) -> List[str]:
        """Create meaningful chunks from database schema"""
        
        chunks = []
        
        # Split by tables first
        table_sections = database_schema.split('\nTABLE: ')
        
        for section in table_sections:
            if section.strip():
                # Each table becomes a chunk
                if not section.startswith('TABLE: '):
                    section = 'TABLE: ' + section
                
                chunks.append(section.strip())
        
        # Also create some general chunks using text splitter
        general_chunks = self.text_splitter.split_text(database_schema)
        chunks.extend(general_chunks)
        
        return chunks
    
    def _retrieve_context(self, query: str, k: int = 5) -> List[str]:
        """Retrieve relevant context chunks for the query"""
        
        if not self.table:
            return []
        
        try:
            # Create query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Search for similar chunks
            results = self.table.search(query_embedding).limit(k).to_list()
            
            # Extract text from results
            context_chunks = []
            for result in results:
                if result.get('text') and result['text'] != 'initialization':
                    context_chunks.append(result['text'])
            
            return context_chunks
        
        except Exception as e:
            logger.error(f"Context retrieval search failed: {str(e)}")
            return []