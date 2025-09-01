"""
Automatic Schema Indexer for LanceDB
Indexes database schema automatically when connecting to database
"""

import os
import logging
from typing import Dict, Any, List, Optional
import lancedb
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import Config

logger = logging.getLogger(__name__)


class SchemaIndexer:
    """Automatically index database schema to LanceDB on connection"""
    
    def __init__(self, 
                 lancedb_path: str = None,
                 embedding_model: str = None,
                 embedding_provider: str = None,
                 api_key: str = None):
        
        self.lancedb_path = lancedb_path or Config.LANCEDB_PATH
        self.embedding_model = embedding_model or Config.EMBEDDING_MODEL
        self.embedding_provider = embedding_provider or Config.EMBEDDING_PROVIDER
        self.api_key = api_key or Config.get_rag_api_key()
        
        # Initialize embedding function
        self.embeddings = self._initialize_embeddings()
        
        # Initialize LanceDB
        self.db = None
        self.schema_table = None
        self._initialize_lancedb()
        
        # Text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\nTABLE:", "\nCOLUMNS:", "\n", " "]
        )
    
    def _initialize_embeddings(self):
        """Initialize embedding function based on provider"""
        
        if self.embedding_provider.lower() == 'openai':
            if not self.api_key:
                raise ValueError("OpenAI API key required for OpenAI embeddings")
            
            return OpenAIEmbeddings(
                model=self._get_embedding_model_name(),
                api_key=self.api_key
            )
        
        elif self.embedding_provider.lower() == 'huggingface':
            # Use local HuggingFace embeddings (no API key needed)
            model_name = self._get_embedding_model_name()
            return HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': 'cpu'}
            )
        
        elif self.embedding_provider.lower() == 'ollama':
            # Use Ollama embeddings (local, no API key needed)
            try:
                from langchain_ollama import OllamaEmbeddings
            except ImportError:
                # Fallback to community version with warning suppression
                import warnings
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                from langchain_community.embeddings import OllamaEmbeddings
            
            model_name = self._get_embedding_model_name()
            return OllamaEmbeddings(
                model=model_name,
                base_url=os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
            )
        
        else:
            # Default to OpenAI
            logger.warning(f"Unknown embedding provider: {self.embedding_provider}, defaulting to OpenAI")
            return OpenAIEmbeddings(
                model="text-embedding-3-small",
                api_key=self.api_key
            )
    
    def _get_embedding_model_name(self) -> str:
        """Get the appropriate embedding model name"""
        
        if self.embedding_provider.lower() == 'openai':
            # Map common model names to OpenAI embedding models
            model_mapping = {
                'gpt-4o': 'text-embedding-3-large',
                'gpt-4o-mini': 'text-embedding-3-small',
                'gpt-4': 'text-embedding-3-large',
                'gpt-3.5-turbo': 'text-embedding-3-small'
            }
            return model_mapping.get(self.embedding_model, 'text-embedding-3-small')
        
        elif self.embedding_provider.lower() == 'huggingface':
            # Use sentence transformers for local embeddings
            return 'sentence-transformers/all-MiniLM-L6-v2'
        
        elif self.embedding_provider.lower() == 'ollama':
            # Use Nomic text embeddings from Ollama
            return 'nomic-embed-text'
        
        else:
            return 'text-embedding-3-small'
    
    def _initialize_lancedb(self):
        """Initialize LanceDB connection"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(self.lancedb_path, exist_ok=True)
            
            # Connect to LanceDB
            self.db = lancedb.connect(self.lancedb_path)
            
            logger.info(f"Connected to LanceDB at: {self.lancedb_path}")
            
            # Test the connection by listing tables
            table_names = self.db.table_names()
            logger.info(f"LanceDB initialized successfully with {len(table_names)} existing tables")
            
        except Exception as e:
            logger.error(f"Failed to initialize LanceDB: {str(e)}")
            import traceback
            traceback.print_exc()
            self.db = None
    
    def index_database_schema(self, db_connection) -> bool:
        """Index the entire database schema"""
        
        if not self.db:
            logger.error("LanceDB not initialized - attempting to reinitialize...")
            self._initialize_lancedb()
            if not self.db:
                logger.error("Failed to reinitialize LanceDB")
                return False
        
        try:
            # Get all tables
            tables_success, tables_result = db_connection.get_tables()
            if not tables_success:
                logger.error(f"Failed to get tables: {tables_result}")
                return False
            
            # Build comprehensive schema information
            schema_documents = []
            doc_id = 0
            
            for _, table_row in tables_result.iterrows():
                table_name = table_row['table_name']
                table_type = table_row['table_type']
                
                if table_type == 'BASE TABLE':  # Only index actual tables
                    # Get table schema
                    schema_success, schema_result = db_connection.get_table_schema(table_name)
                    if schema_success:
                        # Create comprehensive table document
                        table_doc = self._create_table_document(table_name, schema_result, db_connection)
                        
                        # Split into chunks if too large
                        chunks = self.text_splitter.split_text(table_doc)
                        
                        for chunk in chunks:
                            if chunk.strip():  # Skip empty chunks
                                embedding = self.embeddings.embed_query(chunk)
                                
                                schema_documents.append({
                                    "id": str(doc_id),
                                    "text": chunk,
                                    "vector": embedding,
                                    "table_name": table_name,
                                    "document_type": "schema",
                                    "chunk_index": len(schema_documents)
                                })
                                doc_id += 1
            
            if schema_documents:
                # Create or recreate the schema table
                table_name = "database_schema"
                
                # Drop existing table if it exists
                if table_name in self.db.table_names():
                    self.db.drop_table(table_name)
                
                # Create new table
                self.schema_table = self.db.create_table(
                    table_name,
                    data=schema_documents
                )
                
                logger.info(f"Successfully indexed {len(schema_documents)} schema chunks from {len(tables_result)} tables")
                return True
            else:
                logger.warning("No schema documents created")
                return False
        
        except Exception as e:
            logger.error(f"Failed to index database schema: {str(e)}")
            return False
    
    def _create_table_document(self, table_name: str, schema_result, db_connection) -> str:
        """Create a comprehensive document for a table"""
        
        doc = f"TABLE: {table_name}\n\n"
        doc += "COLUMNS:\n"
        
        # Add column information
        for _, col_row in schema_result.iterrows():
            col_name = col_row['column_name']
            col_type = col_row['data_type']
            nullable = col_row['is_nullable']
            default_val = col_row['column_default']
            
            doc += f"- {col_name}: {col_type}"
            if nullable == 'NO':
                doc += " (NOT NULL)"
            if default_val:
                doc += f" DEFAULT {default_val}"
            doc += "\n"
        
        # Add sample data (first 3 rows)
        try:
            sample_success, sample_data = db_connection.execute_query(f"SELECT * FROM {table_name} LIMIT 3")
            if sample_success and not sample_data.empty:
                doc += f"\nSAMPLE DATA:\n"
                doc += sample_data.to_string(index=False)
        except Exception as e:
            logger.debug(f"Could not get sample data for {table_name}: {str(e)}")
        
        return doc
    
    def search_schema(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant schema information"""
        
        if not self.schema_table:
            logger.warning("Schema table not initialized")
            return []
        
        try:
            # Create query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Search for similar chunks
            results = self.schema_table.search(query_embedding).limit(k).to_list()
            
            return results
        
        except Exception as e:
            logger.error(f"Schema search failed: {str(e)}")
            return []
    
    def get_table_info(self, table_name: str) -> List[Dict[str, Any]]:
        """Get all schema information for a specific table"""
        
        if not self.schema_table:
            return []
        
        try:
            # Filter by table name
            results = self.schema_table.search().where(f"table_name = '{table_name}'").to_list()
            return results
        
        except Exception as e:
            logger.error(f"Failed to get table info for {table_name}: {str(e)}")
            return []
    
    def get_indexing_status(self) -> Dict[str, Any]:
        """Get status of schema indexing"""
        
        status = {
            "lancedb_path": self.lancedb_path,
            "embedding_provider": self.embedding_provider,
            "embedding_model": self._get_embedding_model_name(),
            "db_connected": self.db is not None,
            "schema_indexed": self.schema_table is not None,
            "total_chunks": 0,
            "tables_indexed": []
        }
        
        if self.schema_table:
            try:
                # Get total count
                all_results = self.schema_table.search().limit(10000).to_list()
                status["total_chunks"] = len(all_results)
                
                # Get unique table names
                table_names = set()
                for result in all_results:
                    if result.get('table_name'):
                        table_names.add(result['table_name'])
                
                status["tables_indexed"] = list(table_names)
                
            except Exception as e:
                logger.error(f"Failed to get indexing status: {str(e)}")
        
        return status


# Global schema indexer instance
schema_indexer = None

def get_schema_indexer() -> SchemaIndexer:
    """Get or create global schema indexer instance"""
    global schema_indexer
    
    if schema_indexer is None:
        try:
            schema_indexer = SchemaIndexer()
            if not schema_indexer.db:
                logger.warning("Schema indexer created but LanceDB not initialized")
        except Exception as e:
            logger.error(f"Failed to create schema indexer: {e}")
            # Create a minimal indexer that won't crash
            schema_indexer = SchemaIndexer()
    
    return schema_indexer

def auto_index_on_connection(db_connection) -> bool:
    """Automatically index schema when database connects"""
    try:
        indexer = get_schema_indexer()
        if not indexer.db:
            logger.warning("LanceDB not initialized, skipping auto-indexing")
            return False
        return indexer.index_database_schema(db_connection)
    except Exception as e:
        logger.error(f"Auto-indexing failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False