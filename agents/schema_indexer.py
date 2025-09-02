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

    def __init__(
        self,
        lancedb_path: str = None,
        embedding_model: str = None,
        embedding_provider: str = None,
        api_key: str = None,
    ):

        self.lancedb_path = lancedb_path or Config.LANCEDB_PATH
        self.embedding_model = embedding_model or Config.EMBEDDING_MODEL
        self.embedding_provider = embedding_provider or Config.EMBEDDING_PROVIDER
        self.api_key = api_key or Config.get_rag_api_key()

        # Initialize LanceDB first
        self.db = None
        self.schema_table = None
        self._initialize_lancedb()

        # Initialize embedding function
        self.embeddings = None
        if self.db is not None:  # Only initialize embeddings if LanceDB works
            try:
                self.embeddings = self._initialize_embeddings()
                if self.embeddings:
                    # Test embedding to get dimensionality
                    try:
                        test_embedding = self.embeddings.embed_query("test")
                        embedding_dim = len(test_embedding)
                        logger.info(
                            f"✅ {self.embedding_provider} embeddings ready (dim: {embedding_dim})"
                        )
                    except Exception as e:
                        logger.warning(f"Embeddings initialized but test failed: {e}")
                        logger.info(
                            f"✅ {self.embedding_provider} embeddings initialized"
                        )
                else:
                    logger.warning(
                        f"❌ {self.embedding_provider} embeddings failed - schema indexing disabled"
                    )
            except Exception as e:
                logger.error(f"Failed to initialize embeddings: {e}")
                import traceback

                logger.debug(
                    f"Embeddings initialization traceback: {traceback.format_exc()}"
                )
                self.embeddings = None

        # Text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\nTABLE:", "\nCOLUMNS:", "\n", " "],
        )

    def _initialize_embeddings(self):
        """Initialize embedding function based on provider"""

        try:
            if self.embedding_provider.lower() == "openai":
                if not self.api_key:
                    logger.warning(
                        "No OpenAI API key provided, falling back to HuggingFace"
                    )
                    return self._initialize_huggingface_embeddings()

                return OpenAIEmbeddings(
                    model=self._get_embedding_model_name(), api_key=self.api_key
                )

            elif self.embedding_provider.lower() == "huggingface":
                return self._initialize_huggingface_embeddings()

            elif self.embedding_provider.lower() == "ollama":
                return self._initialize_ollama_embeddings()

            else:
                # Default to HuggingFace (free/local)
                logger.warning(
                    f"Unknown embedding provider: {self.embedding_provider}, defaulting to HuggingFace"
                )
                return self._initialize_huggingface_embeddings()

        except Exception as e:
            logger.error(
                f"Failed to initialize {self.embedding_provider} embeddings: {e}"
            )
            # Only fallback to HuggingFace if it was the original provider or if sentence-transformers is available
            if self.embedding_provider.lower() == "huggingface":
                return None  # Already tried HuggingFace, don't retry
            else:
                logger.warning(
                    f"Embedding provider '{self.embedding_provider}' failed. Schema indexing will be disabled."
                )
                logger.info(
                    "To enable schema indexing, ensure your embedding provider is properly configured."
                )
                return None

    def _initialize_huggingface_embeddings(self):
        """Initialize HuggingFace embeddings (fallback)"""
        try:
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            return HuggingFaceEmbeddings(
                model_name=model_name, model_kwargs={"device": "cpu"}
            )
        except ImportError as e:
            if "sentence_transformers" in str(e):
                logger.error(
                    "sentence_transformers package not installed. Schema indexing will be disabled."
                )
                logger.info(
                    "To enable schema indexing, install: pip install sentence-transformers"
                )
                return None
            else:
                raise e

    def _initialize_ollama_embeddings(self):
        """Initialize Ollama embeddings"""
        try:
            from langchain_ollama import OllamaEmbeddings
        except ImportError:
            # Fallback to community version with warning suppression
            import warnings

            warnings.filterwarnings("ignore", category=DeprecationWarning)
            from langchain_community.embeddings import OllamaEmbeddings

        model_name = self._get_embedding_model_name()
        base_url = Config.OLLAMA_BASE_URL

        return OllamaEmbeddings(model=model_name, base_url=base_url)

    def _get_embedding_model_name(self) -> str:
        """Get the appropriate embedding model name"""

        if self.embedding_provider.lower() == "openai":
            # Map common model names to OpenAI embedding models
            model_mapping = {
                "gpt-4o": "text-embedding-3-large",
                "gpt-4o-mini": "text-embedding-3-small",
                "gpt-4": "text-embedding-3-large",
                "gpt-3.5-turbo": "text-embedding-3-small",
            }
            return model_mapping.get(self.embedding_model, "text-embedding-3-small")

        elif self.embedding_provider.lower() == "huggingface":
            # Use sentence transformers for local embeddings
            return "sentence-transformers/all-MiniLM-L6-v2"

        elif self.embedding_provider.lower() == "ollama":
            # Use Nomic text embeddings from Ollama
            return "nomic-embed-text"

        else:
            return "text-embedding-3-small"

    def _initialize_lancedb(self):
        """Initialize LanceDB connection following the official docs pattern"""
        try:
            # Create directory if it doesn't exist with proper permissions
            os.makedirs(self.lancedb_path, exist_ok=True)

            # Verify directory is writable
            test_file = os.path.join(self.lancedb_path, ".test_write")
            try:
                with open(test_file, "w") as f:
                    f.write("test")
                os.remove(test_file)
            except Exception as perm_error:
                logger.error(
                    f"Directory {self.lancedb_path} is not writable: {perm_error}"
                )
                self.db = None
                return

            # Connect to LanceDB - following the basic usage pattern from docs
            self.db = lancedb.connect(self.lancedb_path)

            # Test the connection by listing tables
            table_names = self.db.table_names()
            logger.debug(f"LanceDB connected at: {self.lancedb_path}, tables: {len(table_names)}")

        except Exception as e:
            logger.error(f"❌ Failed to initialize LanceDB: {str(e)}")
            import traceback

            logger.debug(f"LanceDB initialization traceback: {traceback.format_exc()}")
            self.db = None

    def index_database_schema(self, db_connection) -> bool:
        """Index the entire database schema"""

        if self.db is None:
            logger.error("LanceDB not initialized - attempting to reinitialize...")
            self._initialize_lancedb()
            if self.db is None:
                logger.error("Failed to reinitialize LanceDB")
                return False

        if not self.embeddings:
            logger.error("Embeddings not initialized")
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
                table_name = table_row["table_name"]
                table_type = table_row["table_type"]

                if table_type == "BASE TABLE":  # Only index actual tables
                    # Get table schema
                    schema_success, schema_result = db_connection.get_table_schema(
                        table_name
                    )
                    if schema_success:
                        # Create comprehensive table document
                        table_doc = self._create_table_document(
                            table_name, schema_result, db_connection
                        )

                        # Split into chunks if too large
                        chunks = self.text_splitter.split_text(table_doc)

                        for chunk in chunks:
                            if chunk.strip():  # Skip empty chunks
                                try:
                                    embedding = self.embeddings.embed_query(chunk)

                                    schema_documents.append(
                                        {
                                            "id": str(doc_id),
                                            "text": chunk,
                                            "vector": embedding,
                                            "table_name": table_name,
                                            "document_type": "schema",
                                            "chunk_index": len(schema_documents),
                                        }
                                    )
                                    doc_id += 1
                                except Exception as e:
                                    logger.warning(
                                        f"Failed to embed chunk {doc_id}: {e}"
                                    )
                                    continue

            if schema_documents:
                # Create or recreate the schema table - following LanceDB docs pattern
                table_name = "database_schema"

                try:
                    # Create new table with data - using mode="overwrite" as per docs
                    # This automatically handles existing tables
                    self.schema_table = self.db.create_table(
                        table_name, schema_documents, mode="overwrite"
                    )

                    logger.info(
                        f"Successfully indexed {len(schema_documents)} schema chunks from {len(set(doc['table_name'] for doc in schema_documents))} tables"
                    )

                    # Verify table was created
                    if table_name in self.db.table_names():
                        logger.debug(f"Verified table '{table_name}' exists in LanceDB")
                        return True
                    else:
                        logger.error(
                            f"Table '{table_name}' was not created successfully"
                        )
                        return False

                except Exception as e:
                    logger.error(f"Failed to create LanceDB table: {e}")
                    import traceback

                    logger.debug(f"Full traceback: {traceback.format_exc()}")
                    return False
            else:
                logger.warning("No schema documents created")
                return False

        except Exception as e:
            logger.error(f"Failed to index database schema: {str(e)}")
            return False

    def _create_table_document(
        self, table_name: str, schema_result, db_connection
    ) -> str:
        """Create a comprehensive document for a table"""

        doc = f"TABLE: {table_name}\n\n"
        doc += "COLUMNS:\n"

        # Add column information
        for _, col_row in schema_result.iterrows():
            col_name = col_row["column_name"]
            col_type = col_row["data_type"]
            nullable = col_row["is_nullable"]
            default_val = col_row["column_default"]

            doc += f"- {col_name}: {col_type}"
            if nullable == "NO":
                doc += " (NOT NULL)"
            if default_val:
                doc += f" DEFAULT {default_val}"
            doc += "\n"

        # Add sample data (first 3 rows)
        try:
            sample_success, sample_data = db_connection.execute_query(
                f"SELECT * FROM {table_name} LIMIT 3"
            )
            if sample_success and not sample_data.empty:
                doc += f"\nSAMPLE DATA:\n"
                doc += sample_data.to_string(index=False)
        except Exception as e:
            logger.debug(f"Could not get sample data for {table_name}: {str(e)}")

        return doc

    def search_schema(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant schema information"""

        # Try to open the schema table if not already available
        if not self.schema_table and self.db is not None:
            try:
                if "database_schema" in self.db.table_names():
                    self.schema_table = self.db.open_table("database_schema")
                    logger.debug("Opened existing schema table")
                else:
                    logger.warning("No schema table found in LanceDB")
                    return []
            except Exception as e:
                logger.error(f"Failed to open schema table: {e}")
                return []

        if not self.schema_table or not self.embeddings:
            logger.warning("Schema table or embeddings not initialized")
            return []

        try:
            # Create query embedding
            query_embedding = self.embeddings.embed_query(query)

            # Search for similar chunks - following LanceDB docs pattern
            results = self.schema_table.search(query_embedding).limit(k).to_list()

            logger.debug(
                f"Found {len(results)} relevant schema chunks for query: {query}"
            )
            return results

        except Exception as e:
            logger.error(f"Schema search failed: {str(e)}")
            import traceback

            logger.debug(f"Full traceback: {traceback.format_exc()}")
            return []

    def get_table_info(self, table_name: str) -> List[Dict[str, Any]]:
        """Get all schema information for a specific table"""

        if not self.schema_table:
            return []

        try:
            # Filter by table name
            results = (
                self.schema_table.search()
                .where(f"table_name = '{table_name}'")
                .to_list()
            )
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
            "tables_indexed": [],
        }

        if self.schema_table:
            try:
                # Get total count
                all_results = self.schema_table.search().limit(10000).to_list()
                status["total_chunks"] = len(all_results)

                # Get unique table names
                table_names = set()
                for result in all_results:
                    if result.get("table_name"):
                        table_names.add(result["table_name"])

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
            schema_indexer = SchemaIndexer(
                embedding_provider=Config.EMBEDDING_PROVIDER,
                embedding_model=Config.EMBEDDING_MODEL,
            )
            if schema_indexer.db and schema_indexer.embeddings:
                logger.info("Schema indexer created successfully")
            else:
                logger.warning("Schema indexer created but not fully functional")
        except Exception as e:
            logger.error(f"Failed to create schema indexer: {e}")
            # Create a minimal indexer that won't crash
            try:
                schema_indexer = SchemaIndexer()
            except:
                # Last resort - create a dummy indexer
                schema_indexer = type(
                    "DummyIndexer",
                    (),
                    {
                        "db": None,
                        "embeddings": None,
                        "schema_table": None,
                        "index_database_schema": lambda self, db: False,
                        "search_schema": lambda self, query, k=5: [],
                    },
                )()

    return schema_indexer


def auto_index_on_connection(db_connection) -> bool:
    """Automatically index schema when database connects"""
    try:
        # Create a fresh indexer with proper config
        indexer = SchemaIndexer(
            embedding_provider=Config.EMBEDDING_PROVIDER,
            embedding_model=Config.EMBEDDING_MODEL,
        )

        if indexer.db is None:
            logger.error(
                "LanceDB not initialized. Make sure directory exists and permissions are correct."
            )
            # Attempt to create the directory explicitly
            try:
                os.makedirs(indexer.lancedb_path, exist_ok=True)
                logger.info(f"Created directory: {indexer.lancedb_path}")
                # Try to reinitialize
                indexer._initialize_lancedb()
                if indexer.db is None:
                    logger.warning(
                        "LanceDB still not initialized after directory creation, skipping auto-indexing"
                    )
                    return False
            except Exception as dir_error:
                logger.error(f"Failed to create LanceDB directory: {str(dir_error)}")
                return False

        if not indexer.embeddings:
            logger.warning("Embeddings not initialized, skipping auto-indexing")
            return False

        success = indexer.index_database_schema(db_connection)
        if success:
            # Update global indexer
            global schema_indexer
            schema_indexer = indexer

        return success
    except Exception as e:
        logger.error(f"Auto-indexing failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return False
