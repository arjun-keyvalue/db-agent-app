import lancedb
from langchain_community.vectorstores import LanceDB
from typing import List, Dict, Any
from models.embedding_model import SentenceTransformerEmbeddings


class SchemaEmbeddings:
    def __init__(self, api_key, lancedb_uri: str = "./lancedb"):
        self.lancedb_uri = lancedb_uri
        self.embedding_fn = SentenceTransformerEmbeddings(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.lancedb_connection = lancedb.connect(lancedb_uri)
        self.vectorstore = None

    def create_schema_chunks(self, schema_data: List[Dict[str, Any]]) -> List[str]:
        """Create text chunks from schema data"""
        chunks = []
        for table_data in schema_data:
            # Schema chunk
            schema_chunk = f"Table: {table_data['table']}\nSchema: {table_data['schema']}"
            if table_data.get('relationships'):
                schema_chunk += f"\nRelationships: {table_data['relationships']}"
            
            # Sample data chunk
            sample_chunk = f"Table: {table_data['table']}\nSample rows:\n"
            sample_chunk += "\n".join([str(row) for row in table_data['sample_rows']])
            
            chunks.append(schema_chunk + sample_chunk)
        
        return chunks

    def store_embeddings(self, chunks: List[str]) -> None:
        """Store schema chunks in vector database"""
        docs = []
        for i, chunk in enumerate(chunks):
            embedding = self.embedding_fn.embed_query(chunk)
            docs.append({
                "id": str(i),
                "text": chunk,
                "vector": embedding
            })
        
        # Create or overwrite the table
        table = self.lancedb_connection.create_table(
            "schema_chunks",
            data=docs,
            mode="overwrite"
        )
        
        # Initialize vector store
        self.vectorstore = LanceDB(
            connection=self.lancedb_connection,
            table_name="schema_chunks",
            embedding=self.embedding_fn
        )

    def get_relevant_chunks(self, query: str, k: int = 4) -> List[str]:
        """Retrieve relevant schema chunks for a query"""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Call store_embeddings first.")
        
        retrieved_docs = self.vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in retrieved_docs]