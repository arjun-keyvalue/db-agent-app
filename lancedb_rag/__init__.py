"""
LanceDB RAG (Retrieval Augmented Generation) module
"""

# Make directory structure
import os

os.makedirs(os.path.dirname(__file__), exist_ok=True)

# Import required to make the module accessible
try:
    from .vector_store import LanceDBVectorStore
    from .embedding_handler import EmbeddingHandler
except ImportError:
    # Handle the case where dependencies aren't available
    pass
