from typing import List
from sentence_transformers import SentenceTransformer

class SentenceTransformerEmbeddings:
    """Wrapper to make SentenceTransformer behave like OpenAIEmbeddings."""
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name, device="cpu")

    def embed_query(self, text: str):
        return self.model.encode(text).tolist()

    def embed_documents(self, texts: List[str]):
        return [vec.tolist() for vec in self.model.encode(texts)]