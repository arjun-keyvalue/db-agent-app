"""
Embedding handler for LanceDB RAG
"""

import logging
import numpy as np
from typing import List, Union, Dict, Any
from config import Config

logger = logging.getLogger(__name__)


class EmbeddingHandler:
    """Handle different embedding providers for LanceDB RAG"""

    def __init__(self):
        self.provider = Config.EMBEDDING_PROVIDER.lower()
        self.model = Config.EMBEDDING_MODEL
        self._embedding_function = None
        self._setup_embedding_function()

    def _setup_embedding_function(self):
        """Setup the appropriate embedding function based on configuration"""
        try:
            if self.provider == "ollama":
                self._setup_ollama_embeddings()
            elif self.provider == "huggingface":
                self._setup_huggingface_embeddings()
            elif self.provider == "openai":
                self._setup_openai_embeddings()
            else:
                logger.warning(
                    f"Unknown embedding provider: {self.provider}, falling back to huggingface"
                )
                self.provider = "huggingface"
                self._setup_huggingface_embeddings()
        except Exception as e:
            logger.error(
                f"Failed to setup embedding function: {str(e)}, falling back to simple embeddings"
            )
            self._embedding_function = self._simple_embedding

    def _setup_ollama_embeddings(self):
        """Setup Ollama embeddings"""
        try:
            import requests

            def ollama_embedding(texts: Union[str, List[str]]) -> List[List[float]]:
                if isinstance(texts, str):
                    texts = [texts]

                base_url = Config.OLLAMA_BASE_URL
                embeddings = []

                for text in texts:
                    try:
                        response = requests.post(
                            f"{base_url}/api/embeddings",
                            json={"model": self.model, "prompt": text},
                        )
                        response.raise_for_status()
                        embedding = response.json().get("embedding", [])
                        embeddings.append(embedding)
                    except Exception as e:
                        logger.error(f"Error getting Ollama embedding: {str(e)}")
                        # Return a zero vector as fallback
                        embeddings.append([0.0] * 768)

                return embeddings

            self._embedding_function = ollama_embedding
            logger.info(f"Using Ollama embeddings with model: {self.model}")

        except ImportError:
            logger.warning("Could not import required packages for Ollama embeddings")
            self._embedding_function = self._simple_embedding

    def _setup_huggingface_embeddings(self):
        """Setup HuggingFace embeddings"""
        try:
            from sentence_transformers import SentenceTransformer

            # Default to a good general-purpose model if not specified
            model_name = self.model or "sentence-transformers/all-MiniLM-L6-v2"
            model = SentenceTransformer(model_name)

            def hf_embedding(texts: Union[str, List[str]]) -> List[List[float]]:
                if isinstance(texts, str):
                    texts = [texts]

                try:
                    embeddings = model.encode(texts, convert_to_numpy=True).tolist()
                    return embeddings
                except Exception as e:
                    logger.error(f"Error in HuggingFace embedding: {str(e)}")
                    # Return zero vectors as fallback
                    return [[0.0] * 384] * len(
                        texts
                    )  # 384 is typical for many sentence models

            self._embedding_function = hf_embedding
            logger.info(f"Using HuggingFace embeddings with model: {model_name}")

        except ImportError:
            logger.warning(
                "Could not import sentence-transformers, falling back to simple embeddings"
            )
            self._embedding_function = self._simple_embedding

    def _setup_openai_embeddings(self):
        """Setup OpenAI embeddings"""
        try:
            from openai import OpenAI
            import os

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not found in environment variables")

            client = OpenAI(api_key=api_key)
            model_name = self.model or "text-embedding-3-small"

            def openai_embedding(texts: Union[str, List[str]]) -> List[List[float]]:
                if isinstance(texts, str):
                    texts = [texts]

                try:
                    response = client.embeddings.create(model=model_name, input=texts)
                    return [item.embedding for item in response.data]
                except Exception as e:
                    logger.error(f"Error in OpenAI embedding: {str(e)}")
                    # Return zero vectors as fallback
                    return [[0.0] * 1536] * len(
                        texts
                    )  # OpenAI embeddings are typically 1536-dim

            self._embedding_function = openai_embedding
            logger.info(f"Using OpenAI embeddings with model: {model_name}")

        except ImportError:
            logger.warning(
                "Could not import OpenAI package, falling back to simple embeddings"
            )
            self._embedding_function = self._simple_embedding

    def _simple_embedding(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """Simple fallback embedding function using character hashing"""
        if isinstance(texts, str):
            texts = [texts]

        embeddings = []
        dim = 768  # Standard dimension

        for text in texts:
            # Create a simple deterministic embedding from text
            np.random.seed(hash(text) % 2**32)
            embedding = np.random.rand(dim).tolist()
            embeddings.append(embedding)

        return embeddings

    def get_embeddings(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """Get embeddings for one or more texts"""
        try:
            return self._embedding_function(texts)
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            # Return a simple fallback embedding
            return self._simple_embedding(texts)
