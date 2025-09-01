"""
Smart LLM client that handles multiple providers and API keys
"""

import os
import logging
import litellm
from typing import Dict, Any, List
from config import Config

logger = logging.getLogger(__name__)


class SmartLLMClient:
    """
    Smart LLM client that automatically configures API keys and handles multiple providers
    """
    
    def __init__(self, model: str = None, provider: str = None, api_key: str = None):
        self.model = model or Config.RAG_MODEL
        self.provider = provider or Config.RAG_PROVIDER
        self.api_key = api_key
        self._setup_api_keys()
        
        logger.info(f"Initialized SmartLLMClient with model: {self.model}, provider: {self.provider}")
    
    def switch_model(self, provider: str = None, model: str = None, api_key: str = None):
        """Switch to a different model/provider"""
        if provider:
            self.provider = provider
        if model:
            self.model = model
        if api_key:
            self.api_key = api_key
        self._setup_api_keys()
        logger.info(f"Switched to model: {self.model}, provider: {self.provider}")
    
    def get_model_name(self) -> str:
        """Get the full model name for LiteLLM"""
        return f"{self.provider}/{self.model}"
    
    def _setup_api_keys(self):
        """Setup API keys for different providers"""
        
        # Set environment variables for LiteLLM
        provider = self.provider.lower()
        
        if provider == 'openai':
            api_key = self.api_key or Config.get_rag_api_key()
            if api_key:
                os.environ['OPENAI_API_KEY'] = api_key
                litellm.openai_key = api_key
        
        elif provider == 'groq':
            api_key = self.api_key or Config.GROQ_API_KEY or Config.get_rag_api_key()
            if api_key:
                os.environ['GROQ_API_KEY'] = api_key
                litellm.groq_key = api_key
        
        elif provider in ['google', 'gemini']:
            api_key = self.api_key or Config.GEMINI_API_KEY or Config.get_rag_api_key()
            if api_key:
                os.environ['GEMINI_API_KEY'] = api_key
                litellm.gemini_key = api_key
        
        elif provider == 'anthropic':
            api_key = self.api_key or Config.ANTHROPIC_API_KEY or Config.get_rag_api_key()
            if api_key:
                os.environ['ANTHROPIC_API_KEY'] = api_key
                litellm.anthropic_key = api_key
        
        elif provider == 'ollama':
            # Ollama doesn't need API keys, but we can set base URL if needed
            base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
            os.environ['OLLAMA_BASE_URL'] = base_url
    
    def completion(self, messages: List[Dict[str, str]], **kwargs) -> Any:
        """
        Make a completion request with automatic error handling and fallbacks
        """
        
        # Default parameters
        default_params = {
            'model': self.model,
            'messages': messages,
            'temperature': kwargs.get('temperature', 0.1),
            'max_tokens': kwargs.get('max_tokens', 1000)
        }
        
        # Add any additional parameters
        for key, value in kwargs.items():
            if key not in default_params:
                default_params[key] = value
        
        try:
            # Make the request
            response = litellm.completion(**default_params)
            return response
            
        except litellm.AuthenticationError as e:
            logger.error(f"Authentication error with {self.provider}: {str(e)}")
            
            # Try to provide helpful error message
            if self.provider == 'openai':
                error_msg = "OpenAI API key is invalid or missing. Please set RAG_API_KEY or OPENAI_API_KEY in your .env file."
            elif self.provider == 'groq':
                error_msg = "Groq API key is invalid or missing. Please set GROQ_API_KEY in your .env file."
            elif self.provider in ['google', 'gemini']:
                error_msg = "Gemini API key is invalid or missing. Please set GEMINI_API_KEY in your .env file."
            elif self.provider == 'anthropic':
                error_msg = "Anthropic API key is invalid or missing. Please set ANTHROPIC_API_KEY in your .env file."
            else:
                error_msg = f"API key for {self.provider} is invalid or missing."
            
            raise Exception(error_msg) from e
            
        except litellm.RateLimitError as e:
            logger.error(f"Rate limit exceeded for {self.provider}: {str(e)}")
            raise Exception(f"Rate limit exceeded for {self.provider}. Please try again later.") from e
            
        except Exception as e:
            logger.error(f"LLM completion failed: {str(e)}")
            raise Exception(f"LLM completion failed: {str(e)}") from e
    
    def get_embedding(self, text: str, model: str = "text-embedding-3-small") -> List[float]:
        """
        Get text embeddings (primarily for OpenAI)
        """
        try:
            if self.provider == 'openai':
                from langchain_openai import OpenAIEmbeddings
                
                api_key = Config.get_rag_api_key()
                if not api_key:
                    raise Exception("OpenAI API key required for embeddings")
                
                embeddings = OpenAIEmbeddings(
                    model=model,
                    api_key=api_key
                )
                return embeddings.embed_query(text)
            else:
                # For non-OpenAI providers, we might need different embedding approaches
                logger.warning(f"Embeddings not directly supported for {self.provider}, falling back to OpenAI")
                
                # Fallback to OpenAI for embeddings
                openai_key = Config.OPENAI_API_KEY
                if openai_key:
                    from langchain_openai import OpenAIEmbeddings
                    embeddings = OpenAIEmbeddings(
                        model=model,
                        api_key=openai_key
                    )
                    return embeddings.embed_query(text)
                else:
                    raise Exception("OpenAI API key required for embeddings when using non-OpenAI providers")
                    
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            raise
    
    @classmethod
    def create_from_config(cls) -> 'SmartLLMClient':
        """Create client from configuration"""
        return cls(Config.RAG_MODEL, Config.RAG_PROVIDER)
    
    def test_connection(self) -> bool:
        """Test if the LLM connection is working"""
        try:
            test_messages = [
                {"role": "user", "content": "Hello, this is a test. Please respond with 'OK'."}
            ]
            
            response = self.completion(test_messages, max_tokens=10)
            
            if response and response.choices:
                logger.info(f"LLM connection test successful for {self.provider}")
                return True
            else:
                logger.error(f"LLM connection test failed for {self.provider}: No response")
                return False
                
        except Exception as e:
            logger.error(f"LLM connection test failed for {self.provider}: {str(e)}")
            return False