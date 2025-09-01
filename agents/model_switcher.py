"""
Model Switcher Utility for RAG Agent
Provides easy model switching capabilities with LiteLLM
"""

import os
from typing import Dict, Any, Optional
from .llm_client import SmartLLMClient
import config

class ModelSwitcher:
    """Utility class for switching between different LLM models and providers"""
    
    # Predefined model configurations
    MODELS = {
        "openai": {
            "gpt-4o": {"model": "gpt-4o", "context_window": 128000},
            "gpt-4o-mini": {"model": "gpt-4o-mini", "context_window": 128000},
            "gpt-4-turbo": {"model": "gpt-4-turbo", "context_window": 128000},
            "gpt-3.5-turbo": {"model": "gpt-3.5-turbo", "context_window": 16385}
        },
        "groq": {
            "llama-3.1-70b": {"model": "llama-3.1-70b-versatile", "context_window": 131072},
            "llama-3.1-8b": {"model": "llama-3.1-8b-instant", "context_window": 131072},
            "mixtral-8x7b": {"model": "mixtral-8x7b-32768", "context_window": 32768}
        },
        "anthropic": {
            "claude-3.5-sonnet": {"model": "claude-3-5-sonnet-20241022", "context_window": 200000},
            "claude-3-haiku": {"model": "claude-3-haiku-20240307", "context_window": 200000}
        },
        "gemini": {
            "gemini-pro": {"model": "gemini-pro", "context_window": 32768},
            "gemini-1.5-pro": {"model": "gemini-1.5-pro", "context_window": 2000000}
        },
        "ollama": {
            "llama3.1": {"model": "llama3.1", "context_window": 131072},
            "codellama": {"model": "codellama", "context_window": 16384},
            "mistral": {"model": "mistral", "context_window": 32768}
        }
    }
    
    @classmethod
    def get_available_models(cls) -> Dict[str, Dict[str, Any]]:
        """Get all available models grouped by provider"""
        return cls.MODELS
    
    @classmethod
    def get_model_info(cls, provider: str, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model"""
        return cls.MODELS.get(provider, {}).get(model_name)
    
    @classmethod
    def create_client(cls, provider: str, model_name: str, api_key: str = None) -> SmartLLMClient:
        """Create a new LLM client with specified provider and model"""
        model_info = cls.get_model_info(provider, model_name)
        if not model_info:
            raise ValueError(f"Model {model_name} not found for provider {provider}")
        
        return SmartLLMClient(
            provider=provider,
            model=model_info["model"],
            api_key=api_key
        )
    
    @classmethod
    def switch_rag_model(cls, rag_agent, provider: str, model_name: str, api_key: str = None):
        """Switch the RAG agent to use a different model"""
        model_info = cls.get_model_info(provider, model_name)
        if not model_info:
            raise ValueError(f"Model {model_name} not found for provider {provider}")
        
        # Update the RAG agent's LLM client
        rag_agent.llm_client.switch_model(
            provider=provider,
            model=model_info["model"],
            api_key=api_key
        )
        
        # Update all nodes that use the LLM client
        for node in [rag_agent.intent_detector, rag_agent.query_generator, 
                    rag_agent.corrector, rag_agent.output_formatter]:
            if hasattr(node, 'llm_client'):
                node.llm_client = rag_agent.llm_client
        
        print(f"RAG Agent switched to {provider}/{model_name}")
    
    @classmethod
    def list_models_by_provider(cls, provider: str) -> Dict[str, Any]:
        """List all models for a specific provider"""
        return cls.MODELS.get(provider, {})
    
    @classmethod
    def get_recommended_models(cls) -> Dict[str, str]:
        """Get recommended models for different use cases"""
        return {
            "fast": "groq/llama-3.1-8b",
            "balanced": "openai/gpt-4o-mini", 
            "powerful": "openai/gpt-4o",
            "long_context": "gemini/gemini-1.5-pro",
            "local": "ollama/llama3.1"
        }

# Convenience functions
def switch_to_fast_model(rag_agent, api_key: str = None):
    """Switch to a fast model for quick responses"""
    ModelSwitcher.switch_rag_model(rag_agent, "groq", "llama-3.1-8b", api_key)

def switch_to_powerful_model(rag_agent, api_key: str = None):
    """Switch to a powerful model for complex queries"""
    ModelSwitcher.switch_rag_model(rag_agent, "openai", "gpt-4o", api_key)

def switch_to_local_model(rag_agent):
    """Switch to a local Ollama model"""
    ModelSwitcher.switch_rag_model(rag_agent, "ollama", "llama3.1")