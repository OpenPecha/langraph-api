"""Dynamic model routing system for translation API."""

import os
from typing import Optional, Dict, Any
from enum import Enum

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models import BaseChatModel

from ..config import get_settings


class SupportedModel(Enum):
    """Enumeration of supported models."""
    CLAUDE = "claude"
    CLAUDE_SONNET = "claude-sonnet"
    CLAUDE_HAIKU = "claude-haiku"
    CLAUDE_OPUS = "claude-opus"
    GPT4 = "gpt-4"
    GPT4_TURBO = "gpt-4-turbo"
    GPT35_TURBO = "gpt-3.5-turbo"
    GEMINI_PRO = "gemini-pro"
    GEMINI_PRO_VISION = "gemini-pro-vision"


class ModelRouter:
    """Router for dynamically selecting and initializing language models."""
    
    def __init__(self):
        self.settings = get_settings()
        self._model_cache: Dict[str, BaseChatModel] = {}
    
    def get_model(self, model_name: str, **kwargs) -> BaseChatModel:
        """
        Get a language model instance based on the model name.
        
        Args:
            model_name: Name of the model to use
            **kwargs: Additional model configuration parameters
            
        Returns:
            Initialized language model instance
            
        Raises:
            ValueError: If model is not supported or API key is missing
        """
        # Check cache first
        cache_key = f"{model_name}_{hash(str(sorted(kwargs.items())))}"
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]
        
        model = self._create_model(model_name, **kwargs)
        self._model_cache[cache_key] = model
        return model
    
    def _create_model(self, model_name: str, **kwargs) -> BaseChatModel:
        """Create a new model instance."""
        model_name = model_name.lower()
        
        # Default model configurations
        default_configs = {
            "temperature": kwargs.get("temperature", 0.3),
            "max_tokens": kwargs.get("max_tokens", 4000),
        }
        
        if model_name in ["claude", "claude-sonnet", "claude-haiku", "claude-opus"]:
            return self._create_anthropic_model(model_name, default_configs, **kwargs)
        
        elif model_name in ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"]:
            return self._create_openai_model(model_name, default_configs, **kwargs)
        
        elif model_name in ["gemini-pro", "gemini-pro-vision"]:
            return self._create_gemini_model(model_name, default_configs, **kwargs)
        
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    
    def _create_anthropic_model(self, model_name: str, default_configs: dict, **kwargs) -> ChatAnthropic:
        """Create an Anthropic (Claude) model instance."""
        if not self.settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for Claude models")
        
        # Map model names to Anthropic model identifiers
        model_mapping = {
            "claude": "claude-3-5-sonnet-20241022",
            "claude-sonnet": "claude-3-5-sonnet-20241022",
            "claude-haiku": "claude-3-5-haiku-20241022",
            "claude-opus": "claude-3-opus-20240229"
        }
        
        model_id = model_mapping.get(model_name, "claude-3-5-sonnet-20241022")
        
        return ChatAnthropic(
            anthropic_api_key=self.settings.anthropic_api_key,
            model=model_id,
            temperature=default_configs["temperature"],
            max_tokens=default_configs["max_tokens"],
            **{k: v for k, v in kwargs.items() if k not in ["temperature", "max_tokens"]}
        )
    
    def _create_openai_model(self, model_name: str, default_configs: dict, **kwargs) -> ChatOpenAI:
        """Create an OpenAI model instance."""
        if not self.settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI models")
        
        return ChatOpenAI(
            openai_api_key=self.settings.openai_api_key,
            model=model_name,
            temperature=default_configs["temperature"],
            max_tokens=default_configs["max_tokens"],
            **{k: v for k, v in kwargs.items() if k not in ["temperature", "max_tokens"]}
        )
    
    def _create_gemini_model(self, model_name: str, default_configs: dict, **kwargs) -> ChatGoogleGenerativeAI:
        """Create a Google Gemini model instance."""
        if not self.settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required for Gemini models")
        
        return ChatGoogleGenerativeAI(
            google_api_key=self.settings.gemini_api_key,
            model=model_name,
            temperature=default_configs["temperature"],
            max_output_tokens=default_configs["max_tokens"],
            **{k: v for k, v in kwargs.items() if k not in ["temperature", "max_tokens"]}
        )
    
    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about available models based on configured API keys.
        
        Returns:
            Dictionary of available models and their capabilities
        """
        available = {}
        
        if self.settings.anthropic_api_key:
            available.update({
                "claude": {
                    "provider": "Anthropic",
                    "description": "Claude 3.5 Sonnet - Excellent for complex reasoning and translation",
                    "capabilities": ["text", "reasoning", "translation"],
                    "context_window": 200000
                },
                "claude-haiku": {
                    "provider": "Anthropic", 
                    "description": "Claude 3.5 Haiku - Fast and efficient for simpler tasks",
                    "capabilities": ["text", "translation"],
                    "context_window": 200000
                },
                "claude-opus": {
                    "provider": "Anthropic",
                    "description": "Claude 3 Opus - Most capable for complex tasks",
                    "capabilities": ["text", "reasoning", "translation"],
                    "context_window": 200000
                }
            })
        
        if self.settings.openai_api_key:
            available.update({
                "gpt-4": {
                    "provider": "OpenAI",
                    "description": "GPT-4 - High quality reasoning and translation",
                    "capabilities": ["text", "reasoning", "translation"],
                    "context_window": 128000
                },
                "gpt-4-turbo": {
                    "provider": "OpenAI",
                    "description": "GPT-4 Turbo - Faster with large context window",
                    "capabilities": ["text", "reasoning", "translation"],
                    "context_window": 128000
                }
            })
        
        if self.settings.gemini_api_key:
            available.update({
                "gemini-pro": {
                    "provider": "Google",
                    "description": "Gemini Pro - Good for text and reasoning tasks",
                    "capabilities": ["text", "reasoning", "translation"],
                    "context_window": 30720
                }
            })
        
        return available
    
    def validate_model_availability(self, model_name: str) -> bool:
        """
        Check if a model is available based on API key configuration.
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            True if model is available, False otherwise
        """
        available_models = self.get_available_models()
        return model_name.lower() in available_models


# Global model router instance
model_router = ModelRouter()


def get_model_router() -> ModelRouter:
    """Get the global model router instance."""
    return model_router