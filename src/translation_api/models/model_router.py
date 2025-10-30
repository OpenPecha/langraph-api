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
    # Anthropic - use exact model IDs
    CLAUDE_3_5_SONNET_20241022 = "claude-3-5-sonnet-20241022"
    CLAUDE_3_7_SONNET_20250219 = "claude-3-7-sonnet-20250219"
    CLAUDE_SONNET_4_20250514 = "claude-sonnet-4-20250514"
    CLAUDE_3_5_HAIKU_20241022 = "claude-3-5-haiku-20241022"
    CLAUDE_3_OPUS_20240229 = "claude-3-opus-20240229"
    CLAUDE_SONNET_4_5 = "claude-sonnet-4-5"
    # OpenAI
    GPT4 = "gpt-4"
    GPT4_TURBO = "gpt-4-turbo"
    GPT35_TURBO = "gpt-3.5-turbo"
    # Google
    GEMINI_2_5_PRO = "gemini-2.5-pro"
    GEMINI_PRO_VISION = "gemini-pro-vision"
    GEMINI_2_5_FLASH = "gemini-2.5-flash"


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
        
        if model_name in [
            "claude-3-5-sonnet-20241022",
            "claude-3-7-sonnet-20250219",
            "claude-sonnet-4-20250514",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
            "claude-sonnet-4-5",
        ]:
            return self._create_anthropic_model(model_name, default_configs, **kwargs)
        
        elif model_name in ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"]:
            return self._create_openai_model(model_name, default_configs, **kwargs)
        
        elif model_name in [
            "gemini-2.5-pro",
            "gemini-pro",
            "gemini-pro-vision",
            "gemini-2.5-flash",
            "gemini-2.5-flash-thinking",
        ]:
            return self._create_gemini_model(model_name, default_configs, **kwargs)
        
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    
    def _create_anthropic_model(self, model_name: str, default_configs: dict, **kwargs) -> ChatAnthropic:
        """Create an Anthropic (Claude) model instance."""
        # Allow per-call API key override via kwargs['api_key']
        user_api_key = kwargs.pop("api_key", None)
        api_key = user_api_key or self.settings.anthropic_api_key
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for Claude models")

        # Map model names to Anthropic model identifiers
        model_mapping = {
            "claude-3-5-sonnet-20241022": "claude-3-5-sonnet-20241022",
            "claude-3-7-sonnet-20250219": "claude-3-7-sonnet-20250219",
            "claude-sonnet-4-20250514": "claude-sonnet-4-20250514",
            "claude-3-5-haiku-20241022": "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229": "claude-3-opus-20240229",
            "claude-sonnet-4-5": "claude-sonnet-4-20250514",  # ← ADD THIS (use the actual Anthropic API model ID)
        }

        # Expect exact Anthropic model IDs passed through
        model_id = model_mapping.get(model_name)
        if not model_id:
            raise ValueError(f"Unsupported Anthropic model: {model_name}")

        return ChatAnthropic(
            anthropic_api_key=api_key,
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
        # Allow per-call API key override via kwargs['api_key']
        user_api_key = kwargs.pop("api_key", None)
        api_key = user_api_key or self.settings.gemini_api_key
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required for Gemini models")
        
        provider_model_name = model_name
        if model_name == "gemini-2.5-flash-thinking":
            # Alias to the actual provider model id
            provider_model_name = "gemini-2.5-flash"

        # Determine desired max_output_tokens (prefer generation_config override)
        user_gc = kwargs.get("generation_config") or {}
        max_out = user_gc.get("max_output_tokens", kwargs.get("max_tokens", 16000))
        base_model = ChatGoogleGenerativeAI(
            google_api_key=api_key,
            model=provider_model_name,
            temperature=default_configs["temperature"],
            max_output_tokens=max_out,
            **{k: v for k, v in kwargs.items() if k not in ["temperature", "max_tokens"]}
        )
        # Default generation_config with thinking control per model variant
        # Baseline: prefer JSON for non-structured calls and set output cap
        base_gc = {"response_mime_type": "application/json", "max_output_tokens": max_out, **user_gc}
        # Thinking budget logic
        if model_name == "gemini-2.5-flash":
            # Plain flash: no thinking
            thinking_gc = {"thinking_config": {"thinking_budget": 0}}
        elif model_name == "gemini-2.5-flash-thinking":
            # Virtual model: flash with thinking (12k budget)
            thinking_gc = {"thinking_config": {"thinking_budget": 12000}}
        elif model_name in ["gemini-2.5-pro", "gemini-pro", "gemini-pro-vision"]:
            # Pro variants: enable thinking with 12k budget
            thinking_gc = {"thinking_config": {"thinking_budget": 12000}}
        else:
            # Default: enable thinking with 12k budget
            thinking_gc = {"thinking_config": {"thinking_budget": 12000}}

        default_generation_config = {**base_gc, **thinking_gc}
        return _GeminiModelWrapper(base_model, default_generation_config)
    
    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about available models based on configured API keys.
        
        Returns:
            Dictionary of available models and their capabilities
        """
        available = {}
        
        if self.settings.anthropic_api_key:
            available.update({
                "claude-3-5-sonnet-20241022": {
                    "provider": "Anthropic",
                    "description": "Claude 3.5 Sonnet (2024-10-22)",
                    "capabilities": ["text", "reasoning", "translation"],
                    "context_window": 200000
                },
                "claude-3-7-sonnet-20250219": {
                    "provider": "Anthropic",
                    "description": "Claude 3.7 Sonnet (2025-02-19)",
                    "capabilities": ["text", "reasoning", "translation"],
                    "context_window": 200000
                },
                "claude-sonnet-4-20250514": {
                    "provider": "Anthropic",
                    "description": "Claude Sonnet 4.0 (2025-05-14)",
                    "capabilities": ["text", "reasoning", "translation"],
                    "context_window": 200000
                },
                "claude-3-5-haiku-20241022": {
                    "provider": "Anthropic", 
                    "description": "Claude 3.5 Haiku (2024-10-22)",
                    "capabilities": ["text", "translation"],
                    "context_window": 200000
                },
                "claude-3-opus-20240229": {
                    "provider": "Anthropic",
                    "description": "Claude 3 Opus (2024-02-29)",
                    "capabilities": ["text", "reasoning", "translation"],
                    "context_window": 200000
                },
                "claude-sonnet-4-5": {  # ← ADD THIS
                    "provider": "Anthropic",
                    "description": "Claude Sonnet 4.5 - Latest model with improved capabilities",
                    "capabilities": ["text", "reasoning", "translation", "advanced-reasoning"],
                    "context_window": 200000  # Update with actual context window
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
                "gemini-2.5-pro": {
                    "provider": "Google",
                    "description": "Gemini 2.5 Pro - Default thinking enabled (budget -1)",
                    "capabilities": ["text", "reasoning", "translation"],
                    "context_window": 30720
                },
                "gemini-2.5-flash-thinking": {
                    "provider": "Google",
                    "description": "Virtual: Gemini 2.5 Flash with thinking (budget -1)",
                    "capabilities": ["text", "reasoning", "translation"],
                    "context_window": 30720
                },
                "gemini-2.5-flash": {
                    "provider": "Google",
                    "description": "Gemini 2.5 Flash - Fast (no thinking; budget 0)",
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

class _GeminiModelWrapper:
    """Thin wrapper to inject generation_config into all Gemini model calls by default.

    This preserves the same interface used in the codebase (invoke, ainvoke, abatch, with_structured_output).
    Per-call generation_config can override the default by passing it explicitly.
    """

    def __init__(self, base_model: BaseChatModel, generation_config: dict):
        self._base_model = base_model
        self._generation_config = generation_config or {}
        # Structured runs (function-calling) cannot combine with response_mime_type=json
        self._structured_generation_config = {
            k: v for k, v in self._generation_config.items() if k != "response_mime_type"
        }

    # Fallback for any attributes/methods not overridden
    def __getattr__(self, item):
        return getattr(self._base_model, item)

    def invoke(self, input, **kwargs):
        # Allow callers to explicitly disable JSON MIME (plain text)
        gc = kwargs.get("generation_config") or {}
        if gc is None:
            gc = {}
        merged = {**self._generation_config, **gc}
        # If caller asked for plain text, remove JSON MIME
        if merged.get("response_mime_type") == "text/plain":
            merged.pop("response_mime_type", None)
        kwargs["generation_config"] = merged
        return self._base_model.invoke(input, **kwargs)

    async def ainvoke(self, input, **kwargs):
        gc = kwargs.get("generation_config") or {}
        if gc is None:
            gc = {}
        merged = {**self._generation_config, **gc}
        if merged.get("response_mime_type") == "text/plain":
            merged.pop("response_mime_type", None)
        kwargs["generation_config"] = merged
        return await self._base_model.ainvoke(input, **kwargs)

    async def abatch(self, inputs, **kwargs):
        gc = kwargs.get("generation_config") or {}
        if gc is None:
            gc = {}
        merged = {**self._generation_config, **gc}
        if merged.get("response_mime_type") == "text/plain":
            merged.pop("response_mime_type", None)
        kwargs["generation_config"] = merged
        return await self._base_model.abatch(inputs, **kwargs)

    def with_structured_output(self, schema):
        structured = self._base_model.with_structured_output(schema)
        # Use a config compatible with function-calling (no response_mime_type)
        return _GeminiModelWrapper(structured, self._structured_generation_config)


# Global model router instance
model_router = ModelRouter()


def get_model_router() -> ModelRouter:
    """Get the global model router instance."""
    return model_router