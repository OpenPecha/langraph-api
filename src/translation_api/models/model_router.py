"""Dynamic model routing system for translation API."""

import os
from typing import Optional, Dict, Any, List, Union
from enum import Enum

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models import BaseChatModel
import httpx
import re

from ..config import get_settings


class SupportedModel(Enum):
    """Enumeration of supported models."""
    # Anthropic - use exact model IDs
    CLAUDE_SONNET_4_20250514 = "claude-sonnet-4-20250514"
    CLAUDE_3_5_HAIKU_20241022 = "claude-3-5-haiku-20241022"
    CLAUDE_3_OPUS_20240229 = "claude-3-opus-20240229"
    CLAUDE_HAIKU_4_5_20251001 = "claude-haiku-4-5-20251001"
    CLAUDE_SONNET_4_5_20250929 = "claude-sonnet-4-5-20250929"
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
            "claude-sonnet-4-20250514",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
            "claude-haiku-4-5-20251001",
            "claude-sonnet-4-5-20250929",
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
        
        elif model_name in ["dharamitra"]:
            return self._create_dharamitra_model(model_name, default_configs, **kwargs)
        
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
            "claude-sonnet-4-20250514": "claude-sonnet-4-20250514",
            "claude-3-5-haiku-20241022": "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229": "claude-3-opus-20240229",
            "claude-haiku-4-5-20251001": "claude-haiku-4-5-20251001",
            "claude-sonnet-4-5-20250929": "claude-sonnet-4-5-20250929",
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

    def _create_dharamitra_model(self, model_name: str, default_configs: dict, **kwargs):
        """Create a Dharmamitra chat-translate model wrapper (translation only)."""
        # Allow per-call API key override via kwargs['api_key']
        user_api_key = kwargs.pop("api_key", None)
        token = user_api_key or self.settings.dharmamitra_token
        if not token:
            raise ValueError("DHARMAMITRA_TOKEN is required for 'dharamitra' model")

        base_url = kwargs.pop(
            "base_url",
            "https://dharmamitra.org/api-search/chat-translate/v1/chat/completions",
        )

        # Return a lightweight wrapper exposing invoke/ainvoke compatible with our usage
        return _DharmamitraModelWrapper(token=token, base_url=base_url)
    
    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about available models based on configured API keys.
        
        Returns:
            Dictionary of available models and their capabilities
        """
        available = {}
        
        if self.settings.anthropic_api_key:
            available.update({
             
         
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
                "claude-haiku-4-5-20251001": {  # ← ADD THIS
                    "provider": "Anthropic",
                    "description": "Claude haiku 4.5 - Latest model with improved capabilities",
                    "capabilities": ["text", "reasoning", "translation", "advanced-reasoning"],
                    "context_window": 200000  # Update with actual context window
                },
                "claude-sonnet-4-5-20250929": {
                    "provider": "Anthropic",
                    "description": "Claude Sonnet 4.5 (2025-09-29)",
                    "capabilities": ["text", "reasoning", "translation", "advanced-reasoning"],
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
        # Dharmamitra appears only if token is configured
        if self.settings.dharmamitra_token:
            available.update({
                "dharamitra": {
                    "provider": "Dharmamitra",
                    "description": "Dharmamitra Chat Translate (mitra-base)",
                    "capabilities": ["text", "translation"],
                    "context_window": 0
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
        # Structured runs (function-calling) cannot combine with response_mime_type or thinking_config
        self._structured_generation_config = {
            k: v
            for k, v in self._generation_config.items()
            if k not in ["response_mime_type", "thinking_config"]
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
        # Remove unsupported fields (library may not accept thinking_config)
        if "thinking_config" in merged:
            merged.pop("thinking_config", None)
        kwargs["generation_config"] = merged
        return self._base_model.invoke(input, **kwargs)

    async def ainvoke(self, input, **kwargs):
        gc = kwargs.get("generation_config") or {}
        if gc is None:
            gc = {}
        merged = {**self._generation_config, **gc}
        if merged.get("response_mime_type") == "text/plain":
            merged.pop("response_mime_type", None)
        if "thinking_config" in merged:
            merged.pop("thinking_config", None)
        kwargs["generation_config"] = merged
        return await self._base_model.ainvoke(input, **kwargs)

    async def abatch(self, inputs, **kwargs):
        gc = kwargs.get("generation_config") or {}
        if gc is None:
            gc = {}
        merged = {**self._generation_config, **gc}
        if merged.get("response_mime_type") == "text/plain":
            merged.pop("response_mime_type", None)
        if "thinking_config" in merged:
            merged.pop("thinking_config", None)
        kwargs["generation_config"] = merged
        return await self._base_model.abatch(inputs, **kwargs)

    def with_structured_output(self, schema):
        structured = self._base_model.with_structured_output(schema)
        # Use a config compatible with function-calling (no response_mime_type)
        return _GeminiModelWrapper(structured, self._structured_generation_config)

    # Streaming helpers (merge generation_config like other calls)
    def astream_events(self, input, **kwargs):
        gc = kwargs.get("generation_config") or {}
        if gc is None:
            gc = {}
        merged = {**self._generation_config, **gc}
        # If caller requested plain text tokens, remove JSON MIME
        if merged.get("response_mime_type") == "text/plain":
            merged.pop("response_mime_type", None)
        if "thinking_config" in merged:
            merged.pop("thinking_config", None)
        kwargs["generation_config"] = merged
        # Return the async generator directly (caller will async-iterate)
        return self._base_model.astream_events(input, **kwargs)

    def astream(self, input, **kwargs):
        gc = kwargs.get("generation_config") or {}
        if gc is None:
            gc = {}
        merged = {**self._generation_config, **gc}
        if merged.get("response_mime_type") == "text/plain":
            merged.pop("response_mime_type", None)
        if "thinking_config" in merged:
            merged.pop("thinking_config", None)
        kwargs["generation_config"] = merged
        # Return the async generator directly
        return self._base_model.astream(input, **kwargs)


class _SimpleResponse:
    """Minimal response carrying text content (mimics LLM message content)."""
    def __init__(self, content: str):
        self.content = content


class _DharmamitraModelWrapper:
    """Translation-only wrapper integrating Dharmamitra chat-translate API.

    Exposes invoke/ainvoke to match our translation code paths. Structured outputs
    (with_structured_output, abatch) are not supported and will raise ValueError to
    ensure this model is used only for translation.
    """

    def __init__(self, token: str, base_url: str):
        self._token = token
        self._base_url = base_url

    def _extract_source_and_lang(self, content: str) -> (str, str):
        """Heuristically extract source text and target language from our prompt."""
        # Target language
        lang = "english"
        m_lang = re.search(r"Translate\s+the\s+provided\s+text\s+into\s+([A-Za-z\- ]+?)\s+while", content, re.IGNORECASE)
        if m_lang:
            lang = (m_lang.group(1) or "english").strip().lower()

        # Source text block: between 'SOURCE TEXT:' and 'Translation:'
        src = content
        m_src = re.search(r"SOURCE\s+TEXT:\s*(.*?)\s*Translation:\s*\Z", content, re.IGNORECASE | re.DOTALL)
        if m_src:
            src = m_src.group(1).strip()

        # If batch marker leaked in, try to drop instruction scaffolding
        # Keep it simple; upstream expects raw Tibetan input
        return src, lang or "english"

    def _build_payload(self, source_text: str, target_lang: str, stream: bool = True) -> Dict[str, Any]:
        return {
            "model": "mitra-base",
            "messages": [
                {"role": "user", "content": source_text}
            ],
            "stream": bool(stream),
            "do_grammar": False,
            "input_encoding": "auto",
            "target_lang": (target_lang or "english").lower(),
        }

    def _parse_response_text(self, data: Any, fallback_text: str) -> str:
        # Try common shapes; otherwise fallback to raw text
        try:
            if isinstance(data, dict):
                # OpenAI-like
                choices = data.get("choices")
                if isinstance(choices, list) and choices:
                    msg = choices[0].get("message") if isinstance(choices[0], dict) else None
                    if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                        return msg["content"].strip()
                # Direct content
                if isinstance(data.get("content"), str):
                    return data["content"].strip()
                if isinstance(data.get("text"), str):
                    return data["text"].strip()
        except Exception:
            pass
        return (fallback_text or "").strip()

    def invoke(self, input: Union[str, List[Any]], **kwargs):
        # Accept either string or list of messages with .content
        content = input
        if isinstance(input, list) and input:
            try:
                content = "\n".join([getattr(m, "content", str(m)) for m in input])
            except Exception:
                content = str(input)
        if not isinstance(content, str):
            content = str(content)

        source_text, target_lang = self._extract_source_and_lang(content)
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        payload = self._build_payload(source_text, target_lang, stream=True)
        try:
            with httpx.Client(timeout=None) as client:
                chunks: List[str] = []
                with client.stream("POST", self._base_url, headers=headers, json=payload) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if not line:
                            continue
                        try:
                            s = line.decode("utf-8", errors="ignore") if isinstance(line, (bytes, bytearray)) else str(line)
                            if not s.startswith("data: "):
                                continue
                            payload_str = s[6:].strip()
                            if not payload_str or payload_str == "[DONE]":
                                continue
                            obj = None
                            try:
                                obj = httpx.Response(200, content=payload_str).json()
                            except Exception:
                                # Not JSON; append raw
                                chunks.append(payload_str)
                                continue
                            # Try OpenAI-like delta
                            if isinstance(obj, dict) and isinstance(obj.get("choices"), list):
                                for ch in obj["choices"]:
                                    delta = ch.get("delta") if isinstance(ch, dict) else None
                                    if isinstance(delta, dict) and isinstance(delta.get("content"), str):
                                        chunks.append(delta["content"])
                                    else:
                                        msg = ch.get("message") if isinstance(ch, dict) else None
                                        if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                                            chunks.append(msg["content"])  # full content
                            else:
                                # Generic fallbacks
                                if isinstance(obj.get("content"), str):
                                    chunks.append(obj["content"])
                                elif isinstance(obj.get("text"), str):
                                    chunks.append(obj["text"])
                        except Exception:
                            continue
                final_text = "".join(chunks).strip()
                if not final_text:
                    # Fallback: try a non-stream call
                    non_stream_headers = {k: v for k, v in headers.items() if k != "Accept"}
                    non_stream_payload = self._build_payload(source_text, target_lang, stream=False)
                    resp2 = client.post(self._base_url, headers=non_stream_headers, json=non_stream_payload, timeout=60.0)
                    resp2.raise_for_status()
                    try:
                        data2 = resp2.json()
                        final_text = self._parse_response_text(data2, resp2.text)
                    except Exception:
                        final_text = resp2.text
                return _SimpleResponse(final_text)
        except Exception as e:
            return _SimpleResponse(f"LLM invocation error: {str(e)}")

    async def ainvoke(self, input: Union[str, List[Any]], **kwargs):
        content = input
        if isinstance(input, list) and input:
            try:
                content = "\n".join([getattr(m, "content", str(m)) for m in input])
            except Exception:
                content = str(input)
        if not isinstance(content, str):
            content = str(content)

        source_text, target_lang = self._extract_source_and_lang(content)
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        payload = self._build_payload(source_text, target_lang, stream=True)
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                chunks: List[str] = []
                async with client.stream("POST", self._base_url, headers=headers, json=payload) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        try:
                            s = line
                            if not s.startswith("data: "):
                                continue
                            payload_str = s[6:].strip()
                            if not payload_str or payload_str == "[DONE]":
                                continue
                            obj = None
                            try:
                                obj = httpx.Response(200, content=payload_str).json()
                            except Exception:
                                chunks.append(payload_str)
                                continue
                            if isinstance(obj, dict) and isinstance(obj.get("choices"), list):
                                for ch in obj["choices"]:
                                    delta = ch.get("delta") if isinstance(ch, dict) else None
                                    if isinstance(delta, dict) and isinstance(delta.get("content"), str):
                                        chunks.append(delta["content"])
                                    else:
                                        msg = ch.get("message") if isinstance(ch, dict) else None
                                        if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                                            chunks.append(msg["content"])  # full content
                            else:
                                if isinstance(obj.get("content"), str):
                                    chunks.append(obj["content"])
                                elif isinstance(obj.get("text"), str):
                                    chunks.append(obj["text"])
                        except Exception:
                            continue
                final_text = "".join(chunks).strip()
                if not final_text:
                    # Fallback: try non-stream
                    non_stream_headers = {k: v for k, v in headers.items() if k != "Accept"}
                    non_stream_payload = self._build_payload(source_text, target_lang, stream=False)
                    resp2 = await client.post(self._base_url, headers=non_stream_headers, json=non_stream_payload, timeout=60.0)
                    resp2.raise_for_status()
                    try:
                        data2 = resp2.json()
                        final_text = self._parse_response_text(data2, resp2.text)
                    except Exception:
                        final_text = resp2.text
                return _SimpleResponse(final_text)
        except Exception as e:
            return _SimpleResponse(f"LLM invocation error: {str(e)}")

    def with_structured_output(self, schema):
        # Translation-only: disallow structured outputs explicitly
        raise ValueError("'dharamitra' model supports translation only; structured outputs are not available.")


# Global model router instance
model_router = ModelRouter()


def get_model_router() -> ModelRouter:
    """Get the global model router instance."""
    return model_router