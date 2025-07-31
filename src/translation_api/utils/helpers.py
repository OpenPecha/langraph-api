"""Utility functions for the translation API."""

import re
import time
from typing import List, Dict, Any, Optional
from datetime import datetime


def parse_batch_translation_response(response_text: str, separator: str = "---TRANSLATION_SEPARATOR---") -> List[str]:
    """
    Parse a batch translation response into individual translations.
    
    Args:
        response_text: The full response from the model
        separator: Separator used to split translations
        
    Returns:
        List of individual translated texts
    """
    if separator not in response_text:
        # If no separator found, return the whole response as single translation
        return [response_text.strip()]
    
    translations = response_text.split(separator)
    return [t.strip() for t in translations if t.strip()]


def clean_translation_text(text: str) -> str:
    """
    Clean and normalize translated text.
    
    Args:
        text: Raw translated text
        
    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove common prefixes from model responses
    prefixes_to_remove = [
        "Translation:",
        "Translation 1:",
        "Translation 2:",
        "Translation 3:",
        "Translation 4:",
        "Translation 5:",
        "Here is the translation:",
        "The translation is:",
        "As an expert translator",
        "I'll translate",
        "Here is",
        "TRANSLATION 1:",
        "TRANSLATION 2:",
        "TRANSLATION 3:",
    ]
    
    for prefix in prefixes_to_remove:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
            break
    
    # Remove trailing explanatory text (anything after lines starting with common markers)
    explanatory_markers = [
        "\n\nTranslation note",
        "\n\nNote:",
        "\n\nTranslator's",
        "\n\n*Translation",
        "\n\nDeeper meaning",
        "\n\n[Anmerkung",
        "\n\nWould you like",
    ]
    
    for marker in explanatory_markers:
        if marker in text:
            text = text.split(marker)[0].strip()
    
    return text


def validate_tibetan_text(text: str) -> bool:
    """
    Basic validation for Tibetan text input.
    
    Args:
        text: Text to validate
        
    Returns:
        True if text appears to be Tibetan or contains Tibetan characters
    """
    if not text or not text.strip():
        return False
    
    # Check for Tibetan Unicode range (U+0F00–U+0FFF)
    tibetan_pattern = r'[\u0F00-\u0FFF]'
    has_tibetan = bool(re.search(tibetan_pattern, text))
    
    # For now, accept any non-empty text (could be transliterated or English description)
    return True


def estimate_processing_time(num_texts: int, model_name: str, avg_text_length: int = 500) -> float:
    """
    Estimate processing time for a batch of texts.
    
    Args:
        num_texts: Number of texts to process
        model_name: Model being used
        avg_text_length: Average length of texts in characters
        
    Returns:
        Estimated processing time in seconds
    """
    # Base time per text by model (rough estimates)
    base_times = {
        "claude": 3.0,
        "claude-haiku": 1.5,
        "claude-opus": 5.0,
        "gpt-4": 4.0,
        "gpt-4-turbo": 2.5,
        "gpt-3.5-turbo": 2.0,
        "gemini-pro": 3.0
    }
    
    base_time = base_times.get(model_name.lower(), 3.0)
    
    # Adjust for text length
    length_multiplier = max(0.5, min(2.0, avg_text_length / 500))
    
    # Add batch processing overhead
    batch_overhead = 0.5 * num_texts
    
    return (base_time * length_multiplier * num_texts) + batch_overhead


def create_processing_metadata(
    start_time: float,
    end_time: float,
    num_texts: int,
    model_name: str,
    batch_size: int,
    success_count: int,
    error_count: int
) -> Dict[str, Any]:
    """
    Create comprehensive metadata for translation processing.
    
    Args:
        start_time: Processing start time (timestamp)
        end_time: Processing end time (timestamp)
        num_texts: Total number of texts processed
        model_name: Model used for translation
        batch_size: Batch size used
        success_count: Number of successful translations
        error_count: Number of failed translations
        
    Returns:
        Metadata dictionary
    """
    processing_time = end_time - start_time
    
    return {
        "processing_time_seconds": round(processing_time, 2),
        "start_time": datetime.fromtimestamp(start_time).isoformat(),
        "end_time": datetime.fromtimestamp(end_time).isoformat(),
        "total_texts": num_texts,
        "successful_translations": success_count,
        "failed_translations": error_count,
        "success_rate": round(success_count / num_texts * 100, 2) if num_texts > 0 else 0,
        "model_used": model_name,
        "batch_size": batch_size,
        "average_time_per_text": round(processing_time / num_texts, 2) if num_texts > 0 else 0,
        "texts_per_second": round(num_texts / processing_time, 2) if processing_time > 0 else 0
    }


def sanitize_model_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize and validate model parameters.
    
    Args:
        params: Raw model parameters
        
    Returns:
        Sanitized parameters
    """
    sanitized = {}
    
    # Allowed parameters with their constraints
    allowed_params = {
        "temperature": (float, 0.0, 2.0),
        "max_tokens": (int, 1, 8000),
        "top_p": (float, 0.0, 1.0),
        "frequency_penalty": (float, -2.0, 2.0),
        "presence_penalty": (float, -2.0, 2.0),
    }
    
    for key, value in params.items():
        if key in allowed_params:
            param_type, min_val, max_val = allowed_params[key]
            try:
                # Convert to appropriate type
                converted_value = param_type(value)
                # Clamp to allowed range
                clamped_value = max(min_val, min(max_val, converted_value))
                sanitized[key] = clamped_value
            except (ValueError, TypeError):
                # Skip invalid parameters
                continue
    
    return sanitized


def format_error_response(error: Exception, context: str = "") -> Dict[str, Any]:
    """
    Format an error for API response.
    
    Args:
        error: The exception that occurred
        context: Additional context about where the error occurred
        
    Returns:
        Formatted error dictionary
    """
    return {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "context": context,
        "timestamp": datetime.now().isoformat()
    }