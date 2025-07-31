#!/usr/bin/env python3
"""
Example usage of the Tibetan Buddhist Translation API.

This script demonstrates how to use the API for various translation scenarios.
"""

import asyncio
import requests
import json
from typing import List, Dict, Any


# API Configuration
API_BASE_URL = "http://localhost:8000"


def check_api_health() -> Dict[str, Any]:
    """Check if the API is running and healthy."""
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"API health check failed: {e}")
        return {}


def get_available_models() -> Dict[str, Any]:
    """Get information about available models."""
    try:
        response = requests.get(f"{API_BASE_URL}/models")
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Failed to get models: {e}")
        return {}


def translate_single_text(
    text: str,
    target_language: str = "English",
    model_name: str = "claude",
    text_type: str = "mantra"
) -> Dict[str, Any]:
    """Translate a single text."""
    payload = {
        "text": text,
        "target_language": target_language,
        "model_name": model_name,
        "text_type": text_type
    }
    
    try:
        response = requests.post(f"{API_BASE_URL}/translate/single", json=payload)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Translation failed: {e}")
        return {}


def translate_batch(
    texts: List[str],
    target_language: str = "English",
    model_name: str = "claude",
    text_type: str = "mantra",
    batch_size: int = 5
) -> Dict[str, Any]:
    """Translate multiple texts in batch."""
    payload = {
        "texts": texts,
        "target_language": target_language,
        "model_name": model_name,
        "text_type": text_type,
        "batch_size": batch_size
    }
    
    try:
        response = requests.post(f"{API_BASE_URL}/translate", json=payload)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Batch translation failed: {e}")
        return {}


def demo_mantra_translations():
    """Demonstrate translation of Buddhist mantras."""
    print("🔮 Mantra Translation Demo")
    print("=" * 50)
    
    mantras = [
        "OM MANI PADME HUM",
        "GATE GATE PARAGATE PARASAMGATE BODHI SVAHA",
        "TAYATA OM BEKANDZE BEKANDZE MAHA BEKANDZE RADZA SAMUDGATE SVAHA",
        "OM TARE TUTTARE TURE SVAHA"
    ]
    
    result = translate_batch(
        texts=mantras,
        target_language="English",
        text_type="mantra",
        batch_size=4
    )
    
    if result.get("success"):
        for translation in result["results"]:
            print(f"Original: {translation['original_text']}")
            print(f"Translation: {translation['translated_text']}")
            print("-" * 30)
        
        metadata = result["metadata"]
        print(f"Processing time: {metadata['processing_time_seconds']:.2f}s")
        print(f"Success rate: {metadata['success_rate']:.1f}%")
    else:
        print("Translation failed")


def demo_different_languages():
    """Demonstrate translation to different languages."""
    print("\n🌍 Multi-Language Translation Demo")
    print("=" * 50)
    
    text = "OM MANI PADME HUM"
    languages = ["English", "French", "German", "Spanish", "Italian"]
    
    for language in languages:
        result = translate_single_text(
            text=text,
            target_language=language,
            text_type="mantra"
        )
        
        if result.get("success") and result["results"]:
            translation = result["results"][0]["translated_text"]
            print(f"{language}: {translation}")
        else:
            print(f"{language}: Translation failed")


def demo_different_text_types():
    """Demonstrate translation of different Buddhist text types."""
    print("\n📚 Different Text Types Demo")
    print("=" * 50)
    
    texts_by_type = {
        "mantra": "OM MANI PADME HUM",
        "sutra": "Thus have I heard. At one time the Buddha was staying...",
        "commentary": "This passage explains the meaning of compassion in Buddhist philosophy...",
        "practice_manual": "To begin meditation, sit in a comfortable position..."
    }
    
    for text_type, text in texts_by_type.items():
        print(f"\n{text_type.upper()}:")
        result = translate_single_text(
            text=text,
            target_language="English",
            text_type=text_type
        )
        
        if result.get("success") and result["results"]:
            translation = result["results"][0]["translated_text"]
            print(f"Original: {text}")
            print(f"Translation: {translation}")
        else:
            print("Translation failed")


def demo_model_comparison():
    """Compare translations across different models."""
    print("\n🤖 Model Comparison Demo")
    print("=" * 50)
    
    text = "OM MANI PADME HUM"
    
    # Get available models
    models_info = get_available_models()
    if not models_info:
        print("Could not retrieve available models")
        return
    
    available_models = list(models_info.get("models", {}).keys())
    print(f"Available models: {available_models}")
    
    for model in available_models[:3]:  # Test first 3 models
        print(f"\n{model.upper()}:")
        result = translate_single_text(
            text=text,
            target_language="English",
            model_name=model,
            text_type="mantra"
        )
        
        if result.get("success") and result["results"]:
            translation = result["results"][0]["translated_text"]
            print(f"Translation: {translation}")
        else:
            print("Translation failed")


def demo_custom_model_parameters():
    """Demonstrate using custom model parameters."""
    print("\n⚙️ Custom Model Parameters Demo")
    print("=" * 50)
    
    text = "OM MANI PADME HUM"
    
    # Test with different temperature settings
    temperatures = [0.1, 0.5, 0.9]
    
    for temp in temperatures:
        payload = {
            "text": text,
            "target_language": "English",
            "model_name": "claude",
            "text_type": "mantra",
            "model_params": {
                "temperature": temp,
                "max_tokens": 500
            }
        }
        
        try:
            response = requests.post(f"{API_BASE_URL}/translate/single", json=payload)
            response.raise_for_status()
            result = response.json()
            
            if result.get("success") and result["results"]:
                translation = result["results"][0]["translated_text"]
                print(f"Temperature {temp}: {translation}")
            else:
                print(f"Temperature {temp}: Translation failed")
        except requests.RequestException as e:
            print(f"Temperature {temp}: Request failed - {e}")


def main():
    """Run all demo functions."""
    print("🏛️ Tibetan Buddhist Translation API Demo")
    print("=" * 60)
    
    # Check API health
    health = check_api_health()
    if not health.get("status") == "healthy":
        print("❌ API is not healthy. Please ensure the server is running.")
        print("Start the server with: python main.py")
        return
    
    print("✅ API is healthy and ready!")
    print(f"Version: {health.get('version', 'unknown')}")
    
    # Run demonstrations
    try:
        demo_mantra_translations()
        demo_different_languages()
        demo_different_text_types()
        demo_model_comparison()
        demo_custom_model_parameters()
        
        print("\n🎉 Demo completed successfully!")
        print("\nNext steps:")
        print("- Explore the API documentation at http://localhost:8000/docs")
        print("- Try translating your own Buddhist texts")
        print("- Experiment with different models and parameters")
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")


if __name__ == "__main__":
    main()