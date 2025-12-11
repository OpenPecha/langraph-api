"""
Simple in-memory cache for translation and glossary results.
"""
from typing import Dict, Any, Optional, Tuple, List
from .models.glossary import GlossaryTerm

class SimpleCache:
    """A simple singleton-like class for in-memory caching."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SimpleCache, cls).__new__(cls)
            # Cache for [translation_text]
            cls._instance.translation_cache: Dict[str, str] = {}
            # Cache for [List[GlossaryTerm]]
            cls._instance.glossary_cache: Dict[str, List[GlossaryTerm]] = {}
        return cls._instance

    def get_translation_cache_key(self, source_text: str, target_language: str, text_type: str, model_name: str, user_rules: Optional[str]) -> str:
        """Generates a consistent key for translation caching."""
        rules = user_rules or ""
        return f"{source_text}|{target_language}|{text_type}|{model_name}|{rules}"

    def get_glossary_cache_key(self, source_text: str, translated_text: str, model_name: str) -> str:
        """Generates a consistent key for glossary caching."""
        return f"{source_text}|{translated_text}|{model_name}"

    def get_translation(self, key: str) -> Optional[str]:
        return self.translation_cache.get(key)

    def set_translation(self, key: str, translation: str):
        self.translation_cache[key] = translation

    def get_glossary(self, key: str) -> Optional[List[GlossaryTerm]]:
        return self.glossary_cache.get(key)

    def set_glossary(self, key: str, glossary_terms: List[GlossaryTerm]):
        self.glossary_cache[key] = glossary_terms

    def clear_all(self):
        """Clears both the translation and glossary caches."""
        self.translation_cache.clear()
        self.glossary_cache.clear()

    def clear(self) -> int:
        """Clears both the translation and glossary caches and returns the count of cleared items."""
        translation_count = len(self.translation_cache)
        glossary_count = len(self.glossary_cache)
        total_count = translation_count + glossary_count
        self.translation_cache.clear()
        self.glossary_cache.clear()
        return total_count

# Global instance
cache = SimpleCache()

def get_cache() -> SimpleCache:
    """Get the global cache instance."""
    return cache 