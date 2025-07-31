"""Tests for Tibetan Buddhist translation prompts."""

import pytest

from src.translation_api.prompts.tibetan_buddhist import (
    get_translation_prompt,
    get_specialized_prompts,
    TIBETAN_BUDDHIST_TRANSLATION_PROMPT
)


class TestTranslationPrompts:
    """Test translation prompt generation."""
    
    def test_single_text_prompt(self):
        """Test prompt generation for single text."""
        source_text = "OM MANI PADME HUM"
        target_language = "English"
        text_type = "mantra"
        
        prompt = get_translation_prompt(
            source_text=source_text,
            target_language=target_language,
            text_type=text_type
        )
        
        assert "English" in prompt
        assert "mantra" in prompt
        assert "OM MANI PADME HUM" in prompt
        assert "DOCTRINAL ACCURACY" in prompt
        assert "CONTEXTUAL SENSITIVITY" in prompt
    
    def test_batch_prompt(self):
        """Test prompt generation for batch processing."""
        batch_texts = [
            "OM MANI PADME HUM",
            "GATE GATE PARAGATE PARASAMGATE BODHI SVAHA"
        ]
        target_language = "French"
        text_type = "mantra"
        
        prompt = get_translation_prompt(
            source_text="",  # Not used for batch
            target_language=target_language,
            text_type=text_type,
            batch_texts=batch_texts
        )
        
        assert "French" in prompt
        assert "mantra" in prompt
        assert str(len(batch_texts)) in prompt  # Should mention batch size
        assert "TEXT 1:" in prompt
        assert "TEXT 2:" in prompt
        assert "OM MANI PADME HUM" in prompt
        assert "GATE GATE PARAGATE" in prompt
        assert "---TRANSLATION_SEPARATOR---" in prompt
    
    def test_prompt_language_substitution(self):
        """Test that target language is properly substituted."""
        languages = ["English", "French", "German", "Spanish", "Chinese"]
        
        for language in languages:
            prompt = get_translation_prompt(
                source_text="Test text",
                target_language=language,
                text_type="sutra"
            )
            
            # Should appear multiple times in the prompt
            assert prompt.count(language) >= 2
    
    def test_prompt_text_type_substitution(self):
        """Test that text type is properly substituted."""
        text_types = ["sutra", "commentary", "philosophical_treatise", "mantra"]
        
        for text_type in text_types:
            prompt = get_translation_prompt(
                source_text="Test text",
                target_language="English",
                text_type=text_type
            )
            
            assert text_type in prompt
    
    def test_prompt_guidelines_present(self):
        """Test that all critical guidelines are present in prompt."""
        prompt = get_translation_prompt(
            source_text="Test text",
            target_language="English",
            text_type="Buddhist text"
        )
        
        required_sections = [
            "DOCTRINAL ACCURACY",
            "CONTEXTUAL SENSITIVITY", 
            "TECHNICAL REQUIREMENTS",
            "LINGUISTIC EXCELLENCE",
            "CULTURAL BRIDGE"
        ]
        
        for section in required_sections:
            assert section in prompt
    
    def test_batch_prompt_formatting(self):
        """Test proper formatting of batch prompts."""
        batch_texts = ["Text one", "Text two", "Text three"]
        
        prompt = get_translation_prompt(
            source_text="",
            target_language="English",
            text_type="sutra",
            batch_texts=batch_texts
        )
        
        # Check that texts are properly numbered and separated
        for i, text in enumerate(batch_texts, 1):
            assert f"TEXT {i}:" in prompt
            assert text in prompt
        
        assert "3 Tibetan Buddhist texts" in prompt  # Batch size mentioned
    
    def test_empty_batch_texts(self):
        """Test handling of empty batch texts list."""
        prompt = get_translation_prompt(
            source_text="Single text",
            target_language="English",
            text_type="sutra",
            batch_texts=[]
        )
        
        # Should fall back to single text processing
        assert "Single text" in prompt
        assert "TEXT 1:" not in prompt


class TestSpecializedPrompts:
    """Test specialized prompts for different text types."""
    
    def test_get_specialized_prompts_structure(self):
        """Test structure of specialized prompts dictionary."""
        specialized = get_specialized_prompts()
        
        assert isinstance(specialized, dict)
        assert len(specialized) > 0
        
        # Check required text types
        required_types = ["sutra", "commentary", "practice_manual", "philosophical_treatise"]
        for text_type in required_types:
            assert text_type in specialized
    
    def test_specialized_prompt_content(self):
        """Test content of specialized prompts."""
        specialized = get_specialized_prompts()
        
        for text_type, info in specialized.items():
            assert "context" in info
            assert "terminology_focus" in info
            assert isinstance(info["context"], str)
            assert isinstance(info["terminology_focus"], str)
            assert len(info["context"]) > 0
            assert len(info["terminology_focus"]) > 0
    
    def test_sutra_specialization(self):
        """Test sutra-specific prompt content."""
        specialized = get_specialized_prompts()
        sutra_info = specialized["sutra"]
        
        assert "discourse attributed to the Buddha" in sutra_info["context"]
        assert "sacred tone" in sutra_info["context"]
        assert "dharma" in sutra_info["terminology_focus"]
    
    def test_commentary_specialization(self):
        """Test commentary-specific prompt content."""
        specialized = get_specialized_prompts()
        commentary_info = specialized["commentary"]
        
        assert "commentary" in commentary_info["context"]
        assert "analytical" in commentary_info["context"]
        assert "philosophical terminology" in commentary_info["terminology_focus"]
    
    def test_practice_manual_specialization(self):
        """Test practice manual-specific prompt content."""
        specialized = get_specialized_prompts()
        practice_info = specialized["practice_manual"]
        
        assert "practice" in practice_info["context"]
        assert "meditation" in practice_info["context"]
        assert "instructions" in practice_info["context"]
        assert "meditation" in practice_info["terminology_focus"]
    
    def test_philosophical_treatise_specialization(self):
        """Test philosophical treatise-specific prompt content."""
        specialized = get_specialized_prompts()
        philosophical_info = specialized["philosophical_treatise"]
        
        assert "philosophical" in philosophical_info["context"]
        assert "analytical" in philosophical_info["context"]
        assert "argumentation" in philosophical_info["context"]


class TestPromptConstants:
    """Test prompt constants and templates."""
    
    def test_main_prompt_template_placeholders(self):
        """Test that main prompt template has required placeholders."""
        required_placeholders = [
            "{target_language}",
            "{text_type}",
            "{source_text}"
        ]
        
        for placeholder in required_placeholders:
            assert placeholder in TIBETAN_BUDDHIST_TRANSLATION_PROMPT
    
    def test_prompt_template_formatting(self):
        """Test that prompt can be formatted without errors."""
        try:
            formatted = TIBETAN_BUDDHIST_TRANSLATION_PROMPT.format(
                target_language="English",
                text_type="sutra",
                source_text="Test text"
            )
            assert len(formatted) > 0
            assert "{" not in formatted  # No unformatted placeholders
            assert "}" not in formatted
        except KeyError as e:
            pytest.fail(f"Missing placeholder in prompt template: {e}")
    
    def test_prompt_content_quality(self):
        """Test quality and completeness of prompt content."""
        # Test with actual formatting
        formatted_prompt = TIBETAN_BUDDHIST_TRANSLATION_PROMPT.format(
            target_language="English",
            text_type="sutra",
            source_text="Test Buddhist text"
        )
        
        # Should be substantial in length
        assert len(formatted_prompt) > 1000
        
        # Should contain key Buddhist concepts
        buddhist_terms = ["Buddhist", "dharma", "sutra", "bodhisattva"]
        found_terms = sum(1 for term in buddhist_terms if term.lower() in formatted_prompt.lower())
        assert found_terms >= 2
        
        # Should contain translation guidance
        translation_terms = ["translate", "translation", "accuracy", "meaning"]
        found_translation_terms = sum(1 for term in translation_terms if term.lower() in formatted_prompt.lower())
        assert found_translation_terms >= 3


class TestPromptEdgeCases:
    """Test edge cases and error handling."""
    
    def test_very_long_text(self):
        """Test prompt with very long source text."""
        long_text = "OM MANI PADME HUM " * 1000  # Very long text
        
        prompt = get_translation_prompt(
            source_text=long_text,
            target_language="English",
            text_type="mantra"
        )
        
        assert long_text in prompt
        assert len(prompt) > len(long_text)  # Should include guidelines too
    
    def test_special_characters_in_text(self):
        """Test prompt with special characters."""
        special_text = "ཨོཾ་མ་ཎི་པདྨེ་ཧཱུྃ། གཏད་གཏད་པཱ་ར་གཏད་པཱ་ར་སཾ་གཏད་བོ་དྷི་སྭཱ་ཧཱ།"
        
        prompt = get_translation_prompt(
            source_text=special_text,
            target_language="English",
            text_type="mantra"
        )
        
        assert special_text in prompt
    
    def test_large_batch(self):
        """Test prompt with large batch of texts."""
        large_batch = [f"Text number {i}" for i in range(50)]
        
        prompt = get_translation_prompt(
            source_text="",
            target_language="English",
            text_type="sutra",
            batch_texts=large_batch
        )
        
        assert "50 Tibetan Buddhist texts" in prompt
        assert "TEXT 50:" in prompt
        assert large_batch[-1] in prompt