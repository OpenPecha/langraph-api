"""Tests for the model router functionality."""

import pytest
from unittest.mock import patch, MagicMock

from src.translation_api.models.model_router import ModelRouter, SupportedModel
from src.translation_api.config import Settings


@pytest.fixture
def mock_settings():
    """Mock settings with all API keys."""
    settings = MagicMock(spec=Settings)
    settings.anthropic_api_key = "test-anthropic-key"
    settings.openai_api_key = "test-openai-key"
    settings.gemini_api_key = "test-gemini-key"
    return settings


@pytest.fixture
def model_router(mock_settings):
    """Create model router with mocked settings."""
    with patch('src.translation_api.models.model_router.get_settings', return_value=mock_settings):
        return ModelRouter()


class TestModelRouter:
    """Test ModelRouter functionality."""
    
    def test_init(self, model_router):
        """Test ModelRouter initialization."""
        assert model_router is not None
        assert model_router._model_cache == {}
    
    @patch('src.translation_api.models.model_router.ChatAnthropic')
    def test_create_anthropic_model(self, mock_anthropic, model_router):
        """Test creation of Anthropic models."""
        mock_model = MagicMock()
        mock_anthropic.return_value = mock_model
        
        model = model_router.get_model("claude")
        
        assert model == mock_model
        mock_anthropic.assert_called_once()
        
        # Verify correct model mapping
        call_args = mock_anthropic.call_args
        assert call_args[1]['model'] == 'claude-3-5-sonnet-20241022'
    
    @patch('src.translation_api.models.model_router.ChatOpenAI')
    def test_create_openai_model(self, mock_openai, model_router):
        """Test creation of OpenAI models."""
        mock_model = MagicMock()
        mock_openai.return_value = mock_model
        
        model = model_router.get_model("gpt-4")
        
        assert model == mock_model
        mock_openai.assert_called_once()
        
        # Verify model name passed correctly
        call_args = mock_openai.call_args
        assert call_args[1]['model'] == 'gpt-4'
    
    @patch('src.translation_api.models.model_router.ChatGoogleGenerativeAI')
    def test_create_gemini_model(self, mock_gemini, model_router):
        """Test creation of Gemini models."""
        mock_model = MagicMock()
        mock_gemini.return_value = mock_model
        
        model = model_router.get_model("gemini-pro")
        
        assert model == mock_model
        mock_gemini.assert_called_once()
        
        # Verify model name passed correctly
        call_args = mock_gemini.call_args
        assert call_args[1]['model'] == 'gemini-pro'
    
    def test_unsupported_model(self, model_router):
        """Test error handling for unsupported models."""
        with pytest.raises(ValueError, match="Unsupported model"):
            model_router.get_model("unsupported_model")
    
    @patch('src.translation_api.models.model_router.ChatAnthropic')
    def test_model_caching(self, mock_anthropic, model_router):
        """Test that models are cached properly."""
        mock_model = MagicMock()
        mock_anthropic.return_value = mock_model
        
        # First call
        model1 = model_router.get_model("claude")
        # Second call with same parameters
        model2 = model_router.get_model("claude")
        
        assert model1 == model2
        # Should only be called once due to caching
        mock_anthropic.assert_called_once()
    
    @patch('src.translation_api.models.model_router.ChatAnthropic')
    def test_model_caching_different_params(self, mock_anthropic, model_router):
        """Test that different parameters create different cache entries."""
        mock_model1 = MagicMock()
        mock_model2 = MagicMock()
        mock_anthropic.side_effect = [mock_model1, mock_model2]
        
        # Call with different parameters
        model1 = model_router.get_model("claude", temperature=0.5)
        model2 = model_router.get_model("claude", temperature=0.8)
        
        assert model1 != model2
        assert mock_anthropic.call_count == 2
    
    def test_get_available_models_all_keys(self, model_router):
        """Test get_available_models with all API keys present."""
        available = model_router.get_available_models()
        
        # Should have models from all providers
        assert "claude" in available
        assert "gpt-4" in available
        assert "gemini-pro" in available
        
        # Check model info structure
        assert "provider" in available["claude"]
        assert "description" in available["claude"]
        assert "capabilities" in available["claude"]
    
    def test_get_available_models_no_keys(self):
        """Test get_available_models with no API keys."""
        mock_settings = MagicMock(spec=Settings)
        mock_settings.anthropic_api_key = None
        mock_settings.openai_api_key = None
        mock_settings.gemini_api_key = None
        
        with patch('src.translation_api.models.model_router.get_settings', return_value=mock_settings):
            router = ModelRouter()
            available = router.get_available_models()
            
            assert len(available) == 0
    
    def test_validate_model_availability(self, model_router):
        """Test model availability validation."""
        assert model_router.validate_model_availability("claude") is True
        assert model_router.validate_model_availability("gpt-4") is True
        assert model_router.validate_model_availability("gemini-pro") is True
        assert model_router.validate_model_availability("invalid_model") is False
    
    def test_missing_api_key_error(self):
        """Test error when required API key is missing."""
        mock_settings = MagicMock(spec=Settings)
        mock_settings.anthropic_api_key = None
        mock_settings.openai_api_key = "test-key"
        mock_settings.gemini_api_key = "test-key"
        
        with patch('src.translation_api.models.model_router.get_settings', return_value=mock_settings):
            router = ModelRouter()
            
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY is required"):
                router.get_model("claude")


class TestModelMappings:
    """Test model name mappings."""
    
    @patch('src.translation_api.models.model_router.ChatAnthropic')
    def test_claude_model_mappings(self, mock_anthropic, model_router):
        """Test Claude model name mappings."""
        mappings = {
            "claude": "claude-3-5-sonnet-20241022",
            "claude-sonnet": "claude-3-5-sonnet-20241022",
            "claude-haiku": "claude-3-5-haiku-20241022",
            "claude-opus": "claude-3-opus-20240229"
        }
        
        for input_name, expected_model in mappings.items():
            mock_anthropic.reset_mock()
            model_router.get_model(input_name)
            
            call_args = mock_anthropic.call_args
            assert call_args[1]['model'] == expected_model


class TestModelParameters:
    """Test model parameter handling."""
    
    @patch('src.translation_api.models.model_router.ChatAnthropic')
    def test_default_parameters(self, mock_anthropic, model_router):
        """Test default model parameters."""
        model_router.get_model("claude")
        
        call_args = mock_anthropic.call_args
        assert call_args[1]['temperature'] == 0.3
        assert call_args[1]['max_tokens'] == 4000
    
    @patch('src.translation_api.models.model_router.ChatAnthropic')
    def test_custom_parameters(self, mock_anthropic, model_router):
        """Test custom model parameters."""
        custom_params = {
            "temperature": 0.7,
            "max_tokens": 2000,
            "custom_param": "value"
        }
        
        model_router.get_model("claude", **custom_params)
        
        call_args = mock_anthropic.call_args
        assert call_args[1]['temperature'] == 0.7
        assert call_args[1]['max_tokens'] == 2000
        assert call_args[1]['custom_param'] == "value"


class TestSupportedModelEnum:
    """Test SupportedModel enumeration."""
    
    def test_enum_values(self):
        """Test that all expected models are in the enum."""
        expected_models = [
            "claude", "claude-sonnet", "claude-haiku", "claude-opus",
            "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo",
            "gemini-pro", "gemini-pro-vision"
        ]
        
        enum_values = [model.value for model in SupportedModel]
        
        for model in expected_models:
            assert model in enum_values