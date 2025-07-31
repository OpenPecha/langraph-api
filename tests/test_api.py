"""Tests for the translation API endpoints."""

import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from src.translation_api.api import create_app
from src.translation_api.workflows.translation_state import TranslationRequest, TranslationResult


@pytest.fixture
def client():
    """Create test client."""
    app = create_app()
    return TestClient(app)


@pytest.fixture
def sample_translation_request():
    """Sample translation request for testing."""
    return {
        "texts": [
            "OM MANI PADME HUM",
            "GATE GATE PARAGATE PARASAMGATE BODHI SVAHA"
        ],
        "target_language": "English",
        "model_name": "claude",
        "text_type": "mantra",
        "batch_size": 2
    }


@pytest.fixture
def mock_workflow_result():
    """Mock workflow result for testing."""
    return {
        "workflow_status": "completed",
        "final_results": [
            TranslationResult(
                original_text="OM MANI PADME HUM",
                translated_text="Hail the jewel in the lotus",
                metadata={"model_used": "claude"}
            ),
            TranslationResult(
                original_text="GATE GATE PARAGATE PARASAMGATE BODHI SVAHA",
                translated_text="Gone, gone, gone beyond, gone completely beyond, awakening, so be it!",
                metadata={"model_used": "claude"}
            )
        ],
        "metadata": {
            "total_processing_time": 5.2,
            "successful_batches": 1,
            "total_translations": 2
        },
        "errors": []
    }


class TestHealthEndpoint:
    """Test health check endpoint."""
    
    def test_health_check(self, client):
        """Test health check returns correct status."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"
        assert "available_models" in data


class TestModelsEndpoint:
    """Test models information endpoint."""
    
    def test_get_models(self, client):
        """Test models endpoint returns available models."""
        response = client.get("/models")
        assert response.status_code == 200
        
        data = response.json()
        assert "models" in data
        assert isinstance(data["models"], dict)


class TestTranslationEndpoint:
    """Test translation endpoints."""
    
    @patch('src.translation_api.api.run_translation_workflow')
    def test_translate_texts_success(self, mock_workflow, client, sample_translation_request, mock_workflow_result):
        """Test successful translation request."""
        mock_workflow.return_value = mock_workflow_result
        
        response = client.post("/translate", json=sample_translation_request)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert len(data["results"]) == 2
        assert data["results"][0]["original_text"] == "OM MANI PADME HUM"
        assert "metadata" in data
    
    def test_translate_texts_invalid_model(self, client, sample_translation_request):
        """Test translation with invalid model."""
        sample_translation_request["model_name"] = "invalid_model"
        
        response = client.post("/translate", json=sample_translation_request)
        assert response.status_code == 400
        assert "not available" in response.json()["detail"]
    
    def test_translate_texts_empty_texts(self, client):
        """Test translation with empty texts list."""
        request = {
            "texts": [],
            "target_language": "English",
            "model_name": "claude"
        }
        
        response = client.post("/translate", json=request)
        assert response.status_code == 422  # Validation error
    
    def test_translate_texts_too_many_texts(self, client):
        """Test translation with too many texts."""
        request = {
            "texts": ["text"] * 101,  # Exceeds max_items=100
            "target_language": "English",
            "model_name": "claude"
        }
        
        response = client.post("/translate", json=request)
        assert response.status_code == 422  # Validation error
    
    def test_translate_texts_large_batch_size(self, client, sample_translation_request):
        """Test translation with batch size exceeding maximum."""
        sample_translation_request["batch_size"] = 100  # Exceeds max allowed
        
        response = client.post("/translate", json=sample_translation_request)
        assert response.status_code == 400
        assert "exceeds maximum" in response.json()["detail"]
    
    @patch('src.translation_api.api.run_translation_workflow')
    def test_translate_single_text(self, mock_workflow, client, mock_workflow_result):
        """Test single text translation endpoint."""
        mock_workflow.return_value = mock_workflow_result
        
        request = {
            "text": "OM MANI PADME HUM",
            "target_language": "English",
            "model_name": "claude",
            "text_type": "mantra"
        }
        
        response = client.post("/translate/single", json=request)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert len(data["results"]) >= 1


class TestModelValidation:
    """Test model validation functionality."""
    
    @patch('src.translation_api.models.model_router.ModelRouter.validate_model_availability')
    def test_model_availability_check(self, mock_validate, client, sample_translation_request):
        """Test model availability validation."""
        mock_validate.return_value = False
        
        response = client.post("/translate", json=sample_translation_request)
        assert response.status_code == 400
        assert "not available" in response.json()["detail"]


class TestBatchProcessing:
    """Test batch processing functionality."""
    
    @patch('src.translation_api.api.run_translation_workflow')
    def test_different_batch_sizes(self, mock_workflow, client, mock_workflow_result):
        """Test translation with different batch sizes."""
        mock_workflow.return_value = mock_workflow_result
        
        test_cases = [1, 2, 5, 10]
        
        for batch_size in test_cases:
            request = {
                "texts": ["Test text"] * (batch_size * 2),
                "target_language": "English",
                "model_name": "claude",
                "batch_size": batch_size
            }
            
            response = client.post("/translate", json=request)
            assert response.status_code == 200, f"Failed for batch_size={batch_size}"


class TestErrorHandling:
    """Test error handling scenarios."""
    
    @patch('src.translation_api.api.run_translation_workflow')
    def test_workflow_exception(self, mock_workflow, client, sample_translation_request):
        """Test handling of workflow exceptions."""
        mock_workflow.side_effect = Exception("Workflow failed")
        
        response = client.post("/translate", json=sample_translation_request)
        assert response.status_code == 500
        assert "Internal server error" in response.json()["detail"]
    
    def test_invalid_json(self, client):
        """Test handling of invalid JSON."""
        response = client.post("/translate", data="invalid json")
        assert response.status_code == 422


class TestModelParameters:
    """Test model parameter handling."""
    
    @patch('src.translation_api.api.run_translation_workflow')
    def test_custom_model_params(self, mock_workflow, client, mock_workflow_result):
        """Test translation with custom model parameters."""
        mock_workflow.return_value = mock_workflow_result
        
        request = {
            "texts": ["Test text"],
            "target_language": "English",
            "model_name": "claude",
            "model_params": {
                "temperature": 0.7,
                "max_tokens": 2000
            }
        }
        
        response = client.post("/translate", json=request)
        assert response.status_code == 200
        
        # Verify workflow was called with correct parameters
        mock_workflow.assert_called_once()
        call_args = mock_workflow.call_args[0][0]
        assert call_args.model_params["temperature"] == 0.7
        assert call_args.model_params["max_tokens"] == 2000


class TestLanguageSupport:
    """Test different target language support."""
    
    @patch('src.translation_api.api.run_translation_workflow')
    def test_different_target_languages(self, mock_workflow, client, mock_workflow_result):
        """Test translation to different target languages."""
        mock_workflow.return_value = mock_workflow_result
        
        languages = ["English", "French", "German", "Spanish", "Chinese", "Japanese"]
        
        for language in languages:
            request = {
                "texts": ["Test text"],
                "target_language": language,
                "model_name": "claude"
            }
            
            response = client.post("/translate", json=request)
            assert response.status_code == 200, f"Failed for language={language}"


class TestTextTypes:
    """Test different Buddhist text types."""
    
    @patch('src.translation_api.api.run_translation_workflow')
    def test_different_text_types(self, mock_workflow, client, mock_workflow_result):
        """Test translation of different Buddhist text types."""
        mock_workflow.return_value = mock_workflow_result
        
        text_types = ["sutra", "commentary", "practice_manual", "philosophical_treatise", "mantra"]
        
        for text_type in text_types:
            request = {
                "texts": ["Test text"],
                "target_language": "English",
                "model_name": "claude",
                "text_type": text_type
            }
            
            response = client.post("/translate", json=request)
            assert response.status_code == 200, f"Failed for text_type={text_type}"