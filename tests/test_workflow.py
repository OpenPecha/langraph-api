"""Tests for the LangGraph translation workflow."""

import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock

from src.translation_api.workflows.translation_state import (
    TranslationRequest,
    TranslationWorkflowState,
    TranslationBatch,
    TranslationResult
)
from graph import (
    initialize_workflow,
    process_batch,
    check_completion,
    finalize_workflow,
    create_translation_workflow,
    run_translation_workflow
)


@pytest.fixture
def sample_request():
    """Sample translation request for testing."""
    return TranslationRequest(
        texts=["OM MANI PADME HUM", "GATE GATE PARAGATE"],
        target_language="English",
        model_name="claude",
        text_type="mantra",
        batch_size=2
    )


@pytest.fixture
def initial_state(sample_request):
    """Initial workflow state for testing."""
    return {
        "original_request": sample_request,
        "batches": [],
        "current_batch_index": 0,
        "batch_results": [],
        "final_results": [],
        "total_texts": 0,
        "processed_texts": 0,
        "workflow_start_time": 0.0,
        "workflow_status": "initializing",
        "errors": [],
        "retry_count": 0,
        "model_name": sample_request.model_name,
        "model_params": sample_request.model_params,
        "custom_steps": {},
        "metadata": {}
    }


class TestWorkflowInitialization:
    """Test workflow initialization."""
    
    def test_initialize_workflow(self, initial_state, sample_request):
        """Test workflow initialization creates correct state."""
        result_state = initialize_workflow(initial_state)
        
        assert result_state["workflow_status"] == "running"
        assert len(result_state["batches"]) == 1  # 2 texts with batch_size=2
        assert result_state["total_texts"] == 2
        assert result_state["current_batch_index"] == 0
        
        # Check batch creation
        batch = result_state["batches"][0]
        assert len(batch.texts) == 2
        assert batch.target_language == "English"
        assert batch.model_name == "claude"
    
    def test_initialize_workflow_large_batch(self, initial_state):
        """Test initialization with large text list requiring multiple batches."""
        # Create request with 5 texts and batch size of 2
        large_request = TranslationRequest(
            texts=["text1", "text2", "text3", "text4", "text5"],
            target_language="English",
            model_name="claude",
            batch_size=2
        )
        initial_state["original_request"] = large_request
        
        result_state = initialize_workflow(initial_state)
        
        assert len(result_state["batches"]) == 3  # 2, 2, 1 texts per batch
        assert result_state["total_texts"] == 5
        assert result_state["metadata"]["total_batches"] == 3


class TestBatchProcessing:
    """Test batch processing functionality."""
    
    @patch('graph.get_model_router')
    def test_process_batch_success(self, mock_router, initial_state, sample_request):
        """Test successful batch processing."""
        # Setup mock model
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Hail the jewel in the lotus---TRANSLATION_SEPARATOR---Gone, gone, gone beyond"
        mock_model.invoke.return_value = mock_response
        
        mock_router_instance = MagicMock()
        mock_router_instance.get_model.return_value = mock_model
        mock_router.return_value = mock_router_instance
        
        # Initialize state first
        state = initialize_workflow(initial_state)
        
        # Process the batch
        result_state = process_batch(state)
        
        assert len(result_state["batch_results"]) == 1
        assert result_state["batch_results"][0].success is True
        assert len(result_state["final_results"]) == 2
        assert result_state["current_batch_index"] == 1
        assert result_state["processed_texts"] == 2
    
    @patch('graph.get_model_router')
    def test_process_batch_single_text(self, mock_router, initial_state):
        """Test processing batch with single text."""
        # Setup for single text
        single_request = TranslationRequest(
            texts=["OM MANI PADME HUM"],
            target_language="English",
            model_name="claude",
            batch_size=1
        )
        initial_state["original_request"] = single_request
        
        # Setup mock model
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Hail the jewel in the lotus"
        mock_model.invoke.return_value = mock_response
        
        mock_router_instance = MagicMock()
        mock_router_instance.get_model.return_value = mock_model
        mock_router.return_value = mock_router_instance
        
        # Initialize and process
        state = initialize_workflow(initial_state)
        result_state = process_batch(state)
        
        assert len(result_state["final_results"]) == 1
        assert result_state["final_results"][0].translated_text == "Hail the jewel in the lotus"
    
    @patch('graph.get_model_router')
    def test_process_batch_error(self, mock_router, initial_state, sample_request):
        """Test batch processing with model error."""
        # Setup mock to raise exception
        mock_router_instance = MagicMock()
        mock_router_instance.get_model.side_effect = Exception("Model error")
        mock_router.return_value = mock_router_instance
        
        # Initialize state first
        state = initialize_workflow(initial_state)
        
        # Process the batch
        result_state = process_batch(state)
        
        assert len(result_state["batch_results"]) == 1
        assert result_state["batch_results"][0].success is False
        assert "Model error" in result_state["batch_results"][0].error_message
        assert len(result_state["errors"]) == 1
        assert result_state["current_batch_index"] == 1


class TestWorkflowCompletion:
    """Test workflow completion logic."""
    
    def test_check_completion_continue(self, initial_state, sample_request):
        """Test completion check when more batches remain."""
        state = initialize_workflow(initial_state)
        # Current index is 0, and we have 1 batch, so should continue
        
        result = check_completion(state)
        assert result == "continue"
    
    def test_check_completion_finalize(self, initial_state, sample_request):
        """Test completion check when all batches are done."""
        state = initialize_workflow(initial_state)
        state["current_batch_index"] = 1  # Processed all batches
        
        result = check_completion(state)
        assert result == "finalize"
    
    def test_finalize_workflow(self, initial_state, sample_request):
        """Test workflow finalization."""
        # Setup state as if processing is complete
        state = initialize_workflow(initial_state)
        state["workflow_start_time"] = 1000.0
        state["batch_results"] = [
            MagicMock(success=True),
            MagicMock(success=False)
        ]
        state["final_results"] = [MagicMock(), MagicMock()]
        
        with patch('time.time', return_value=1005.0):  # 5 seconds later
            result_state = finalize_workflow(state)
        
        assert result_state["workflow_status"] == "completed"
        assert result_state["metadata"]["total_processing_time"] == 5.0
        assert result_state["metadata"]["successful_batches"] == 1
        assert result_state["metadata"]["failed_batches"] == 1
        assert result_state["metadata"]["total_translations"] == 2


class TestWorkflowGraph:
    """Test the complete workflow graph."""
    
    def test_create_workflow_graph(self):
        """Test workflow graph creation."""
        workflow = create_translation_workflow()
        
        assert workflow is not None
        # The graph should have our defined nodes
        assert "initialize" in workflow.nodes
        assert "process_batch" in workflow.nodes
        assert "finalize" in workflow.nodes
    
    @patch('graph.get_model_router')
    @pytest.mark.asyncio
    async def test_run_translation_workflow(self, mock_router, sample_request):
        """Test complete workflow execution."""
        # Setup mock model
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Hail the jewel in the lotus---TRANSLATION_SEPARATOR---Gone, gone, gone beyond"
        mock_model.invoke.return_value = mock_response
        
        mock_router_instance = MagicMock()
        mock_router_instance.get_model.return_value = mock_model
        mock_router.return_value = mock_router_instance
        
        # Run the workflow
        result_state = await run_translation_workflow(sample_request)
        
        assert result_state["workflow_status"] == "completed"
        assert len(result_state["final_results"]) == 2
        assert result_state["processed_texts"] == 2


class TestErrorHandling:
    """Test error handling in workflow."""
    
    @patch('graph.get_model_router')
    def test_batch_processing_recovers_from_error(self, mock_router, initial_state):
        """Test that workflow continues after batch error."""
        # Create request with multiple batches
        multi_batch_request = TranslationRequest(
            texts=["text1", "text2", "text3", "text4"],
            target_language="English",
            model_name="claude",
            batch_size=2
        )
        initial_state["original_request"] = multi_batch_request
        
        # Setup mock to fail on first batch, succeed on second
        mock_model_success = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "translation1---TRANSLATION_SEPARATOR---translation2"
        mock_model_success.invoke.return_value = mock_response
        
        mock_router_instance = MagicMock()
        # First call fails, second succeeds
        mock_router_instance.get_model.side_effect = [
            Exception("First batch fails"),
            mock_model_success
        ]
        mock_router.return_value = mock_router_instance
        
        # Initialize and process
        state = initialize_workflow(initial_state)
        
        # Process first batch (should fail)
        state = process_batch(state)
        assert len(state["errors"]) == 1
        assert state["current_batch_index"] == 1
        
        # Process second batch (should succeed)
        state = process_batch(state)
        assert len(state["batch_results"]) == 2
        assert state["batch_results"][0].success is False
        assert state["batch_results"][1].success is True


class TestModelIntegration:
    """Test integration with different models."""
    
    @patch('graph.get_model_router')
    def test_different_models(self, mock_router, initial_state):
        """Test workflow with different model configurations."""
        models_to_test = ["claude", "gpt-4", "gemini-pro"]
        
        for model_name in models_to_test:
            # Setup request for current model
            request = TranslationRequest(
                texts=["Test text"],
                target_language="English",
                model_name=model_name,
                batch_size=1
            )
            initial_state["original_request"] = request
            
            # Setup mock
            mock_model = MagicMock()
            mock_response = MagicMock()
            mock_response.content = f"Translation by {model_name}"
            mock_model.invoke.return_value = mock_response
            
            mock_router_instance = MagicMock()
            mock_router_instance.get_model.return_value = mock_model
            mock_router.return_value = mock_router_instance
            
            # Test processing
            state = initialize_workflow(initial_state)
            result_state = process_batch(state)
            
            assert len(result_state["final_results"]) == 1
            assert result_state["final_results"][0].metadata["model_used"] == model_name