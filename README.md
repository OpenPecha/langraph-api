# Tibetan Buddhist Text Translation API

A robust, flexible translation API using LangGraph for translating Tibetan Buddhist texts into multiple target languages. The API supports batch processing, model selection, and is designed to preserve the spiritual, doctrinal, and contextual integrity of Buddhist teachings.

## 🌟 Features

- **Domain-Aware Translation**: Specialized prompts designed for Tibetan Buddhist texts including sutras, commentaries, and philosophical writings
- **Multi-Model Support**: Dynamic routing between Claude (Anthropic), GPT (OpenAI), and Gemini (Google) models
- **Batch Processing**: Efficient handling of multiple texts with configurable batch sizes
- **Flexible Architecture**: LangGraph-based workflow that can easily evolve with new features
- **Comprehensive API**: RESTful endpoints with full OpenAPI documentation
- **Robust Testing**: Extensive test suite covering all components
- **Production Ready**: Built with FastAPI, includes error handling, logging, and monitoring

## 🏗️ Architecture Overview

The system is built with a modular architecture:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FastAPI       │    │   LangGraph      │    │   Model         │
│   Endpoints     │────▶   Workflow       │────▶   Router        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Request       │    │   State          │    │   LangChain     │
│   Validation    │    │   Management     │    │   Models        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Core Components

1. **FastAPI Application** (`src/translation_api/api.py`): RESTful API with endpoints for translation
2. **LangGraph Workflow** (`graph.py`): Flexible workflow for processing translation requests
3. **Model Router** (`src/translation_api/models/model_router.py`): Dynamic model selection and routing
4. **Specialized Prompts** (`src/translation_api/prompts/tibetan_buddhist.py`): Domain-aware prompts for Buddhist texts
5. **State Management** (`src/translation_api/workflows/translation_state.py`): Workflow state definitions

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- API keys for at least one supported model provider (Anthropic, OpenAI, or Google)

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd langraph-api
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**:
   The `.env` file is already configured with API keys. Update as needed:
   ```env
   # Model API Keys
   ANTHROPIC_API_KEY=your-anthropic-key
   OPENAI_API_KEY=your-openai-key  # Optional
   GEMINI_API_KEY=your-gemini-key  # Optional
   
   # LangSmith (for tracing)
   LANGSMITH_API_KEY=your-langsmith-key
   LANGSMITH_PROJECT="Translation"
   LANGSMITH_TRACING=true
   
   # API Configuration
   API_HOST=0.0.0.0
   API_PORT=8000
   DEFAULT_MODEL=claude
   MAX_BATCH_SIZE=50
   ```

4. **Run the API**:
   ```bash
   python main.py
   ```

The API will be available at `http://localhost:8000` with documentation at `http://localhost:8000/docs`.

## 📚 API Usage

### Basic Translation

Translate a single Tibetan Buddhist text:

```python
import requests

response = requests.post("http://localhost:8000/translate/single", json={
    "text": "OM MANI PADME HUM",
    "target_language": "English",
    "model_name": "claude",
    "text_type": "mantra"
})

result = response.json()
print(result["results"][0]["translated_text"])
```

### Batch Translation

Translate multiple texts efficiently:

```python
response = requests.post("http://localhost:8000/translate", json={
    "texts": [
        "OM MANI PADME HUM",
        "GATE GATE PARAGATE PARASAMGATE BODHI SVAHA",
        "TAYATA OM BEKANDZE BEKANDZE MAHA BEKANDZE RADZA SAMUDGATE SVAHA"
    ],
    "target_language": "English",
    "model_name": "claude",
    "text_type": "mantra",
    "batch_size": 3
})

results = response.json()
for result in results["results"]:
    print(f"Original: {result['original_text']}")
    print(f"Translation: {result['translated_text']}")
    print("---")
```

### Available Models

Check which models are available:

```python
response = requests.get("http://localhost:8000/models")
models = response.json()["models"]

for model_name, info in models.items():
    print(f"{model_name}: {info['description']}")
```

### Model Parameters

Customize model behavior:

```python
response = requests.post("http://localhost:8000/translate", json={
    "texts": ["Text to translate"],
    "target_language": "English",
    "model_name": "claude",
    "model_params": {
        "temperature": 0.7,
        "max_tokens": 2000
    }
})
```

## 🎯 Text Types

The API supports different types of Buddhist texts with specialized handling:

- **`sutra`**: Discourses attributed to the Buddha
- **`commentary`**: Explanations and analyses of Buddhist teachings  
- **`practice_manual`**: Meditation and practice instructions
- **`philosophical_treatise`**: Rigorous philosophical analyses
- **`mantra`**: Sacred sounds and phrases

Example with specific text type:

```python
response = requests.post("http://localhost:8000/translate", json={
    "texts": ["Dharma text content here"],
    "target_language": "French",
    "text_type": "sutra",  # Preserves formal sacred tone
    "model_name": "claude"
})
```

## 🔧 Model Selection & Routing

### Supported Models

| Model | Provider | Description | Context Window |
|-------|----------|-------------|----------------|
| `claude` | Anthropic | Claude 3.5 Sonnet - Best overall performance | 200K tokens |
| `claude-haiku` | Anthropic | Fast and efficient | 200K tokens |
| `claude-opus` | Anthropic | Most capable for complex tasks | 200K tokens |
| `gpt-4` | OpenAI | High-quality reasoning | 128K tokens |
| `gpt-4-turbo` | OpenAI | Faster with large context | 128K tokens |
| `gemini-pro` | Google | Good for text tasks | 30K tokens |

### Model Router Implementation

The `ModelRouter` class dynamically selects and initializes models:

```python
from src.translation_api.models.model_router import get_model_router

router = get_model_router()

# Get available models based on API keys
available = router.get_available_models()

# Get a specific model instance
model = router.get_model("claude", temperature=0.5)
```

### Adding New Models

To add support for new models:

1. Update `SupportedModel` enum in `model_router.py`
2. Add model creation logic in `ModelRouter._create_model()`
3. Update model mappings and available models info

## 🔄 LangGraph Workflow

The translation workflow is implemented using LangGraph for flexibility and extensibility:

### Workflow Nodes

1. **Initialize**: Sets up batches and workflow state
2. **Process Batch**: Translates texts using selected model
3. **Finalize**: Aggregates results and creates final output

### Workflow State

The `TranslationWorkflowState` tracks:

- Original request and configuration
- Processing batches and current progress  
- Translation results and metadata
- Errors and retry information
- Extensible custom steps and metadata

### Extending the Workflow

To add new workflow steps (e.g., quality review, post-editing):

```python
def quality_review_node(state: TranslationWorkflowState) -> TranslationWorkflowState:
    # Implement quality review logic
    # Access translations via state["final_results"]
    # Add review metadata to state["custom_steps"]
    return state

# Add to workflow in graph.py
workflow.add_node("quality_review", quality_review_node)
workflow.add_edge("process_batch", "quality_review")
workflow.add_edge("quality_review", "finalize")
```

## 📝 Prompt Design Strategy

### Domain-Aware Prompting

The system uses specialized prompts that ensure:

1. **Doctrinal Accuracy**: Preserves Buddhist terminology and concepts
2. **Contextual Sensitivity**: Adapts to different text types (sutra, commentary, etc.)
3. **Technical Requirements**: Handles Sanskrit/Tibetan terms appropriately
4. **Linguistic Excellence**: Produces fluent target language text
5. **Cultural Bridge**: Makes concepts accessible while maintaining authenticity

### Prompt Structure

```python
from src.translation_api.prompts.tibetan_buddhist import get_translation_prompt

# Single text
prompt = get_translation_prompt(
    source_text="OM MANI PADME HUM",
    target_language="English", 
    text_type="mantra"
)

# Batch processing
prompt = get_translation_prompt(
    source_text="",
    target_language="English",
    text_type="sutra",
    batch_texts=["Text 1", "Text 2", "Text 3"]
)
```

### Specialized Text Type Handling

Each text type has specific guidelines:

```python
from src.translation_api.prompts.tibetan_buddhist import get_specialized_prompts

specialized = get_specialized_prompts()
sutra_info = specialized["sutra"]
print(sutra_info["context"])  # Specific guidance for sutras
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m "not slow"    # Skip slow tests

# Run with coverage
pytest --cov=src --cov-report=html
```

### Test Structure

- `tests/test_api.py`: API endpoint tests
- `tests/test_model_router.py`: Model routing tests  
- `tests/test_workflow.py`: LangGraph workflow tests
- `tests/test_prompts.py`: Prompt generation tests

### Testing Different Models

The test suite includes parameterized tests for different models:

```python
@pytest.mark.parametrize("model_name", ["claude", "gpt-4", "gemini-pro"])
def test_model_translation(model_name):
    # Test translation with different models
    pass
```

### Testing Batch Sizes

Tests validate various batch processing scenarios:

```python
@pytest.mark.parametrize("batch_size", [1, 2, 5, 10, 25])
def test_batch_processing(batch_size):
    # Test different batch sizes
    pass
```

## 🔍 Monitoring & Observability

### LangSmith Integration

The API integrates with LangSmith for tracing and monitoring:

```env
LANGSMITH_API_KEY=your-key
LANGSMITH_PROJECT="Translation"
LANGSMITH_TRACING=true
```

### Request/Response Logging

All translation requests include comprehensive metadata:

```json
{
  "success": true,
  "results": [...],
  "metadata": {
    "processing_time_seconds": 5.2,
    "successful_translations": 10,
    "failed_translations": 0,
    "success_rate": 100.0,
    "model_used": "claude",
    "batch_size": 5,
    "texts_per_second": 1.92
  },
  "errors": []
}
```

## 📊 Performance Considerations

### Batch Size Optimization

- **Small batches (1-5)**: Lower latency, better error isolation
- **Medium batches (5-15)**: Good balance of efficiency and control
- **Large batches (15-50)**: Higher throughput, but increased failure impact

### Model Selection Guidelines

- **Claude**: Best for complex Buddhist philosophical texts
- **Claude Haiku**: Optimal for simple mantras and brief passages
- **GPT-4**: Good alternative with strong multilingual support
- **Gemini Pro**: Cost-effective option for standard translations

### Rate Limiting

Consider implementing rate limiting for production use:

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/translate")
@limiter.limit("10/minute")
async def translate_texts(request: Request, ...):
    # Translation logic
    pass
```

## 🚀 Production Deployment

### Docker Deployment

Create a `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["python", "main.py"]
```

### Environment Configuration

Production environment variables:

```env
# Security
API_HOST=0.0.0.0
API_PORT=8000

# Performance
MAX_BATCH_SIZE=25
DEFAULT_BATCH_SIZE=5

# Monitoring
LANGSMITH_TRACING=true
LOG_LEVEL=INFO
```

### Health Checks

The API includes health check endpoints:

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "available_models": {
    "claude": {...},
    "gpt-4": {...}
  }
}
```

## 🔐 Security Considerations

### API Key Management

- Store API keys in environment variables or secure key management systems
- Rotate keys regularly
- Use different keys for development/staging/production
- Never commit keys to version control

### Input Validation

The API includes comprehensive input validation:

- Text length limits
- Batch size restrictions  
- Model parameter sanitization
- Language code validation

### Rate Limiting & Abuse Prevention

Implement appropriate rate limiting based on your use case:

- Per-IP limits for public APIs
- Per-user limits for authenticated APIs
- Batch size restrictions to prevent resource exhaustion

## 🛠️ Development & Contributing

### Development Setup

1. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

2. Set up pre-commit hooks:
   ```bash
   pre-commit install
   ```

3. Run tests before committing:
   ```bash
   pytest
   black src/ tests/
   isort src/ tests/
   flake8 src/ tests/
   ```

### Code Style

The project uses:
- **Black** for code formatting
- **isort** for import sorting  
- **flake8** for linting
- **Type hints** for better code documentation

### Adding New Features

1. **Workflow Extensions**: Add new nodes to the LangGraph workflow
2. **Model Support**: Extend the ModelRouter for new model providers
3. **Text Types**: Add specialized prompts for new Buddhist text categories
4. **API Endpoints**: Add new FastAPI endpoints for additional functionality

## 📈 Roadmap

### Planned Features

- [ ] **Async Task Queue**: Long-running translation jobs with status polling
- [ ] **Quality Scoring**: Automatic quality assessment of translations
- [ ] **Translation Memory**: Cache and reuse previous translations
- [ ] **Multi-language Support**: Translate between any language pairs
- [ ] **Audio Support**: Integration with speech-to-text for audio dharma talks
- [ ] **Terminology Management**: Custom glossaries for specific traditions
- [ ] **Collaborative Review**: Multi-translator review workflows

### Performance Improvements

- [ ] **Caching Layer**: Redis-based caching for repeated translations
- [ ] **Load Balancing**: Support for multiple model instances
- [ ] **Streaming Responses**: Real-time translation streaming
- [ ] **Batch Optimization**: Dynamic batch size optimization

## 🤝 Support & Community

### Getting Help

- **Documentation**: This README and `/docs` endpoint
- **Issues**: Report bugs and request features via GitHub issues
- **API Documentation**: Interactive docs at `/docs` when running the server

### Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

### License

[Add your license information here]

---

**Built with ❤️ for preserving and sharing Buddhist wisdom across languages and cultures.**