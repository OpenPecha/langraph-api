# Quick Start Guide

## 🚀 Getting Started in 3 Steps

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Start the Server
```bash
python main.py
```

The server will start on `http://localhost:8000`

### Step 3: Test the API
In a new terminal window:
```bash
python test_api_basic.py
```

## 📱 Using the API

### Web Interface
Visit `http://localhost:8000/docs` for interactive API documentation

### Quick Test
```bash
curl -X POST "http://localhost:8000/translate/single" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "OM MANI PADME HUM",
    "target_language": "English",
    "model_name": "claude",
    "text_type": "mantra"
  }'
```

### Python Example
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

## 🛠️ Troubleshooting

### Server Won't Start
- Check that port 8000 is available
- Ensure all dependencies are installed
- Verify Python version is 3.8+

### Translation Fails
- Check that API keys are configured in `.env`
- Verify the model name is available (check `/models` endpoint)
- Check server logs for detailed error messages

### Import Errors
- Make sure you're running from the project root directory
- Ensure all required packages are installed

## 🔧 Configuration

The `.env` file contains your API keys and configuration:
- `ANTHROPIC_API_KEY`: Required for Claude models
- `OPENAI_API_KEY`: Optional, for GPT models  
- `GEMINI_API_KEY`: Optional, for Gemini models

## 📚 More Examples

Run the comprehensive example script:
```bash
python example_usage.py
```

This demonstrates:
- Different translation models
- Various target languages
- Different Buddhist text types
- Batch processing
- Custom model parameters

## 🆘 Need Help?

- Check the full documentation in `README.md`
- Visit the interactive docs at `http://localhost:8000/docs`
- Review the example scripts in the project