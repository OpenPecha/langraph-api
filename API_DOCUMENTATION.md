# Tibetan Buddhist Text Translation API Documentation

## Overview

The Tibetan Buddhist Text Translation API is a robust, flexible translation service built with FastAPI and LangGraph. It specializes in translating Tibetan Buddhist texts while preserving spiritual and doctrinal integrity. The API supports multiple AI models, batch processing, real-time progress tracking, and custom user rules.

## Base URL

```
http://localhost:8001
```

## Features

- **Multi-Model Support**: Claude (Sonnet, Haiku, Opus), Gemini Pro
- **Batch Processing**: Process multiple texts efficiently
- **Real-time Streaming**: Server-Sent Events (SSE) for live progress updates
- **Custom User Rules**: Add custom translation instructions
- **Domain-Aware Prompts**: Specialized prompts for different Buddhist text types
- **Extensible Architecture**: Built with LangGraph for easy feature additions

---

## Authentication

Currently, no authentication is required. API keys for AI models are configured server-side.

---

## Core Endpoints

The API has been streamlined to include only the essential, production-ready endpoints:

### 1. Health Check

**GET** `/health`

Check API status and available models.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "available_models": {
    "claude": {
      "name": "Claude Sonnet",
      "provider": "anthropic",
      "capabilities": ["translation", "analysis"]
    },
    "claude-haiku": {
      "name": "Claude Haiku",
      "provider": "anthropic",
      "capabilities": ["translation", "fast_processing"]
    },
    "claude-opus": {
      "name": "Claude Opus",
      "provider": "anthropic",
      "capabilities": ["translation", "high_quality"]
    },
    "gemini-pro": {
      "name": "Gemini Pro",
      "provider": "google",
      "capabilities": ["translation", "multilingual"]
    }
  }
}
```

### 2. Standard Translation

**POST** `/translate`

Translate multiple texts with standard processing.

**Request Body:**
```json
{
  "texts": ["OM MANI PADME HUM", "May all beings be happy"],
  "target_language": "English",
  "model_name": "claude",
  "text_type": "mantra",
  "batch_size": 3,
  "model_params": {},
  "user_rules": "Include Sanskrit transliteration when applicable"
}
```

**Parameters:**
- `texts` (required): Array of strings to translate (1-100 items)
- `target_language` (required): Target language name
- `model_name` (optional): Model to use (default: "claude")
- `text_type` (optional): Type of Buddhist text (default: "Buddhist text")
- `batch_size` (optional): Batch size for processing (1-50, default: 5)
- `model_params` (optional): Additional model parameters
- `user_rules` (optional): Custom translation instructions

**Text Types:**
- `mantra`
- `sutra`
- `commentary`
- `practice_manual`
- `philosophical_treatise`

**Response:**
```json
{
  "success": true,
  "results": [
    {
      "original_text": "OM MANI PADME HUM",
      "translated_text": "Om Mani Padme Hum (Hail to the jewel in the lotus)",
      "confidence_score": null,
      "metadata": {
        "batch_id": "uuid-string",
        "model_used": "claude",
        "text_type": "mantra"
      }
    }
  ],
  "metadata": {
    "initialized_at": "2025-01-31T12:00:00",
    "completed_at": "2025-01-31T12:00:30",
    "total_processing_time": 30.5,
    "successful_batches": 1,
    "failed_batches": 0,
    "total_translations": 2,
    "success_rate": 1.0
  },
  "errors": []
}
```

### 3. Single Text Translation

**POST** `/translate/single`

Translate a single text (convenience endpoint).

**Request Body:**
```json
{
  "text": "OM MANI PADME HUM",
  "target_language": "English",
  "model_name": "claude",
  "text_type": "mantra",
  "model_params": {},
  "user_rules": "Include Sanskrit transliteration"
}
```

**Response:** Same format as standard translation endpoint.

### 4. Streaming Translation (Recommended)

**POST** `/translate/stream`

**IMPORTANT FOR FRONTEND**: This endpoint uses Server-Sent Events (SSE) for real-time progress updates.

**Request Body:** Same as standard translation endpoint.

**Response:** Server-Sent Events stream

**Content-Type:** `text/event-stream`

**Event Types:**

1. **initialization**
```
data: {
  "timestamp": "2025-01-31T12:00:00",
  "type": "initialization",
  "status": "starting",
  "total_texts": 5,
  "target_language": "English",
  "model": "claude",
  "batch_size": 3
}
```

2. **planning**
```
data: {
  "timestamp": "2025-01-31T12:00:01",
  "type": "planning",
  "status": "batches_created",
  "total_batches": 2,
  "batch_size": 3
}
```

3. **batch_start**
```
data: {
  "timestamp": "2025-01-31T12:00:02",
  "type": "batch_start",
  "status": "processing_batch",
  "batch_number": 1,
  "batch_id": "batch_1",
  "texts_in_batch": 3,
  "progress_percent": 0
}
```

4. **translation_start**
```
data: {
  "timestamp": "2025-01-31T12:00:03",
  "type": "translation_start",
  "status": "translating_batch",
  "batch_size": 3
}
```

5. **text_completed**
```
data: {
  "timestamp": "2025-01-31T12:00:05",
  "type": "text_completed",
  "status": "text_translated",
  "text_number": 1,
  "total_texts": 5,
  "progress_percent": 20,
  "translation_preview": "Om Mani Padme Hum (Hail to the jewel..."
}
```

6. **batch_completed** ⭐ **KEY EVENT FOR UI**
```
data: {
  "timestamp": "2025-01-31T12:00:08",
  "type": "batch_completed",
  "status": "batch_completed",
  "batch_number": 1,
  "batch_id": "batch_1",
  "processing_time": 5.2,
  "texts_processed": 3,
  "cumulative_progress": 60,
  "batch_results": [
    {
      "original_text": "OM MANI PADME HUM",
      "translated_text": "Om Mani Padme Hum (Hail to the jewel in the lotus)",
      "metadata": {
        "batch_id": "batch_1",
        "model_used": "claude",
        "text_type": "mantra"
      }
    }
  ]
}
```

7. **completion**
```
data: {
  "timestamp": "2025-01-31T12:00:15",
  "type": "completion",
  "status": "completed",
  "total_texts": 5,
  "successful_translations": 5,
  "total_processing_time": 15.3,
  "average_time_per_text": 3.06
}
```

8. **error**
```
data: {
  "timestamp": "2025-01-31T12:00:10",
  "type": "error",
  "status": "failed",
  "error": "Model unavailable",
  "processed_texts": 2,
  "total_texts": 5
}
```

### 5. Single Text Streaming

**POST** `/translate/single/stream`

Stream progress for single text translation.

**Request Body:** Same as single translation endpoint.

**Response:** SSE stream with same event types as streaming translation.

### 6. Web UI

**GET** `/` or **GET** `/ui`

Access the built-in web interface for testing translations.

---

## Frontend Integration Guide

### Using Standard Endpoints

```javascript
// Standard translation
const response = await fetch('http://localhost:8001/translate', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    texts: ['OM MANI PADME HUM', 'May all beings be happy'],
    target_language: 'English',
    model_name: 'claude',
    text_type: 'mantra',
    user_rules: 'Include pronunciation guide'
  })
});

const result = await response.json();
console.log(result.results);
```

### Using Server-Sent Events (Recommended)

```javascript
function streamTranslation(requestData) {
  return fetch('http://localhost:8001/translate/stream', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'text/event-stream'
    },
    body: JSON.stringify(requestData)
  }).then(response => {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    
    const processStream = ({ done, value }) => {
      if (done) return;
      
      const chunk = decoder.decode(value);
      const lines = chunk.split('\n');
      
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            let jsonStr = line.slice(6).trim();
            
            // Handle potential double "data: " prefix
            if (jsonStr.startsWith('data: ')) {
              jsonStr = jsonStr.slice(6).trim();
            }
            
            if (jsonStr) {
              const data = JSON.parse(jsonStr);
              handleStreamEvent(data);
            }
          } catch (e) {
            console.log('Skipping invalid JSON:', line);
          }
        }
      }
      
      return reader.read().then(processStream);
    };
    
    return reader.read().then(processStream);
  });
}

function handleStreamEvent(data) {
  switch (data.type) {
    case 'initialization':
      updateProgress(0, `Starting translation of ${data.total_texts} texts...`);
      break;
      
    case 'batch_completed':
      // ⭐ KEY: Display results immediately as batches complete
      if (data.batch_results) {
        displayBatchResults(data.batch_results);
      }
      updateProgress(data.cumulative_progress, `Batch ${data.batch_number} completed`);
      break;
      
    case 'completion':
      updateProgress(100, 'Translation completed!');
      break;
      
    case 'error':
      handleError(data.error);
      break;
  }
}
```

### React Hook Example

```javascript
import { useState, useEffect } from 'react';

function useTranslationStream() {
  const [results, setResults] = useState([]);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState('');
  const [isTranslating, setIsTranslating] = useState(false);

  const startTranslation = async (requestData) => {
    setIsTranslating(true);
    setResults([]);
    setProgress(0);
    
    try {
      const response = await fetch('http://localhost:8001/translate/stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'text/event-stream'
        },
        body: JSON.stringify(requestData)
      });

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      const processStream = async ({ done, value }) => {
        if (done) {
          setIsTranslating(false);
          return;
        }

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              let jsonStr = line.slice(6).trim();
              if (jsonStr.startsWith('data: ')) {
                jsonStr = jsonStr.slice(6).trim();
              }

              if (jsonStr) {
                const data = JSON.parse(jsonStr);
                
                switch (data.type) {
                  case 'batch_completed':
                    if (data.batch_results) {
                      setResults(prev => [...prev, ...data.batch_results]);
                    }
                    setProgress(data.cumulative_progress || 0);
                    setStatus(`Batch ${data.batch_number} completed`);
                    break;
                    
                  case 'completion':
                    setProgress(100);
                    setStatus('Translation completed!');
                    break;
                    
                  case 'error':
                    setStatus(`Error: ${data.error}`);
                    setIsTranslating(false);
                    break;
                }
              }
            } catch (e) {
              console.log('Skipping invalid JSON:', line);
            }
          }
        }

        return reader.read().then(processStream);
      };

      reader.read().then(processStream);
    } catch (error) {
      setStatus(`Error: ${error.message}`);
      setIsTranslating(false);
    }
  };

  return { results, progress, status, isTranslating, startTranslation };
}
```

---

## Error Handling

### HTTP Status Codes

- `200`: Success
- `400`: Bad Request (invalid parameters)
- `500`: Internal Server Error

### Error Response Format

```json
{
  "detail": "Model 'invalid-model' is not available. Available models: ['claude', 'claude-haiku', 'claude-opus', 'gemini-pro']"
}
```

### Common Errors

1. **Invalid Model**: Model name not in available models list
2. **Batch Size Exceeded**: Batch size larger than maximum allowed (50)
3. **Empty Texts**: No texts provided in request
4. **Model Unavailable**: AI model temporarily unavailable

---

## Best Practices

### For Frontend Developers

1. **Use Streaming for Better UX**: Always use `/translate/stream` for better user experience
2. **Handle Batch Results**: Display translations as `batch_completed` events arrive
3. **Progress Updates**: Use `cumulative_progress` for progress bars
4. **Error Handling**: Handle both HTTP errors and stream error events
5. **Debounce Requests**: Prevent duplicate requests while translation is in progress

### Request Optimization

1. **Batch Size**: Use 3-5 texts per batch for optimal performance
2. **Text Length**: Shorter texts translate faster
3. **Model Selection**: 
   - `claude-haiku`: Fastest, good quality
   - `claude`: Balanced speed/quality (recommended)
   - `claude-opus`: Best quality, slower
   - `gemini-pro`: Alternative option

### User Rules Examples

```json
{
  "user_rules": "Include Sanskrit transliteration in parentheses"
}
```

```json
{
  "user_rules": "Use formal language and include brief explanations of Buddhist terms"
}
```

```json
{
  "user_rules": "Maintain poetic meter and rhythm where possible"
}
```

---

## Rate Limits

Currently no rate limits are implemented. For production use, consider implementing:
- Rate limiting per IP
- Request size limits
- Concurrent request limits

---

## Example Complete Implementation

```html
<!DOCTYPE html>
<html>
<head>
    <title>Translation API Example</title>
</head>
<body>
    <div>
        <textarea id="inputText" placeholder="Enter texts (one per line)"></textarea>
        <select id="language">
            <option value="English">English</option>
            <option value="Spanish">Spanish</option>
        </select>
        <select id="model">
            <option value="claude">Claude</option>
            <option value="gemini-pro">Gemini Pro</option>
        </select>
        <textarea id="userRules" placeholder="Custom rules (optional)"></textarea>
        <button onclick="translate()">Translate</button>
    </div>
    
    <div>
        <div id="progress"></div>
        <div id="results"></div>
    </div>

    <script>
        async function translate() {
            const texts = document.getElementById('inputText').value
                .split('\n')
                .map(t => t.trim())
                .filter(t => t.length > 0);
            
            const requestData = {
                texts: texts,
                target_language: document.getElementById('language').value,
                model_name: document.getElementById('model').value,
                user_rules: document.getElementById('userRules').value || null,
                batch_size: 3
            };

            const response = await fetch('http://localhost:8001/translate/stream', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'text/event-stream'
                },
                body: JSON.stringify(requestData)
            });

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let results = [];

            const processStream = async ({ done, value }) => {
                if (done) return;

                const chunk = decoder.decode(value);
                const lines = chunk.split('\n');

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            let jsonStr = line.slice(6).trim();
                            if (jsonStr.startsWith('data: ')) {
                                jsonStr = jsonStr.slice(6).trim();
                            }

                            if (jsonStr) {
                                const data = JSON.parse(jsonStr);
                                
                                if (data.type === 'batch_completed' && data.batch_results) {
                                    results.push(...data.batch_results);
                                    displayResults(results);
                                }
                                
                                if (data.cumulative_progress !== undefined) {
                                    document.getElementById('progress').textContent = 
                                        `Progress: ${data.cumulative_progress}%`;
                                }
                            }
                        } catch (e) {
                            console.log('Skipping invalid JSON:', line);
                        }
                    }
                }

                return reader.read().then(processStream);
            };

            reader.read().then(processStream);
        }

        function displayResults(results) {
            const html = results.map(result => `
                <div>
                    <strong>Original:</strong> ${result.original_text}<br>
                    <strong>Translation:</strong> ${result.translated_text}
                </div>
            `).join('<hr>');
            
            document.getElementById('results').innerHTML = html;
        }
    </script>
</body>
</html>
```

---

## Support

For questions or issues:
1. Check API health: `GET /health`
2. View interactive docs: `http://localhost:8001/docs`
3. Test endpoints: `http://localhost:8001/docs#/`

This API is designed to be developer-friendly with comprehensive error messages and detailed response metadata.