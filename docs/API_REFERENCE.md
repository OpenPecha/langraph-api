# API Reference

This document provides a detailed reference for all the API endpoints.

---

## Translation Service

### `POST /translate`

*   **Description**: Translates a list of source texts. This is a non-streaming endpoint that returns the full list of translations after the entire batch is complete.
*   **Request Body**: `TranslationAPIRequest`
*   **Response Body**: `TranslationAPIResponse`
*   **Example Usage**:
    ```bash
    curl -X POST http://localhost:8001/translate -H "Content-Type: application/json" -d '{
      "texts": ["OM MANI PADME HUM"],
      "target_language": "English",
      "model_name": "claude"
    }'
    ```

### `POST /translate/stream`

*   **Description**: Translates a list of source texts and streams the results in real-time as batches are completed.
*   **Request Body**: `TranslationAPIRequest`
*   **Response**: A stream of Server-Sent Events (SSE). See the Swagger UI for detailed event examples.

---

## Glossary Service

### `POST /glossary/extract`

*   **Description**: Extracts a consolidated glossary from a list of previously translated texts.
*   **Request Body**: `GlossaryExtractionRequest`
*   **Response Body**: `Glossary`
*   **Example Usage**:
    ```bash
    curl -X POST http://localhost:8001/glossary/extract -H "Content-Type: application/json" -d '{
      "items": [{"original_text": "...", "translated_text": "..."}],
      "model_name": "claude"
    }'
    ```

### `POST /glossary/extract/stream`

*   **Description**: Extracts a glossary and streams the results in real-time as batches of terms are extracted.
*   **Request Body**: `GlossaryExtractionRequest`
*   **Response**: A stream of Server-Sent Events (SSE).

---

## Standardization Service

### `POST /standardize/analyze`

*   **Description**: Analyzes a list of translations to find source terms with multiple, inconsistent translations. This is a non-LLM, synchronous endpoint.
*   **Request Body**: `AnalysisRequest`
*   **Response Body**: `AnalysisResponse`
*   **Example Usage**:
    ```bash
    curl -X POST http://localhost:8001/standardize/analyze -H "Content-Type: application/json" -d '{
      "items": [
        {
          "original_text": "...", "translated_text": "...", 
          "glossary": [{"source_term": "bodhicitta", "translated_term": "mind of enlightenment"}]
        },
        {
          "original_text": "...", "translated_text": "...", 
          "glossary": [{"source_term": "bodhicitta", "translated_term": "enlightenment mind"}]
        }
      ]
    }'
    ```

### `POST /standardize/apply`

*   **Description**: Applies a set of standardization rules to a list of translations, intelligently re-translating only the affected sentences.
*   **Request Body**: `StandardizationRequest`
*   **Response Body**: `StandardizationResponse`

### `POST /standardize/apply/stream`

*   **Description**: Applies standardization rules and streams the updated translations in real-time as they are corrected.
*   **Request Body**: `StandardizationRequest`
*   **Response**: A stream of Server-Sent Events (SSE).

---

## System Service

### `POST /system/clear-cache`

*   **Description**: Clears the server's in-memory cache for both translations and glossaries.
*   **Request Body**: None
*   **Response**: A JSON object confirming the cache has been cleared. 