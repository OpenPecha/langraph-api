# LangGraph Buddhist Text Translation API

[![API Version](https://img.shields.io/badge/API-v2.0.0-blue.svg)](./docs/API_REFERENCE.md)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](./tests)
[![Documentation](https://img.shields.io/badge/docs-comprehensive-blue.svg)](./docs/README.md)

A sophisticated, multi-stage API for translating, analyzing, and standardizing Tibetan Buddhist texts. This project uses a modular, streaming-first architecture built with FastAPI and LangGraph to provide a flexible and powerful pipeline for high-quality, consistent translations.

---

### Key Features

-   **Modular Workflow**: A decoupled three-stage pipeline for Translation, Glossary Extraction, and Standardization.
-   **Streaming First**: Real-time events for all long-running operations, providing a transparent and interactive user experience.
-   **Intelligent Standardization**: A powerful suite of tools to analyze and enforce terminological consistency across large datasets.
-   **Multi-Model Support**: Dynamically route requests to models from Anthropic, OpenAI, and Google.
-   **Performance Optimized**: In-memory caching for repeated requests and parallel, batched processing for glossary and standardization tasks.
-   **Comprehensive Documentation**: Includes a full API reference, an architectural overview, and a guide to the project's evolution.

---

### Architecture: A Decoupled, Multi-Stage Pipeline

The API is designed as a set of independent services, giving the user full control over each step of the process.

```mermaid
graph TD;
    subgraph "User Interaction"
        A[Client]
    end

    subgraph "API Services"
        B[1. Translation Service]
        C[2. Glossary Service]
        D[3. Standardization Service]
    end
    
    subgraph "AI Backend"
        E[Language Models]
    end

    A -- "Translate Texts" --> B;
    A -- "Extract Glossary" --> C;
    A -- "Analyze & Apply Standards" --> D;
    
    B -- Calls --> E;
    C -- Calls --> E;
    D -- Calls --> E;
```

For a deep dive into the architecture, the key technical decisions, and the project's evolution, please see our **[Comprehensive Documentation](./docs/README.md)**.

---

### Quick Start

#### 1. Prerequisites

-   Python 3.10+
-   An API key for at least one supported model provider (e.g., Anthropic).

#### 2. Installation & Setup

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/OpenPecha/langraph-api.git
    cd langraph-api
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure API Keys**:
    Create a `.env` file in the project root and add your API keys.
    ```env
    # .env
    ANTHROPIC_API_KEY="your-anthropic-key"
    OPENAI_API_KEY="your-openai-key"
    GEMINI_API_KEY="your-google-key"
    ```

#### 3. Run the Server

Start the application using Uvicorn.

```bash
uvicorn src.translation_api.api:app --reload --port 8001
```

-   The **Web UI** will be available at [http://localhost:8001/](http://localhost:8001/).
-   The interactive **Swagger Docs** will be at [http://localhost:8001/docs](http://localhost:8001/docs).

---

### Example Workflow using `curl`

Here is how to use the full pipeline from the command line.

**Step 1: Get a Translation**
```bash
curl -X POST http://localhost:8001/translate -H "Content-Type: application/json" -d '{
  "texts": ["om mani padme hum"], "target_language": "english"
}' > translation_output.json
```

**Step 2: Extract the Glossary**
```bash
curl -X POST http://localhost:8001/glossary/extract -H "Content-Type: application/json" -d '{
  "items": '"$(jq '.results' translation_output.json)"'
}' > glossary_output.json
```
*(Requires `jq` to be installed)*

---

### Comprehensive Documentation

We have created an extensive set of documents covering every aspect of this project.

-   **[Architectural Overview](./docs/ARCHITECTURE.md)**
-   **[Project Evolution & Key Decisions](./docs/EVOLUTION.md)**
-   **[Full API Reference](./docs/API_REFERENCE.md)**
-   **[Frontend UI Guide](./docs/UI_GUIDE.md)**
-   **[Setup & Deployment Guide](./docs/SETUP.md)**

Please refer to these documents for a deep understanding of the system's design and capabilities.