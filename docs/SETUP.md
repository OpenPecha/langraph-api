# Setup and Deployment Guide

This guide provides instructions for setting up and running the API server locally.

### 1. Prerequisites

*   Python 3.10+
*   `pip` for package management

### 2. Installation

1.  **Clone the Repository**:
    ```bash
    git clone <your-repository-url>
    cd langraph-api
    ```

2.  **Install Dependencies**:
    It is recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

### 3. Configuration

The API requires API keys for the language models it uses. These are managed via a `.env` file.

1.  **Create a `.env` file** in the root of the project directory.

2.  **Add Your API Keys**: Add the keys for the services you want to use. You only need to provide keys for the models you intend to use.

    ```dotenv
    # .env file

    # For Anthropic (Claude) models
    ANTHROPIC_API_KEY="sk-ant-..."

    # For OpenAI (GPT) models
    OPENAI_API_KEY="sk-..."

    # For Google (Gemini) models
    GEMINI_API_KEY="..."
    ```

### 4. Running the Server

Once the dependencies are installed and the `.env` file is configured, you can run the server.

1.  **Start the Uvicorn Server**:
    The application is run using `uvicorn`. From the root project directory, run:
    ```bash
    uvicorn src.translation_api.api:app --reload --port 8001
    ```
    *   `--reload`: This flag enables hot-reloading, so the server will automatically restart when you make changes to the code.
    *   `--port 8001`: This specifies the port to run on.

2.  **Access the API**:
    *   **Web UI**: Open your browser and navigate to [http://localhost:8001](http://localhost:8001).
    *   **Swagger Docs**: The interactive API documentation (Swagger UI) is available at [http://localhost:8001/docs](http://localhost:8001/docs).

### 5. Deployment (Conceptual)

This application is a standard FastAPI project and can be deployed to any platform that supports ASGI applications.

*   **Containerization**: A `Dockerfile` is included in the project, allowing you to build a Docker container for easy deployment.
    ```bash
    docker build -t translation-api .
    docker run -d -p 8001:8001 --env-file .env translation-api
    ```
*   **Cloud Services**: This container can be deployed to services like AWS ECS, Google Cloud Run, or Azure App Service. 