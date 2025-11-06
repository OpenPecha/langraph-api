"""Configuration management for the translation API."""

import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Keys
    anthropic_api_key: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    gemini_api_key: Optional[str] = Field(None, env="GEMINI_API_KEY")
    # External service credentials
    dharmamitra_password: Optional[str] = Field(None, env="DHARMAMITRA_PASSWORD")
    dharmamitra_token: Optional[str] = Field(None, env="DHARMAMITRA_TOKEN")
    
    # LangSmith Configuration
    langsmith_api_key: Optional[str] = Field(None, env="LANGSMITH_API_KEY")
    langsmith_project: str = Field("Translation", env="LANGSMITH_PROJECT")
    langsmith_tracing: bool = Field(True, env="LANGSMITH_TRACING")
    langsmith_endpoint: str = Field("https://api.smith.langchain.com", env="LANGSMITH_ENDPOINT")
    
    # API Configuration
    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")
    
    # Translation Configuration
    default_model: str = Field("claude", env="DEFAULT_MODEL")
    max_batch_size: int = Field(50, env="MAX_BATCH_SIZE")
    default_batch_size: int = Field(5, env="DEFAULT_BATCH_SIZE")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the application settings."""
    return settings