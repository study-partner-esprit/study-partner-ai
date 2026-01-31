"""Application settings and configuration."""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings."""
    
    # API settings
    app_name: str = "Study Partner AI"
    version: str = "0.1.0"
    api_prefix: str = "/api/v1"
    debug: bool = False
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8001
    
    # Database settings
    database_url: str = "postgresql://user:password@localhost:5432/study_partner"
    
    # AI/LLM settings
    openai_api_key: str = ""
    model_name: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 2000
    
    # CORS settings
    allowed_origins: list[str] = ["http://localhost:3000"]
    
    # Logging
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
