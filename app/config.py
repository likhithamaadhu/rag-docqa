from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://rag:rag@localhost:5432/rag"
    redis_url: str = "redis://localhost:6379/0"

    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_api_key: str = ""
    llm_model: str = "openai/gpt-oss-120b"

    embedding_model: str = "BAAI/bge-small-en-v1.5"

    chunk_size: int = 800
    chunk_overlap: int = 150
    top_k: int = 5
    upload_dir: str = "./uploads"


@lru_cache
def get_settings() -> Settings:
    return Settings()
