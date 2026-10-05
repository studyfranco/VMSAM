"""Database settings shared by the public and internal uvicorn instances.

Separate from the public app so the internal worker can read DATABASE_URL without
building the public FastAPI app.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Settings read from the environment or `.env`."""
    DATABASE_URL: str
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')
