"""Database settings shared by both uvicorn instances.

Kept in its own module so the internal worker does not have to import the public
app just to read DATABASE_URL, which would build a second unused FastAPI object
and register routes the internal instance never serves.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    DATABASE_URL: str
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')
