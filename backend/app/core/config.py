from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    PROJECT_NAME: str = "SpatialForge"
    VERSION: str = "0.2.0"
    API_V1_STR: str = "/api/v1"
    
    # Vision Language Model configuration (optional, for sketch classification & multi-view plausibility review)
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY", "")
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY", "")

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
