from pydantic_settings import BaseSettings
from typing import Optional, List
import os


class Settings(BaseSettings):
    PROJECT_NAME: str = "SpatialForge"
    VERSION: str = "0.2.0"
    API_V1_STR: str = "/api/v1"
    
    # Server network settings
    HOST: str = os.getenv("HOST", "127.0.0.1")
    PORT: int = int(os.getenv("PORT", "8085"))
    CORS_ORIGINS: List[str] = [orig.strip() for orig in os.getenv("CORS_ORIGINS", "*").split(",")]

    # Upload limitations
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50"))

    # Geometric Reprojection & Safety Gate Thresholds
    F1_NEEDS_REVIEW_THRESHOLD: float = float(os.getenv("F1_NEEDS_REVIEW_THRESHOLD", "0.85"))
    EDGE_RECALL_THRESHOLD: float = float(os.getenv("EDGE_RECALL_THRESHOLD", "0.80"))
    EDGE_PRECISION_THRESHOLD: float = float(os.getenv("EDGE_PRECISION_THRESHOLD", "0.75"))

    # Vision Language Model configuration (optional, for sketch classification & review)
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY", "")
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY", "")

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
