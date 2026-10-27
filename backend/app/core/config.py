from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    PROJECT_NAME: str = "SpatialForge"
    VERSION: str = "0.1.0"
    API_V1_STR: str = "/api/v1"
    
    # Muse Spark Open / LLM Configuration
    MUSE_API_KEY: Optional[str] = os.getenv("MUSE_API_KEY", "")
    MUSE_ENDPOINT: str = os.getenv("MUSE_ENDPOINT", "https://api.baseten.co/v1/models/meta-models/muse-spark-open-1026")
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY", "")
    
    # Manufacturing Constraints (Default: Aluminum 6061 / CNC standard)
    MIN_WALL_THICKNESS_MM: float = 2.0
    DEFAULT_PLATE_THICKNESS_MM: float = 16.0
    DEFAULT_CHAMFER_MM: float = 1.5
    
    # Pipeline Settings
    IOU_CONVERGENCE_THRESHOLD: float = 0.98  # 98% IoU alignment required
    MAX_HEALING_ITERATIONS: int = 3

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
