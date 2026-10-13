"""
app/core/config.py

Application Configuration for FullStrength.ai TEQS Service.
Student ID: IT23166660
"""

import os
from pathlib import Path
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load .env file if present in project root
BASE_DIR = Path(__file__).resolve().parent.parent.parent
env_path = BASE_DIR / ".env"
if env_path.is_file():
    load_dotenv(dotenv_path=env_path)


class Settings(BaseModel):
    PROJECT_NAME: str = "FullStrength.ai - Environmental Quality & Nutrition Timing Optimizer (TEQS)"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    STUDENT_ID: str = "IT23166660"
    COMPONENT_NAME: str = "Environmental Quality & Nutrition Timing Optimizer (TEQS)"

    # Base paths
    BASE_DIR: Path = Field(default_factory=lambda: BASE_DIR)
    ARTIFACTS_DIR: Path = Field(default_factory=lambda: BASE_DIR / "artifacts")

    # Model artifact paths
    RPE_MODEL_FILENAME: str = "best_rpe_model.joblib"
    MODEL_SUMMARY_FILENAME: str = "model_summary.json"

    # OpenWeather API Configuration
    OPENWEATHER_API_KEY: str = Field(
        default_factory=lambda: os.getenv("OPENWEATHER_API_KEY", "")
    )
    OPENWEATHER_BASE_URL: str = "https://api.openweathermap.org"
    OPENWEATHER_GEO_URL: str = "http://api.openweathermap.org/geo/1.0/direct"
    WEATHER_CACHE_TTL_SECONDS: int = 600  # 10 minutes TTL
    WEATHER_REQUEST_TIMEOUT_SECONDS: float = 10.0

    @property
    def rpe_model_path(self) -> Path:
        return self.ARTIFACTS_DIR / self.RPE_MODEL_FILENAME

    @property
    def model_summary_path(self) -> Path:
        return self.ARTIFACTS_DIR / self.MODEL_SUMMARY_FILENAME


settings = Settings()
