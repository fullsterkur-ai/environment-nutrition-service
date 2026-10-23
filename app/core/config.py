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

    # RPE model variant used by the API (research demo switch).
    #   baseline      -> artifacts/best_rpe_model.joblib (Ridge, default)
    #   pp1_optimized -> artifacts/pp1_model_optimization/best_rpe_model_pp1.joblib
    # Any other value is rejected at start-up; there is NO silent fallback.
    RPE_MODEL_VARIANT: str = Field(
        default_factory=lambda: (os.getenv("RPE_MODEL_VARIANT", "baseline").strip().lower() or "baseline")
    )
    PP1_ARTIFACTS_SUBDIR: str = "pp1_model_optimization"
    PP1_RPE_MODEL_FILENAME: str = "best_rpe_model_pp1.joblib"
    PP1_METRICS_FILENAME: str = "final_holdout_metrics.json"

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

    @property
    def pp1_rpe_model_path(self) -> Path:
        return self.ARTIFACTS_DIR / self.PP1_ARTIFACTS_SUBDIR / self.PP1_RPE_MODEL_FILENAME

    @property
    def pp1_metrics_path(self) -> Path:
        return self.ARTIFACTS_DIR / self.PP1_ARTIFACTS_SUBDIR / self.PP1_METRICS_FILENAME


settings = Settings()
