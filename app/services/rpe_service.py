"""
app/services/rpe_service.py

RPE inference service loading and evaluating the synthetic development model pipeline.
Student ID: IT23166660
"""

import hashlib
import json
from pathlib import Path
from typing import Optional
import joblib
import pandas as pd
from fastapi import HTTPException, status

from app.core.config import settings
from app.schemas.rpe import RPEPredictRequest, RPEPredictResponse

VARIANT_BASELINE = "baseline"
VARIANT_PP1 = "pp1_optimized"
SUPPORTED_VARIANTS = (VARIANT_BASELINE, VARIANT_PP1)

FEATURES = [
    "temperature_c",
    "humidity_pct",
    "air_quality_index",
    "sleep_hours",
    "sleep_quality_1_5",
    "mood_readiness_1_5",
    "planned_duration_min",
    "planned_intensity_1_10",
    "workout_type",
]


class ModelArtifactNotFoundError(HTTPException):
    """Raised when the trained model artifact file is missing or unreadable."""

    def __init__(self, message: str = "RPE model artifact is unavailable."):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=message,
        )


class RPEService:
    """Service encapsulating the trained synthetic RPE regression pipeline.

    The model variant is selected explicitly (``RPE_MODEL_VARIANT`` env var or the
    ``variant`` argument). An unsupported variant raises ``ValueError``; a missing
    artifact for the selected variant makes predictions fail with HTTP 503. The service
    never substitutes a different model.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        summary_path: Optional[Path] = None,
        variant: Optional[str] = None,
    ):
        chosen = (variant if variant is not None else settings.RPE_MODEL_VARIANT).strip().lower()
        if chosen not in SUPPORTED_VARIANTS:
            raise ValueError(
                f"Unsupported RPE_MODEL_VARIANT '{chosen}'. "
                f"Supported values: {', '.join(SUPPORTED_VARIANTS)}."
            )
        self.variant = chosen
        if chosen == VARIANT_PP1:
            self.model_path = model_path or settings.pp1_rpe_model_path
            self.summary_path = summary_path or settings.pp1_metrics_path
            self._model_name = VARIANT_PP1
        else:
            self.model_path = model_path or settings.rpe_model_path
            self.summary_path = summary_path or settings.model_summary_path
            self._model_name = "ridge"
        self._model = None
        self._model_version = "unavailable"
        self._load_model_metadata()
        self.load_model()

    def _load_model_metadata(self) -> None:
        """Reads model identity from the evaluation summary written with the artifact."""
        if not (self.summary_path and self.summary_path.is_file()):
            return
        try:
            with open(self.summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
        except Exception:
            return
        if self.variant == VARIANT_PP1:
            self._model_name = summary.get("best_model_name", VARIANT_PP1)
        else:
            self._model_name = summary.get("regression", {}).get("best_model", "ridge")

    @staticmethod
    def _artifact_digest(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()[:12]

    def load_model(self) -> bool:
        """Attempts to load the model artifact from disk."""
        if not self.model_path.is_file():
            self._model = None
            self._model_version = "unavailable"
            return False

        try:
            self._model = joblib.load(self.model_path)
            self._model_version = f"{self.variant}+sha256:{self._artifact_digest(self.model_path)}"
            return True
        except Exception:
            self._model = None
            self._model_version = "unavailable"
            return False

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_version(self) -> str:
        return self._model_version

    def predict(self, request: RPEPredictRequest) -> RPEPredictResponse:
        """
        Executes pre-session prediction using the loaded model.
        Does NOT invent or mock a prediction if model artifacts are unavailable.
        """
        if not self.is_loaded:
            raise ModelArtifactNotFoundError(
                f"RPE model artifact is not available at '{self.model_path}'. "
                "The server will not fabricate an ungrounded prediction."
            )

        row_dict = {
            "temperature_c": request.temperature_c,
            "humidity_pct": request.humidity_pct,
            "air_quality_index": request.air_quality_index,
            "sleep_hours": request.sleep_hours,
            "sleep_quality_1_5": request.sleep_quality_1_5,
            "mood_readiness_1_5": request.mood_readiness_1_5,
            "planned_duration_min": request.planned_duration_min,
            "planned_intensity_1_10": request.planned_intensity_1_10,
            "workout_type": request.workout_type,
        }

        df = pd.DataFrame([row_dict], columns=FEATURES)

        try:
            pred = self._model.predict(df)[0]
            val = float(pred) if hasattr(pred, "__float__") else float(pred[0])
            rounded_pred = round(val, 2)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Inference error during RPE prediction: {str(e)}",
            )

        return RPEPredictResponse(
            predicted_session_rpe=rounded_pred,
            model_name=self.model_name,
            model_variant=self.variant,
            model_version=self.model_version,
            model_status="DEVELOPMENT_ONLY",
            message=(
                "This prediction was generated by a development model trained strictly "
                "on synthetic data and has NOT been validated for real athlete decisions or clinical load management."
            ),
        )


rpe_service = RPEService()
