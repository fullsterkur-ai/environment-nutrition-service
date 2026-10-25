"""
app/api/v1/endpoints/rpe.py

API route for Session RPE prediction using synthetic development model.
Student ID: IT23166660
"""

from fastapi import APIRouter, status
from app.schemas.rpe import RPEPredictRequest, RPEPredictResponse
from app.services.rpe_service import rpe_service

router = APIRouter(prefix="/rpe", tags=["Session RPE Prediction"])


@router.post(
    "/predict",
    response_model=RPEPredictResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict session RPE using synthetic development model",
    description=(
        "Executes regression inference on the pre-session environmental, readiness, "
        "and planned workout inputs using the saved synthetic Ridge pipeline. "
        "Returns model status DEVELOPMENT_ONLY and a clear scientific advisory notice. "
        "Will return HTTP 503 if model artifacts are unavailable."
    ),
)
def predict_rpe(payload: RPEPredictRequest) -> RPEPredictResponse:
    """Predicts session RPE using pre-session features and synthetic model artifact."""
    return rpe_service.predict(payload)
