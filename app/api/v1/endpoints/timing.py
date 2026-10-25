"""
app/api/v1/endpoints/timing.py

API route for the context-aware nutrition and hydration timing engine.
Component: Environmental Quality & Nutrition Timing Optimizer (TEQS)
Student ID: IT23166660
"""

from fastapi import APIRouter, status
from app.schemas.timing import TimingEvaluationRequest, TimingEvaluationResponse
from app.services.timing_service import timing_service

router = APIRouter(prefix="/timing", tags=["Nutrition & Hydration Timing"])


@router.post(
    "/evaluate",
    response_model=TimingEvaluationResponse,
    status_code=status.HTTP_200_OK,
    summary="Evaluate context-aware nutrition and hydration timing guidance",
    description=(
        "Evaluates contextual workout parameters (modality, duration, intensity), "
        "environmental conditions (temperature, humidity, indoor/outdoor classification), "
        "meal proximity, and readiness state to generate qualitative, explainable educational timing guidance. "
        "Returns pre-workout, during-workout, and post-workout phase recommendations, triggered rules, "
        "and contextual explanations. "
        "NOTE: This is an educational research prototype (EXPERIMENTAL_RULE_BASED), not a clinical prescription engine."
    ),
)
def evaluate_timing(payload: TimingEvaluationRequest) -> TimingEvaluationResponse:
    """Evaluates contextual inputs and produces educational pre/during/post nutrition timing guidance."""
    return timing_service.evaluate_timing(payload)
