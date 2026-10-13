"""
app/api/v1/endpoints/teqs.py

API route for TEQS evaluation.
Student ID: IT23166660
"""

from fastapi import APIRouter, status
from app.schemas.teqs import TEQSRequest, TEQSResponse
from app.services.teqs_service import teqs_service

router = APIRouter(prefix="/teqs", tags=["TEQS Evaluation"])


@router.post(
    "/evaluate",
    response_model=TEQSResponse,
    status_code=status.HTTP_200_OK,
    summary="Evaluate pre-session TEQS baseline score",
    description=(
        "Calculates the development TEQS baseline score, environmental sub-scores, "
        "readiness sub-scores, limiting factors, and factor impact explanations. "
        "Returns model status EXPERIMENTAL_UNVALIDATED."
    ),
)
def evaluate_teqs(payload: TEQSRequest) -> TEQSResponse:
    """Accepts environmental and readiness inputs to compute exploratory TEQS."""
    return teqs_service.evaluate(payload)
