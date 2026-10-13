"""
app/schemas/health.py

Health check schemas.
Student ID: IT23166660
"""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(default="healthy", description="Service operating status")
    version: str = Field(..., description="API semantic version")
    service: str = Field(..., description="Service name")
    student_id: str = Field(..., description="Lead researcher / student ID")
    component: str = Field(..., description="System component identifier")
    rpe_model_loaded: bool = Field(..., description="Whether the synthetic RPE model artifact is loaded and ready")
    model_status: str = Field(default="DEVELOPMENT_ONLY", description="Current validation state of the ML subsystem")
