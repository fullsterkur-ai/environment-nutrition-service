"""
app/schemas/health.py

Health check schemas.
Student ID: IT23166660
"""

from typing import Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(default="healthy", description="Service operating status")
    version: str = Field(..., description="API semantic version")
    service: str = Field(..., description="Service name")
    component: str = Field(..., description="System component identifier")
    rpe_model_loaded: bool = Field(..., description="Whether the synthetic RPE model artifact is loaded and ready")
    rpe_model_variant: str = Field(default="baseline", description="Configured RPE model variant")
    rpe_model_name: Optional[str] = Field(default=None, description="Name of the configured RPE model")
    rpe_model_version: Optional[str] = Field(default=None, description="Variant + SHA-256 prefix of the loaded artifact")
    model_status: str = Field(default="DEVELOPMENT_ONLY", description="Current validation state of the ML subsystem")
