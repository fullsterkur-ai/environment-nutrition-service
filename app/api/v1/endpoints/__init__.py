"""
app/api/v1/endpoints package initialization.
"""

from app.api.v1.endpoints.teqs import router as teqs_router
from app.api.v1.endpoints.rpe import router as rpe_router
from app.api.v1.endpoints.environment import router as environment_router
from app.api.v1.endpoints.timing import router as timing_router

__all__ = ["teqs_router", "rpe_router", "environment_router", "timing_router"]
