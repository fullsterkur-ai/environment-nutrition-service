"""
app/api/v1/router.py

Aggregation router for API v1 endpoints.
Student ID: IT23166660
"""

from fastapi import APIRouter
from app.api.v1.endpoints.teqs import router as teqs_router
from app.api.v1.endpoints.rpe import router as rpe_router
from app.api.v1.endpoints.environment import router as environment_router

api_v1_router = APIRouter()
api_v1_router.include_router(teqs_router)
api_v1_router.include_router(rpe_router)
api_v1_router.include_router(environment_router)
