"""
app/main.py

Main FastAPI Application Entrypoint.
Component: Environmental Quality & Nutrition Timing Optimizer (TEQS)
Student ID: IT23166660
"""

from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.schemas.health import HealthResponse
from app.services.rpe_service import rpe_service
from app.api.v1.router import api_v1_router


def create_application() -> FastAPI:
    application = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description=(
            "Research Backend for FullStrength.ai — Environmental Quality & Nutrition Timing Optimizer (TEQS). "
            "Exposes pre-session environmental and readiness evaluation (TEQS baseline) and "
            "session RPE prediction using the trained synthetic development model. "
            "All model inferences are strictly tagged for development and unvalidated research usage."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # CORS Middleware
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Health check endpoint
    @application.get(
        "/health",
        response_model=HealthResponse,
        status_code=status.HTTP_200_OK,
        tags=["System Health"],
        summary="Service Health and Validation Status",
        description="Returns current service status, semantic version, and ML artifact readiness flag.",
    )
    def health_check() -> HealthResponse:
        return HealthResponse(
            status="healthy",
            version=settings.VERSION,
            service=settings.PROJECT_NAME,
            student_id=settings.STUDENT_ID,
            component=settings.COMPONENT_NAME,
            rpe_model_loaded=rpe_service.is_loaded,
            model_status="DEVELOPMENT_ONLY",
        )

    # Include API v1 router
    application.include_router(api_v1_router, prefix=settings.API_V1_STR)

    return application


app = create_application()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
