"""
app/schemas/teqs.py

Pydantic schemas for TEQS baseline evaluation.
Student ID: IT23166660
"""

from typing import List, Dict
from pydantic import BaseModel, Field


class TEQSRequest(BaseModel):
    temperature_c: float = Field(
        ...,
        ge=-10.0,
        le=60.0,
        description="Ambient gym/training temperature in Celsius (-10.0 to 60.0)",
        examples=[28.5],
    )
    humidity_pct: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Relative ambient humidity percentage (0.0 to 100.0)",
        examples=[65.0],
    )
    air_quality_index: float = Field(
        ...,
        ge=0.0,
        le=500.0,
        description="Air Quality Index (AQI) (0.0 to 500.0)",
        examples=[45.0],
    )
    sleep_hours: float = Field(
        ...,
        ge=0.0,
        le=24.0,
        description="Total sleep duration in hours (0.0 to 24.0)",
        examples=[7.5],
    )
    sleep_quality_1_5: float = Field(
        ...,
        ge=1.0,
        le=5.0,
        description="Subjective sleep recovery quality rating (1.0 to 5.0 scale)",
        examples=[4.0],
    )
    mood_readiness_1_5: float = Field(
        ...,
        ge=1.0,
        le=5.0,
        description="Subjective mood and pre-training readiness rating (1.0 to 5.0 scale)",
        examples=[4.0],
    )


class EnvironmentalSubscores(BaseModel):
    temperature: float = Field(..., description="Temperature sub-score on 0-100 scale")
    humidity: float = Field(..., description="Humidity sub-score on 0-100 scale")
    air_quality: float = Field(..., description="Air quality sub-score on 0-100 scale")


class ReadinessSubscores(BaseModel):
    sleep_duration: float = Field(..., description="Sleep duration sub-score on 0-100 scale")
    sleep_quality: float = Field(..., description="Sleep quality sub-score on 0-100 scale")
    mood_readiness: float = Field(..., description="Subjective mood/readiness sub-score on 0-100 scale")


class TEQSResponse(BaseModel):
    overall_teqs_score: float = Field(
        ...,
        description="Aggregated TEQS development score on a 0-100 scale",
        examples=[82.4],
    )
    environmental_subscores: EnvironmentalSubscores = Field(
        ...,
        description="Environmental domain sub-scores",
    )
    readiness_subscores: ReadinessSubscores = Field(
        ...,
        description="Athlete readiness domain sub-scores",
    )
    main_limiters: List[str] = Field(
        ...,
        description="Ordered list of top limiting factors restricting the TEQS score",
        examples=[["humidity", "sleep_duration", "temperature"]],
    )
    short_explanations: List[str] = Field(
        ...,
        description="Concise diagnostic explanations for key factor impacts",
    )
    model_status: str = Field(
        default="EXPERIMENTAL_UNVALIDATED",
        description="Validation lifecycle status flag",
    )
    research_disclaimer: str = Field(
        default="TEQS currently uses an exploratory, unvalidated development baseline. Never use for real athlete clinical or load-management decisions without supervised calibration.",
        description="Mandatory scientific disclaimer",
    )
