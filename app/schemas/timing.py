"""
app/schemas/timing.py

Pydantic schemas for the context-aware nutrition and hydration timing engine.
Component: Environmental Quality & Nutrition Timing Optimizer (TEQS)
Student ID: IT23166660
"""

from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class EnvironmentType(str, Enum):
    INDOOR_AC = "INDOOR_AC"
    INDOOR_NON_AC = "INDOOR_NON_AC"
    SEMI_OUTDOOR = "SEMI_OUTDOOR"
    OUTDOOR = "OUTDOOR"


class PreWorkoutMealCategory(str, Enum):
    FASTED_OR_EMPTY = "FASTED_OR_EMPTY"
    LESS_THAN_1_HOUR = "LESS_THAN_1_HOUR"
    ONE_TO_TWO_HOURS = "1_TO_2_HOURS"
    TWO_TO_FOUR_HOURS = "2_TO_4_HOURS"
    MORE_THAN_4_HOURS = "MORE_THAN_4_HOURS"


class TimingEvaluationRequest(BaseModel):
    experimental_teqs_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=100.0,
        description="Optional exploratory TEQS development score (0.0 to 100.0). Tagged EXPERIMENTAL_UNVALIDATED.",
        examples=[72.5],
    )
    temperature_c: Optional[float] = Field(
        None,
        ge=-10.0,
        le=60.0,
        description="Ambient temperature in Celsius (-10.0 to 60.0)",
        examples=[29.0],
    )
    humidity_pct: Optional[float] = Field(
        None,
        ge=0.0,
        le=100.0,
        description="Relative ambient humidity percentage (0.0 to 100.0)",
        examples=[75.0],
    )
    environment_type: EnvironmentType = Field(
        default=EnvironmentType.INDOOR_NON_AC,
        description="Training environment classification: INDOOR_AC, INDOOR_NON_AC, SEMI_OUTDOOR, OUTDOOR",
        examples=[EnvironmentType.INDOOR_NON_AC],
    )
    environmental_data_source: Optional[str] = Field(
        default="MANUAL",
        description="Source of weather/environment data: OPENWEATHER, SENSOR, MANUAL, NOT_SPECIFIED",
        examples=["OPENWEATHER"],
    )
    sleep_hours: Optional[float] = Field(
        None,
        ge=0.0,
        le=24.0,
        description="Prior sleep duration in hours (0.0 to 24.0)",
        examples=[7.0],
    )
    readiness_1_5: Optional[float] = Field(
        None,
        ge=1.0,
        le=5.0,
        description="Subjective readiness rating (1.0 to 5.0 scale)",
        examples=[3.5],
    )
    workout_type: str = Field(
        default="Mixed Strength",
        min_length=1,
        max_length=100,
        description="Workout modality (e.g., 'Mixed Strength', 'Squat', 'Bench Press', 'Deadlift', 'Hypertrophy', 'Conditioning')",
        examples=["Mixed Strength"],
    )
    planned_duration_min: float = Field(
        ...,
        gt=0.0,
        le=360.0,
        description="Planned workout duration in minutes (1.0 to 360.0)",
        examples=[75.0],
    )
    planned_intensity_1_10: float = Field(
        ...,
        ge=1.0,
        le=10.0,
        description="Planned session intensity rating on 1.0 to 10.0 scale",
        examples=[8.0],
    )
    pre_workout_meal_timing_category: Optional[PreWorkoutMealCategory] = Field(
        None,
        description="Time interval between last major meal and training start",
        examples=[PreWorkoutMealCategory.TWO_TO_FOUR_HOURS],
    )


class TimingPhaseGuidance(BaseModel):
    phase: str = Field(
        ...,
        description="Timing phase identifier: PRE_WORKOUT, DURING_WORKOUT, POST_WORKOUT",
        examples=["PRE_WORKOUT"],
    )
    title: str = Field(
        ...,
        description="Summary descriptive title for the guidance phase",
        examples=["Pre-Workout Preparation & Energy Availability"],
    )
    guidance: List[str] = Field(
        ...,
        description="List of qualitative, educational guidance statements",
    )
    contextual_factors: List[str] = Field(
        ...,
        description="List of contextual environmental or workout factors that triggered this guidance",
        examples=[["warm_environment", "high_intensity", "extended_duration"]],
    )
    explanation: str = Field(
        ...,
        description="Educational explanation detailing the underlying physiological or situational rationale",
    )
    evidence_status: str = Field(
        default="EDUCATIONAL_PROTOTYPE_UNVALIDATED",
        description="Scientific status tag",
    )


class TimingEvaluationResponse(BaseModel):
    validated_input_context: Dict[str, Any] = Field(
        ...,
        description="Echoed, validated, and normalized contextual inputs for transparency",
    )
    pre_workout_guidance: TimingPhaseGuidance = Field(
        ...,
        description="Educational pre-workout nutrition and fluid guidance",
    )
    during_workout_guidance: TimingPhaseGuidance = Field(
        ...,
        description="Educational during-workout hydration and comfort guidance",
    )
    post_workout_guidance: TimingPhaseGuidance = Field(
        ...,
        description="Educational post-workout recovery and replenishment guidance",
    )
    applied_rules: List[str] = Field(
        ...,
        description="List of rule IDs evaluated and applied based on the input context",
        examples=[["RULE_ENV_HEAT_HUMIDITY", "RULE_DURATION_EXTENDED", "RULE_INTENSITY_HIGH"]],
    )
    explanation: str = Field(
        ...,
        description="Comprehensive narrative summarizing how the contextual inputs influenced the guidance",
    )
    evidence_status: str = Field(
        default="EXPERIMENTAL_RULE_BASED",
        description="Overall engine evidence lifecycle status",
    )
    research_disclaimer: str = Field(
        default=(
            "This timing engine is an educational research prototype providing general sports nutrition guidance. "
            "It does NOT offer medical nutrition therapy, clinical diagnoses, or individualized dietary prescriptions. "
            "Exploratory TEQS scores are unvalidated development placeholders and must not be treated as clinical safety thresholds."
        ),
        description="Mandatory scientific advisory notice",
    )
