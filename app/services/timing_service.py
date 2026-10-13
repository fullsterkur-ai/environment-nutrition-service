"""
app/services/timing_service.py

Context-aware nutrition and hydration timing engine service.
Component: Environmental Quality & Nutrition Timing Optimizer (TEQS)
Student ID: IT23166660

Generates qualitative, explainable educational timing guidance for pre-workout,
during-workout, and post-workout phases based on contextual workout, environmental,
and readiness variables.
"""

from typing import List, Dict, Any
from app.schemas.timing import (
    TimingEvaluationRequest,
    TimingEvaluationResponse,
    TimingPhaseGuidance,
    EnvironmentType,
    PreWorkoutMealCategory,
)
from app.core.timing_rules import TIMING_RULES


class TimingService:
    """Service to evaluate context-aware nutrition and hydration timing guidance."""

    @staticmethod
    def _evaluate_triggered_rules(req: TimingEvaluationRequest) -> List[str]:
        """Evaluates situational triggers and returns list of applicable rule IDs."""
        applied_rules: List[str] = []

        # 1. Environmental Rules
        if req.environment_type == EnvironmentType.INDOOR_AC:
            applied_rules.append("RULE_ENV_INDOOR_AC")
        elif (
            (req.temperature_c is not None and req.temperature_c >= 28.0)
            or (req.humidity_pct is not None and req.humidity_pct >= 70.0)
        ) and req.environment_type in [
            EnvironmentType.INDOOR_NON_AC,
            EnvironmentType.SEMI_OUTDOOR,
            EnvironmentType.OUTDOOR,
        ]:
            applied_rules.append("RULE_ENV_HEAT_HUMIDITY")

        if (
            req.environmental_data_source == "OPENWEATHER"
            or req.environment_type in [EnvironmentType.OUTDOOR, EnvironmentType.SEMI_OUTDOOR]
        ):
            applied_rules.append("RULE_ENV_OUTDOOR_DISCLAIMER")

        # 2. Duration & Intensity Rules
        if req.planned_duration_min <= 45.0:
            applied_rules.append("RULE_DURATION_SHORT")
        elif req.planned_duration_min >= 75.0:
            applied_rules.append("RULE_DURATION_EXTENDED")

        if req.planned_intensity_1_10 >= 8.0:
            applied_rules.append("RULE_INTENSITY_HIGH")

        # 3. Meal Proximity Rules
        if req.pre_workout_meal_timing_category in [
            PreWorkoutMealCategory.FASTED_OR_EMPTY,
            PreWorkoutMealCategory.MORE_THAN_4_HOURS,
        ]:
            applied_rules.append("RULE_MEAL_FASTED_OR_REMOTE")
        elif req.pre_workout_meal_timing_category == PreWorkoutMealCategory.LESS_THAN_1_HOUR:
            applied_rules.append("RULE_MEAL_RECENT")

        # 4. Recovery & Readiness Rules
        if (req.sleep_hours is not None and req.sleep_hours < 6.0) or (
            req.readiness_1_5 is not None and req.readiness_1_5 <= 2.0
        ):
            applied_rules.append("RULE_RECOVERY_DEPRECATED")

        # 5. Exploratory TEQS Score Rules
        if req.experimental_teqs_score is not None and req.experimental_teqs_score < 60.0:
            applied_rules.append("RULE_TEQS_CONSTRAINED")

        return applied_rules

    def evaluate_timing(self, request: TimingEvaluationRequest) -> TimingEvaluationResponse:
        applied_rule_ids = self._evaluate_triggered_rules(request)

        # Build Normalized Input Context
        validated_context: Dict[str, Any] = {
            "workout_type": request.workout_type,
            "planned_duration_min": request.planned_duration_min,
            "planned_intensity_1_10": request.planned_intensity_1_10,
            "environment_type": request.environment_type.value,
            "environmental_data_source": request.environmental_data_source,
            "temperature_c": request.temperature_c,
            "humidity_pct": request.humidity_pct,
            "sleep_hours": request.sleep_hours,
            "readiness_1_5": request.readiness_1_5,
            "pre_workout_meal_timing_category": (
                request.pre_workout_meal_timing_category.value
                if request.pre_workout_meal_timing_category
                else None
            ),
            "experimental_teqs_score": request.experimental_teqs_score,
            "teqs_validation_status": (
                "EXPERIMENTAL_UNVALIDATED"
                if request.experimental_teqs_score is not None
                else "NOT_PROVIDED"
            ),
        }

        # 1. PRE-WORKOUT GUIDANCE
        pre_guidance_items: List[str] = [
            "Maintain baseline fluid readiness before training by drinking water comfortably in the hours leading to your workout to avoid starting in a fluid deficit."
        ]
        pre_factors: List[str] = ["baseline_hydration"]

        if "RULE_MEAL_FASTED_OR_REMOTE" in applied_rule_ids:
            pre_guidance_items.append(
                "Because your last meal was remote or you are training fasted, an optional easily digestible carbohydrate-rich snack 30–60 minutes prior can help support training comfort and circulating energy if well tolerated."
            )
            pre_factors.append("fasted_or_remote_meal")

        if "RULE_MEAL_RECENT" in applied_rule_ids:
            pre_guidance_items.append(
                "Because a meal was consumed recently (<1 hour), prioritize gastric comfort. Avoid consuming heavy foods or large fluid boluses immediately before lifting to prevent digestive fullness or reflux during intra-abdominal bracing."
            )
            pre_factors.append("recent_meal_proximity")

        if "RULE_ENV_HEAT_HUMIDITY" in applied_rule_ids:
            pre_guidance_items.append(
                "In warm or humid non-air-conditioned conditions, prioritize pre-session fluid availability to support early sweat onset and evaporative heat regulation."
            )
            pre_factors.append("heat_humidity_strain")

        if "RULE_INTENSITY_HIGH" in applied_rule_ids:
            pre_guidance_items.append(
                "For high-intensity exertion (intensity ≥ 8/10), ensure pre-session energy availability to support elevated anaerobic glycolytic demand."
            )
            pre_factors.append("high_intensity")

        if "RULE_RECOVERY_DEPRECATED" in applied_rule_ids:
            pre_guidance_items.append(
                "With reduced sleep or readiness, perceived exertion may be elevated. Focus on foundational hydration and easily tolerated nutrition rather than relying heavily on stimulants."
            )
            pre_factors.append("low_readiness")

        pre_explanation = (
            "Pre-workout guidance focuses on establishing adequate fluid readiness and gastrointestinal comfort "
            "conditioned on meal proximity, thermal environment, and planned workout intensity."
        )

        pre_workout = TimingPhaseGuidance(
            phase="PRE_WORKOUT",
            title="Pre-Workout Preparation & Fluid Readiness",
            guidance=pre_guidance_items,
            contextual_factors=pre_factors,
            explanation=pre_explanation,
            evidence_status="EDUCATIONAL_PROTOTYPE_UNVALIDATED",
        )

        # 2. DURING-WORKOUT GUIDANCE
        during_guidance_items: List[str] = [
            "Keep water or fluids readily accessible and drink according to thirst and comfort to maintain training stamina."
        ]
        during_factors: List[str] = ["baseline_fluid_access"]

        if "RULE_ENV_HEAT_HUMIDITY" in applied_rule_ids:
            during_guidance_items.append(
                "In warm or humid settings, sweat rates increase significantly. Drink to thirst regularly between exercise sets and seek airflow or shade between working intervals to support cooling comfort."
            )
            during_factors.append("elevated_sweat_rate")

        if "RULE_ENV_INDOOR_AC" in applied_rule_ids:
            during_guidance_items.append(
                "In climate-controlled air conditioning, cooler ambient sensations may blunt thirst cues while respiratory and metabolic fluid losses still occur. Keep fluids accessible and sip periodically."
            )
            during_factors.append("air_conditioned_environment")

        if "RULE_DURATION_SHORT" in applied_rule_ids:
            during_guidance_items.append(
                "For brief sessions (≤ 45 minutes) at moderate intensity, plain water for comfort is typically adequate without requiring intra-workout carbohydrate or electrolyte supplementation."
            )
            during_factors.append("short_duration")

        if "RULE_DURATION_EXTENDED" in applied_rule_ids:
            during_guidance_items.append(
                "For extended training exceeding 75 minutes, muscular glycogen depletion and cumulative fluid loss become more pronounced. Consider access to easily digestible carbohydrate fluids or foods to sustain training output."
            )
            during_factors.append("extended_duration")

        if "RULE_TEQS_CONSTRAINED" in applied_rule_ids:
            during_guidance_items.append(
                "Because cumulative environmental and readiness indicators are sub-optimal, rate of perceived exertion (RPE) may elevate faster than usual. Take slightly longer rest intervals between heavy working sets if necessary."
            )
            during_factors.append("teqs_limiters_active")

        during_explanation = (
            "During-workout guidance balances thermal cooling comfort, session duration demands, "
            "and hydration availability without imposing rigid fluid volume mandates."
        )

        during_workout = TimingPhaseGuidance(
            phase="DURING_WORKOUT",
            title="Intra-Workout Hydration & Session Comfort",
            guidance=during_guidance_items,
            contextual_factors=during_factors,
            explanation=during_explanation,
            evidence_status="EDUCATIONAL_PROTOTYPE_UNVALIDATED",
        )

        # 3. POST-WORKOUT GUIDANCE
        post_guidance_items: List[str] = [
            "Plan a balanced recovery meal or snack containing protein and carbohydrates according to your regular daily meal schedule to support muscle protein synthesis and recovery.",
            "Rehydrate gradually after exercise by drinking fluids to quench thirst and restoring fluid balance across subsequent hours.",
        ]
        post_factors: List[str] = ["protein_synthesis", "fluid_restoration"]

        if "RULE_ENV_HEAT_HUMIDITY" in applied_rule_ids:
            post_guidance_items.append(
                "Following workouts in warm or humid conditions with high sweat loss, include sodium/electrolytes through regular dietary foods or beverages to facilitate effective fluid retention."
            )
            post_factors.append("electrolyte_restoration")

        if "RULE_DURATION_EXTENDED" in applied_rule_ids or "RULE_INTENSITY_HIGH" in applied_rule_ids:
            post_guidance_items.append(
                "Given the high intensity or extended duration, prioritize carbohydrate-containing whole foods in your post-training meal to replenish glycogen stores."
            )
            post_factors.append("glycogen_replenishment")

        if "RULE_RECOVERY_DEPRECATED" in applied_rule_ids:
            post_guidance_items.append(
                "With reduced baseline sleep or readiness, prioritize restorative nutrition and adequate sleep in the following evening to facilitate neuromuscular recovery."
            )
            post_factors.append("neuromuscular_recovery")

        post_explanation = (
            "Post-workout recovery emphasizes gradual rehydration to thirst and balanced dietary intake "
            "supporting muscle remodeling and energy store replenishment."
        )

        post_workout = TimingPhaseGuidance(
            phase="POST_WORKOUT",
            title="Post-Workout Recovery & Rehydration",
            guidance=post_guidance_items,
            contextual_factors=post_factors,
            explanation=post_explanation,
            evidence_status="EDUCATIONAL_PROTOTYPE_UNVALIDATED",
        )

        # 4. OVERALL EXPLANATION & NARRATIVE
        overall_narrative_parts: List[str] = [
            f"Guidance tailored for a {request.planned_duration_min:.0f}-minute {request.workout_type} session "
            f"at planned intensity {request.planned_intensity_1_10:.1f}/10 in a {request.environment_type.value} setting."
        ]

        if "RULE_ENV_HEAT_HUMIDITY" in applied_rule_ids:
            overall_narrative_parts.append(
                f"Elevated thermal conditions ({request.temperature_c or 'N/A'}°C, {request.humidity_pct or 'N/A'}% humidity) "
                "increase sweat loss, making fluid availability and cooling comfort primary priorities."
            )
        elif "RULE_ENV_INDOOR_AC" in applied_rule_ids:
            overall_narrative_parts.append(
                "Indoor air conditioning provides thermal relief; baseline hydration availability remains recommended."
            )

        if "RULE_ENV_OUTDOOR_DISCLAIMER" in applied_rule_ids:
            overall_narrative_parts.append(
                "Note: Outdoor weather values represent ambient estimates and must not be interpreted as direct indoor gym measurements."
            )

        if "RULE_TEQS_CONSTRAINED" in applied_rule_ids:
            overall_narrative_parts.append(
                f"An exploratory TEQS score of {request.experimental_teqs_score:.1f}/100 reflects active situational limiters. "
                "(Note: TEQS is strictly an EXPERIMENTAL_UNVALIDATED research baseline, not a clinical threshold)."
            )

        overall_explanation = " ".join(overall_narrative_parts)

        return TimingEvaluationResponse(
            validated_input_context=validated_context,
            pre_workout_guidance=pre_workout,
            during_workout_guidance=during_workout,
            post_workout_guidance=post_workout,
            applied_rules=applied_rule_ids,
            explanation=overall_explanation,
            evidence_status="EXPERIMENTAL_RULE_BASED",
            research_disclaimer=(
                "This timing engine is an educational research prototype providing general sports nutrition guidance. "
                "It does NOT offer medical nutrition therapy, clinical diagnoses, or individualized dietary prescriptions. "
                "Exploratory TEQS scores are unvalidated development placeholders and must not be treated as clinical safety thresholds."
            ),
        )


timing_service = TimingService()
