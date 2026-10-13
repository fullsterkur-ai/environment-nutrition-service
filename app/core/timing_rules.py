"""
app/core/timing_rules.py

Rule definitions and scientific rationales for the context-aware nutrition and hydration timing engine.
Component: Environmental Quality & Nutrition Timing Optimizer (TEQS)
Student ID: IT23166660

Note: All rules are educational prototypes derived from general sports science principles.
They are NOT medical prescriptions or validated clinical algorithms.
"""

from dataclasses import dataclass
from typing import List, Dict


@dataclass(frozen=True)
class TimingRuleMetadata:
    rule_id: str
    name: str
    rationale: str
    evidence_status: str = "EDUCATIONAL_RULE_BASED"


TIMING_RULES: Dict[str, TimingRuleMetadata] = {
    "RULE_ENV_HEAT_HUMIDITY": TimingRuleMetadata(
        rule_id="RULE_ENV_HEAT_HUMIDITY",
        name="Elevated Thermal Strain Consideration",
        rationale=(
            "Ambient temperature exceeding 28°C or relative humidity exceeding 70% in non-AC or outdoor environments "
            "impairs evaporative sweat cooling, increasing thermal strain and fluid loss. Keeping fluids readily accessible "
            "and drinking according to thirst/comfort is emphasized."
        ),
    ),
    "RULE_ENV_INDOOR_AC": TimingRuleMetadata(
        rule_id="RULE_ENV_INDOOR_AC",
        name="Climate-Controlled Environment Consideration",
        rationale=(
            "Indoor air conditioning mitigates external heat stress. While thermal sensation is comfortable, "
            "respiratory and baseline fluid losses still occur; standard fluid availability remains recommended."
        ),
    ),
    "RULE_ENV_OUTDOOR_DISCLAIMER": TimingRuleMetadata(
        rule_id="RULE_ENV_OUTDOOR_DISCLAIMER",
        name="Outdoor Atmospheric Estimate Notice",
        rationale=(
            "Outdoor weather observations represent regional atmospheric estimates and do not reflect "
            "indoor gym microclimates. Environmental factors are interpreted as general situational context."
        ),
    ),
    "RULE_DURATION_SHORT": TimingRuleMetadata(
        rule_id="RULE_DURATION_SHORT",
        name="Brief Session Energy Demand",
        rationale=(
            "Workouts under 45 minutes rely primarily on resting endogenous glycogen stores. Plain water availability "
            "is typically sufficient without requiring intra-session carbohydrate ingestion."
        ),
    ),
    "RULE_DURATION_EXTENDED": TimingRuleMetadata(
        rule_id="RULE_DURATION_EXTENDED",
        name="Extended Session Glycogen and Fluid Demand",
        rationale=(
            "Sessions exceeding 75 minutes substantially deplete muscular and liver glycogen and accumulate sweat loss. "
            "Educational focus highlights access to easily digestible carbohydrates and proactive hydration pacing."
        ),
    ),
    "RULE_INTENSITY_HIGH": TimingRuleMetadata(
        rule_id="RULE_INTENSITY_HIGH",
        name="High Intensity Glycolytic Demand",
        rationale=(
            "Training at intensity ≥ 8/10 increases glycolytic flux, rate of perceived exertion (RPE), and metabolic heat. "
            "Ensuring adequate pre-workout carbohydrate availability and post-workout replenishment is prioritized."
        ),
    ),
    "RULE_MEAL_FASTED_OR_REMOTE": TimingRuleMetadata(
        rule_id="RULE_MEAL_FASTED_OR_REMOTE",
        name="Fasted or Remote Meal Energy Availability",
        rationale=(
            "Training fasted or >4 hours after the last meal may coincide with lower circulating glucose. "
            "A light, easily tolerated carbohydrate-rich snack 30–60 minutes prior can help support training comfort if tolerated."
        ),
    ),
    "RULE_MEAL_RECENT": TimingRuleMetadata(
        rule_id="RULE_MEAL_RECENT",
        name="Recent Meal Gastric Comfort Consideration",
        rationale=(
            "Eating a full meal <1 hour before high-intensity lifting can divert blood flow to digestion and cause gastrointestinal discomfort. "
            "Prioritizing fluid comfort and avoiding heavy, fibrous, or high-fat foods immediately before exercise is suggested."
        ),
    ),
    "RULE_RECOVERY_DEPRECATED": TimingRuleMetadata(
        rule_id="RULE_RECOVERY_DEPRECATED",
        name="Sub-optimal Recovery & Sleep State",
        rationale=(
            "Sleep duration <6h or subjective readiness ≤2/5 elevates perceived exertion. "
            "Athletes should emphasize baseline hydration and post-workout restorative nutrition rather than relying on heavy stimulants."
        ),
    ),
    "RULE_TEQS_CONSTRAINED": TimingRuleMetadata(
        rule_id="RULE_TEQS_CONSTRAINED",
        name="Exploratory TEQS Low Score Alert",
        rationale=(
            "An exploratory TEQS score below 60/100 indicates cumulative environmental and recovery limiters. "
            "Perceived exertion is expected to be higher. NOTE: TEQS is strictly an EXPERIMENTAL_UNVALIDATED development index."
        ),
    ),
}
