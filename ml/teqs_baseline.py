
"""
Development-only TEQS baseline scaffold.

This file intentionally does NOT hard-code "scientifically final" weights or thresholds.
Use peer-reviewed literature + supervisor/domain feedback + real-session calibration
before finalising weights.

The baseline exists so the software architecture can be implemented before the
final research calibration is complete.
"""

from dataclasses import dataclass


@dataclass
class TEQSInputs:
    temperature_c: float
    humidity_pct: float
    air_quality_index: float
    sleep_hours: float
    sleep_quality_1_5: float
    mood_readiness_1_5: float


def clamp(x, low=0.0, high=100.0):
    return max(low, min(high, x))


def development_subscores(x: TEQSInputs):
    """
    DEVELOPMENT PLACEHOLDERS ONLY.
    These functions are deliberately simple and must be replaced/calibrated.
    They are useful for API/UI testing and explainability plumbing.
    """
    # Comfort-like scoring shapes for development, not final sports-science rules.
    temp_score = clamp(100 - abs(x.temperature_c - 27.0) * 7.0)
    humidity_score = clamp(100 - abs(x.humidity_pct - 60.0) * 1.8)
    air_score = clamp(100 - max(0, x.air_quality_index - 25) * 0.7)
    sleep_duration_score = clamp((x.sleep_hours / 8.0) * 100)
    sleep_quality_score = clamp((x.sleep_quality_1_5 / 5.0) * 100)
    mood_score = clamp((x.mood_readiness_1_5 / 5.0) * 100)

    return {
        "temperature": temp_score,
        "humidity": humidity_score,
        "air_quality": air_score,
        "sleep_duration": sleep_duration_score,
        "sleep_quality": sleep_quality_score,
        "mood_readiness": mood_score,
    }


def calculate_development_teqs(x: TEQSInputs, weights=None):
    """
    Returns a transparent placeholder TEQS for software testing.

    Do NOT use default weights in a final report as validated science.
    """
    subscores = development_subscores(x)

    if weights is None:
        # Equal weighting is intentionally neutral for development.
        # Replace only after literature review + calibration.
        weights = {k: 1.0 for k in subscores}

    total_weight = sum(weights.get(k, 0.0) for k in subscores)
    if total_weight <= 0:
        raise ValueError("At least one positive weight is required.")

    score = sum(subscores[k] * weights.get(k, 0.0) for k in subscores) / total_weight

    ordered = sorted(subscores.items(), key=lambda kv: kv[1])
    main_limiters = [name for name, value in ordered[:3]]

    return {
        "teqs": round(score, 1),
        "subscores": {k: round(v, 1) for k, v in subscores.items()},
        "main_limiters": main_limiters,
        "status": "DEVELOPMENT PLACEHOLDER — NOT CALIBRATED",
    }


if __name__ == "__main__":
    demo = TEQSInputs(
        temperature_c=35,
        humidity_pct=82,
        air_quality_index=70,
        sleep_hours=4.5,
        sleep_quality_1_5=2,
        mood_readiness_1_5=2,
    )
    print(calculate_development_teqs(demo))
