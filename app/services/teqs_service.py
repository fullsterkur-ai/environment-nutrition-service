"""
app/services/teqs_service.py

TEQS scoring service reusing the baseline calculation scaffold from ml/teqs_baseline.py.
Student ID: IT23166660
"""

from typing import List
from ml.teqs_baseline import TEQSInputs, calculate_development_teqs
from app.schemas.teqs import (
    TEQSRequest,
    TEQSResponse,
    EnvironmentalSubscores,
    ReadinessSubscores,
)


class TEQSService:
    """Service to evaluate Training Environmental Quality Score (TEQS)."""

    @staticmethod
    def _generate_factor_explanations(
        req: TEQSRequest, subscores: dict, limiters: List[str]
    ) -> List[str]:
        explanations: List[str] = []

        # Temperature explanation
        if req.temperature_c == 27.0:
            explanations.append("Temperature is optimal at 27.0°C (thermal neutrality score: 100.0/100).")
        else:
            explanations.append(
                f"Temperature is {req.temperature_c:.1f}°C, deviating from 27.0°C baseline "
                f"(thermal comfort sub-score: {subscores['temperature']:.1f}/100)."
            )

        # Humidity explanation
        if req.humidity_pct == 60.0:
            explanations.append("Relative humidity is optimal at 60.0% (sub-score: 100.0/100).")
        else:
            explanations.append(
                f"Relative humidity is {req.humidity_pct:.1f}%, deviating from 60.0% baseline "
                f"(humidity sub-score: {subscores['humidity']:.1f}/100)."
            )

        # Air quality explanation
        if req.air_quality_index <= 25.0:
            explanations.append(
                f"Air quality is clean (AQI {req.air_quality_index:.0f} <= 25, sub-score: {subscores['air_quality']:.1f}/100)."
            )
        else:
            explanations.append(
                f"Air Quality Index of {req.air_quality_index:.0f} exceeds 25 clean-air threshold "
                f"(air quality sub-score: {subscores['air_quality']:.1f}/100)."
            )

        # Sleep duration explanation
        if req.sleep_hours >= 8.0:
            explanations.append(
                f"Sleep duration of {req.sleep_hours:.1f}h meets/exceeds 8.0h reference standard "
                f"(sub-score: {subscores['sleep_duration']:.1f}/100)."
            )
        else:
            explanations.append(
                f"Sleep duration of {req.sleep_hours:.1f}h is below 8.0h reference standard "
                f"(sub-score: {subscores['sleep_duration']:.1f}/100)."
            )

        # Sleep quality explanation
        explanations.append(
            f"Subjective sleep quality rated {req.sleep_quality_1_5:.1f}/5.0 "
            f"(sub-score: {subscores['sleep_quality']:.1f}/100)."
        )

        # Mood readiness explanation
        explanations.append(
            f"Subjective pre-training readiness rated {req.mood_readiness_1_5:.1f}/5.0 "
            f"(sub-score: {subscores['mood_readiness']:.1f}/100)."
        )

        # Summary limiter note
        if limiters:
            explanations.insert(
                0,
                f"Primary factor constraints reducing overall score: {', '.join(limiters)}."
            )
        else:
            explanations.insert(0, "No limiting factor: every sub-score is at its 100/100 optimum.")

        return explanations

    def evaluate(self, request: TEQSRequest) -> TEQSResponse:
        inputs = TEQSInputs(
            temperature_c=request.temperature_c,
            humidity_pct=request.humidity_pct,
            air_quality_index=request.air_quality_index,
            sleep_hours=request.sleep_hours,
            sleep_quality_1_5=request.sleep_quality_1_5,
            mood_readiness_1_5=request.mood_readiness_1_5,
        )

        result = calculate_development_teqs(inputs)
        subscores = result["subscores"]
        # A factor already at 100/100 cannot be reducing the score, so it is never a limiter.
        main_limiters = [name for name in result["main_limiters"] if subscores[name] < 100.0]

        env_subscores = EnvironmentalSubscores(
            temperature=subscores["temperature"],
            humidity=subscores["humidity"],
            air_quality=subscores["air_quality"],
        )

        readiness_subscores = ReadinessSubscores(
            sleep_duration=subscores["sleep_duration"],
            sleep_quality=subscores["sleep_quality"],
            mood_readiness=subscores["mood_readiness"],
        )

        short_explanations = self._generate_factor_explanations(
            request, subscores, main_limiters
        )

        return TEQSResponse(
            overall_teqs_score=result["teqs"],
            environmental_subscores=env_subscores,
            readiness_subscores=readiness_subscores,
            main_limiters=main_limiters,
            short_explanations=short_explanations,
            model_status="EXPERIMENTAL_UNVALIDATED",
            research_disclaimer=(
                "TEQS currently uses an exploratory, unvalidated development baseline. "
                "Never use for real athlete clinical or load-management decisions without supervised calibration."
            ),
        )


teqs_service = TEQSService()
