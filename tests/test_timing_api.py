"""
tests/test_timing_api.py

Automated Test Suite for Context-Aware Nutrition and Hydration Timing Engine.
Component: Environmental Quality & Nutrition Timing Optimizer (TEQS)
Student ID: IT23166660

Covers:
1. Valid timing requests with full pre/during/post guidance generation.
2. Graceful handling of missing optional inputs with appropriate defaults.
3. Input validation rejections (HTTP 422 for out-of-range duration, intensity, score, sleep).
4. Indoor AC vs Outdoor heat/humidity context-awareness and rule triggering.
5. Duration (<45 min vs >75 min) and intensity (>=8) context sensitivity.
6. Meal timing proximity (fasted vs <1 hour) and recovery (sleep/readiness) rules.
7. Exploratory TEQS score tagging as EXPERIMENTAL_UNVALIDATED (never medical threshold).
8. Strict absence of unsupported precise quantities (no ml, g, kcal, or supplement doses).
"""

import re
import unittest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.timing import EnvironmentType, PreWorkoutMealCategory


class TestTimingAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_valid_timing_request_baseline(self):
        """Verify POST /api/v1/timing/evaluate returns structured 3-phase educational guidance."""
        payload = {
            "workout_type": "Mixed Strength",
            "planned_duration_min": 60.0,
            "planned_intensity_1_10": 7.0,
            "environment_type": "INDOOR_NON_AC",
        }
        resp = self.client.post("/api/v1/timing/evaluate", json=payload)
        self.assertEqual(resp.status_code, 200)

        data = resp.json()
        self.assertIn("validated_input_context", data)
        self.assertIn("pre_workout_guidance", data)
        self.assertIn("during_workout_guidance", data)
        self.assertIn("post_workout_guidance", data)
        self.assertIn("applied_rules", data)
        self.assertIn("explanation", data)
        self.assertEqual(data["evidence_status"], "EXPERIMENTAL_RULE_BASED")
        self.assertIn("educational research prototype", data["research_disclaimer"])

        # Check phase objects
        for phase_key, expected_phase in [
            ("pre_workout_guidance", "PRE_WORKOUT"),
            ("during_workout_guidance", "DURING_WORKOUT"),
            ("post_workout_guidance", "POST_WORKOUT"),
        ]:
            phase = data[phase_key]
            self.assertEqual(phase["phase"], expected_phase)
            self.assertIsInstance(phase["guidance"], list)
            self.assertGreater(len(phase["guidance"]), 0)
            self.assertIsInstance(phase["contextual_factors"], list)
            self.assertIsInstance(phase["explanation"], str)
            self.assertEqual(phase["evidence_status"], "EDUCATIONAL_PROTOTYPE_UNVALIDATED")

    def test_missing_optional_inputs_graceful(self):
        """Verify minimal payload with only required duration and intensity succeeds cleanly."""
        payload = {
            "planned_duration_min": 45.0,
            "planned_intensity_1_10": 6.0,
        }
        resp = self.client.post("/api/v1/timing/evaluate", json=payload)
        self.assertEqual(resp.status_code, 200)

        data = resp.json()
        ctx = data["validated_input_context"]
        self.assertIsNone(ctx["temperature_c"])
        self.assertIsNone(ctx["humidity_pct"])
        self.assertIsNone(ctx["sleep_hours"])
        self.assertIsNone(ctx["readiness_1_5"])
        self.assertIsNone(ctx["experimental_teqs_score"])
        self.assertEqual(ctx["teqs_validation_status"], "NOT_PROVIDED")
        self.assertEqual(ctx["environment_type"], "INDOOR_NON_AC")

    def test_invalid_input_validation(self):
        """Verify boundary violations trigger HTTP 422 Unprocessable Entity."""
        base = {
            "workout_type": "Deadlift",
            "planned_duration_min": 60.0,
            "planned_intensity_1_10": 7.0,
        }

        # Duration <= 0
        self.assertEqual(
            self.client.post("/api/v1/timing/evaluate", json=dict(base, planned_duration_min=0.0)).status_code, 422
        )

        # Duration > 360
        self.assertEqual(
            self.client.post("/api/v1/timing/evaluate", json=dict(base, planned_duration_min=400.0)).status_code, 422
        )

        # Intensity < 1
        self.assertEqual(
            self.client.post("/api/v1/timing/evaluate", json=dict(base, planned_intensity_1_10=0.0)).status_code, 422
        )

        # Intensity > 10
        self.assertEqual(
            self.client.post("/api/v1/timing/evaluate", json=dict(base, planned_intensity_1_10=11.0)).status_code, 422
        )

        # TEQS score > 100
        self.assertEqual(
            self.client.post("/api/v1/timing/evaluate", json=dict(base, experimental_teqs_score=110.0)).status_code, 422
        )

        # TEQS score < 0
        self.assertEqual(
            self.client.post("/api/v1/timing/evaluate", json=dict(base, experimental_teqs_score=-5.0)).status_code, 422
        )

        # Sleep hours > 24
        self.assertEqual(
            self.client.post("/api/v1/timing/evaluate", json=dict(base, sleep_hours=25.0)).status_code, 422
        )

        # Readiness > 5
        self.assertEqual(
            self.client.post("/api/v1/timing/evaluate", json=dict(base, readiness_1_5=6.0)).status_code, 422
        )

        # Invalid EnvironmentType enum
        self.assertEqual(
            self.client.post("/api/v1/timing/evaluate", json=dict(base, environment_type="UNDERWATER")).status_code, 422
        )

        # Invalid MealCategory enum
        self.assertEqual(
            self.client.post("/api/v1/timing/evaluate", json=dict(base, pre_workout_meal_timing_category="YESTERDAY")).status_code, 422
        )

    def test_indoor_ac_context_handling(self):
        """Verify INDOOR_AC triggers RULE_ENV_INDOOR_AC and provides climate-control advice."""
        payload = {
            "workout_type": "Bench Press",
            "planned_duration_min": 60.0,
            "planned_intensity_1_10": 7.0,
            "environment_type": "INDOOR_AC",
        }
        resp = self.client.post("/api/v1/timing/evaluate", json=payload)
        self.assertEqual(resp.status_code, 200)

        data = resp.json()
        self.assertIn("RULE_ENV_INDOOR_AC", data["applied_rules"])
        during_text = " ".join(data["during_workout_guidance"]["guidance"])
        self.assertIn("air conditioning", during_text.lower())

    def test_outdoor_heat_humidity_context_handling(self):
        """Verify hot outdoor setting triggers RULE_ENV_HEAT_HUMIDITY and RULE_ENV_OUTDOOR_DISCLAIMER."""
        payload = {
            "workout_type": "Cardio / Conditioning",
            "planned_duration_min": 60.0,
            "planned_intensity_1_10": 7.0,
            "environment_type": "OUTDOOR",
            "environmental_data_source": "OPENWEATHER",
            "temperature_c": 32.0,
            "humidity_pct": 78.0,
        }
        resp = self.client.post("/api/v1/timing/evaluate", json=payload)
        self.assertEqual(resp.status_code, 200)

        data = resp.json()
        self.assertIn("RULE_ENV_HEAT_HUMIDITY", data["applied_rules"])
        self.assertIn("RULE_ENV_OUTDOOR_DISCLAIMER", data["applied_rules"])

        during_text = " ".join(data["during_workout_guidance"]["guidance"])
        self.assertIn("sweat", during_text.lower())
        self.assertIn("Outdoor weather values represent ambient estimates", data["explanation"])

    def test_duration_and_intensity_context(self):
        """Verify short vs extended duration and high intensity contextual rule differentiation."""
        # Short session (30 min)
        short_resp = self.client.post("/api/v1/timing/evaluate", json={
            "planned_duration_min": 30.0,
            "planned_intensity_1_10": 5.0,
        })
        self.assertEqual(short_resp.status_code, 200)
        self.assertIn("RULE_DURATION_SHORT", short_resp.json()["applied_rules"])
        during_short = " ".join(short_resp.json()["during_workout_guidance"]["guidance"])
        self.assertIn("plain water", during_short.lower())

        # Extended session (90 min) with high intensity (9.0)
        extended_resp = self.client.post("/api/v1/timing/evaluate", json={
            "planned_duration_min": 90.0,
            "planned_intensity_1_10": 9.0,
        })
        self.assertEqual(extended_resp.status_code, 200)
        self.assertIn("RULE_DURATION_EXTENDED", extended_resp.json()["applied_rules"])
        self.assertIn("RULE_INTENSITY_HIGH", extended_resp.json()["applied_rules"])
        during_ext = " ".join(extended_resp.json()["during_workout_guidance"]["guidance"])
        self.assertIn("carbohydrate", during_ext.lower())

    def test_meal_proximity_and_recovery_context(self):
        """Verify fasted vs recent meal proximity and sub-optimal sleep triggers."""
        # Fasted + low sleep
        payload_fasted = {
            "planned_duration_min": 60.0,
            "planned_intensity_1_10": 7.0,
            "pre_workout_meal_timing_category": "FASTED_OR_EMPTY",
            "sleep_hours": 5.0,
            "readiness_1_5": 2.0,
        }
        resp1 = self.client.post("/api/v1/timing/evaluate", json=payload_fasted)
        self.assertEqual(resp1.status_code, 200)
        data1 = resp1.json()
        self.assertIn("RULE_MEAL_FASTED_OR_REMOTE", data1["applied_rules"])
        self.assertIn("RULE_RECOVERY_DEPRECATED", data1["applied_rules"])
        pre1 = " ".join(data1["pre_workout_guidance"]["guidance"])
        self.assertIn("fasted", pre1.lower())

        # Recent meal (< 1 hour)
        payload_recent = {
            "planned_duration_min": 60.0,
            "planned_intensity_1_10": 7.0,
            "pre_workout_meal_timing_category": "LESS_THAN_1_HOUR",
        }
        resp2 = self.client.post("/api/v1/timing/evaluate", json=payload_recent)
        self.assertEqual(resp2.status_code, 200)
        self.assertIn("RULE_MEAL_RECENT", resp2.json()["applied_rules"])
        pre2 = " ".join(resp2.json()["pre_workout_guidance"]["guidance"])
        self.assertIn("gastric comfort", pre2.lower())

    def test_experimental_teqs_score_handling(self):
        """Verify TEQS scores below 60 trigger alert while strictly maintaining unvalidated tag."""
        payload = {
            "planned_duration_min": 60.0,
            "planned_intensity_1_10": 7.0,
            "experimental_teqs_score": 48.5,
        }
        resp = self.client.post("/api/v1/timing/evaluate", json=payload)
        self.assertEqual(resp.status_code, 200)

        data = resp.json()
        self.assertIn("RULE_TEQS_CONSTRAINED", data["applied_rules"])
        self.assertEqual(data["validated_input_context"]["teqs_validation_status"], "EXPERIMENTAL_UNVALIDATED")
        self.assertIn("EXPERIMENTAL_UNVALIDATED", data["explanation"])
        self.assertIn("not a clinical threshold", data["explanation"])

    def test_no_unsupported_precise_quantities(self):
        """Verify that guidance text never fabricates precise fluid volumes, grams, or calorie prescriptions."""
        test_cases = [
            {"planned_duration_min": 40.0, "planned_intensity_1_10": 5.0},
            {"planned_duration_min": 90.0, "planned_intensity_1_10": 9.0, "temperature_c": 33.0, "humidity_pct": 80.0},
            {"planned_duration_min": 60.0, "planned_intensity_1_10": 8.0, "pre_workout_meal_timing_category": "FASTED_OR_EMPTY"},
        ]

        forbidden_patterns = [
            re.compile(r"\b\d+\s*ml\b", re.IGNORECASE),
            re.compile(r"\b\d+\s*liters?\b", re.IGNORECASE),
            re.compile(r"\b\d+\s*g\s+protein\b", re.IGNORECASE),
            re.compile(r"\b\d+\s*grams?\b", re.IGNORECASE),
            re.compile(r"\b\d+\s*kcal\b", re.IGNORECASE),
            re.compile(r"\b\d+\s*calories\b", re.IGNORECASE),
            re.compile(r"\bcreatine\b", re.IGNORECASE),
            re.compile(r"\bpre-workout\s+supplement\b", re.IGNORECASE),
        ]

        for tc in test_cases:
            resp = self.client.post("/api/v1/timing/evaluate", json=tc)
            self.assertEqual(resp.status_code, 200)
            data = resp.json()

            all_guidance_texts = (
                data["pre_workout_guidance"]["guidance"]
                + data["during_workout_guidance"]["guidance"]
                + data["post_workout_guidance"]["guidance"]
            )

            for text in all_guidance_texts:
                for pat in forbidden_patterns:
                    match = pat.search(text)
                    self.assertIsNone(
                        match,
                        f"Found forbidden prescriptive pattern '{pat.pattern}' in guidance: '{text}'"
                    )


if __name__ == "__main__":
    unittest.main()
