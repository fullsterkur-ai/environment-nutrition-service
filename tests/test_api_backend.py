"""
tests/test_api_backend.py

Automated Backend Verification Suite for TEQS & RPE Prediction API.
Student ID: IT23166660 - FullStrength.ai TEQS Component.

Covers:
1. Health endpoint response and metadata.
2. Valid TEQS baseline evaluation.
3. Invalid inputs rejected with HTTP 422 validation errors.
4. Synthetic model loading and inference pipeline execution.
5. Missing model artifacts returning HTTP 503 (no invented predictions).
6. API response schema integrity and scientific disclaimers.
"""

import unittest
from pathlib import Path
from fastapi.testclient import TestClient
from sklearn.pipeline import Pipeline

from app.main import app
from app.services.rpe_service import rpe_service, RPEService, ModelArtifactNotFoundError
from app.schemas.rpe import RPEPredictRequest


class TestBackendAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health_endpoint(self):
        """Verify GET /health returns 200 with service metadata and student ID."""
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)

        data = resp.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["version"], "1.0.0")
        self.assertEqual(data["student_id"], "IT23166660")
        self.assertIn("TEQS", data["component"])
        self.assertEqual(data["model_status"], "DEVELOPMENT_ONLY")
        self.assertTrue(data["rpe_model_loaded"])

    def test_teqs_evaluate_valid(self):
        """Verify POST /api/v1/teqs/evaluate returns expected baseline score and sub-scores."""
        payload = {
            "temperature_c": 35.0,
            "humidity_pct": 82.0,
            "air_quality_index": 70.0,
            "sleep_hours": 4.5,
            "sleep_quality_1_5": 2.0,
            "mood_readiness_1_5": 2.0,
        }
        resp = self.client.post("/api/v1/teqs/evaluate", json=payload)
        self.assertEqual(resp.status_code, 200)

        data = resp.json()
        self.assertEqual(data["overall_teqs_score"], 51.5)
        self.assertEqual(data["model_status"], "EXPERIMENTAL_UNVALIDATED")

        # Environmental sub-scores
        env = data["environmental_subscores"]
        self.assertEqual(env["temperature"], 44.0)
        self.assertEqual(env["humidity"], 60.4)
        self.assertEqual(env["air_quality"], 68.5)

        # Readiness sub-scores
        readiness = data["readiness_subscores"]
        self.assertEqual(readiness["sleep_duration"], 56.2)
        self.assertEqual(readiness["sleep_quality"], 40.0)
        self.assertEqual(readiness["mood_readiness"], 40.0)

        # Main limiters (top 3 lowest sub-scores)
        self.assertIsInstance(data["main_limiters"], list)
        self.assertEqual(len(data["main_limiters"]), 3)
        self.assertIn("sleep_quality", data["main_limiters"])

        # Explanations
        self.assertIsInstance(data["short_explanations"], list)
        self.assertGreater(len(data["short_explanations"]), 0)
        self.assertTrue(any("deviating from 27.0" in exp for exp in data["short_explanations"]))

        # Scientific disclaimer
        self.assertIn("exploratory, unvalidated", data["research_disclaimer"])

    def test_teqs_evaluate_invalid_inputs(self):
        """Verify invalid environmental/readiness inputs are rejected with HTTP 422."""
        base_payload = {
            "temperature_c": 28.0,
            "humidity_pct": 60.0,
            "air_quality_index": 20.0,
            "sleep_hours": 8.0,
            "sleep_quality_1_5": 4.0,
            "mood_readiness_1_5": 4.0,
        }

        # Temperature too high (> 60 C)
        p1 = dict(base_payload, temperature_c=75.0)
        self.assertEqual(self.client.post("/api/v1/teqs/evaluate", json=p1).status_code, 422)

        # Temperature too low (< -10 C)
        p2 = dict(base_payload, temperature_c=-25.0)
        self.assertEqual(self.client.post("/api/v1/teqs/evaluate", json=p2).status_code, 422)

        # Humidity negative (< 0%)
        p3 = dict(base_payload, humidity_pct=-5.0)
        self.assertEqual(self.client.post("/api/v1/teqs/evaluate", json=p3).status_code, 422)

        # Humidity over 100%
        p4 = dict(base_payload, humidity_pct=110.0)
        self.assertEqual(self.client.post("/api/v1/teqs/evaluate", json=p4).status_code, 422)

        # Negative AQI
        p5 = dict(base_payload, air_quality_index=-1.0)
        self.assertEqual(self.client.post("/api/v1/teqs/evaluate", json=p5).status_code, 422)

        # Sleep hours > 24
        p6 = dict(base_payload, sleep_hours=26.0)
        self.assertEqual(self.client.post("/api/v1/teqs/evaluate", json=p6).status_code, 422)

        # Sleep quality > 5
        p7 = dict(base_payload, sleep_quality_1_5=6.0)
        self.assertEqual(self.client.post("/api/v1/teqs/evaluate", json=p7).status_code, 422)

        # Sleep quality < 1
        p8 = dict(base_payload, sleep_quality_1_5=0.5)
        self.assertEqual(self.client.post("/api/v1/teqs/evaluate", json=p8).status_code, 422)

        # Missing required field
        p9 = dict(base_payload)
        del p9["mood_readiness_1_5"]
        self.assertEqual(self.client.post("/api/v1/teqs/evaluate", json=p9).status_code, 422)

        # Wrong datatype (string for temperature)
        p10 = dict(base_payload, temperature_c="extreme_heat")
        self.assertEqual(self.client.post("/api/v1/teqs/evaluate", json=p10).status_code, 422)

    def test_rpe_predict_valid(self):
        """Verify POST /api/v1/rpe/predict returns model prediction and DEVELOPMENT_ONLY status."""
        payload = {
            "temperature_c": 30.0,
            "humidity_pct": 75.0,
            "air_quality_index": 50.0,
            "sleep_hours": 7.0,
            "sleep_quality_1_5": 3.0,
            "mood_readiness_1_5": 3.0,
            "planned_duration_min": 60.0,
            "planned_intensity_1_10": 7.0,
            "workout_type": "Bench Press",
        }
        resp = self.client.post("/api/v1/rpe/predict", json=payload)
        self.assertEqual(resp.status_code, 200)

        data = resp.json()
        self.assertIn("predicted_session_rpe", data)
        self.assertIsInstance(data["predicted_session_rpe"], float)
        self.assertGreaterEqual(data["predicted_session_rpe"], 1.0)
        self.assertLessEqual(data["predicted_session_rpe"], 10.0)

        self.assertEqual(data["model_name"], "ridge")
        self.assertEqual(data["model_status"], "DEVELOPMENT_ONLY")
        self.assertIn("synthetic data", data["message"])
        self.assertIn("NOT been validated", data["message"])

    def test_rpe_predict_invalid_inputs(self):
        """Verify invalid workout duration and planned intensity are rejected with HTTP 422."""
        base_payload = {
            "temperature_c": 30.0,
            "humidity_pct": 75.0,
            "air_quality_index": 50.0,
            "sleep_hours": 7.0,
            "sleep_quality_1_5": 3.0,
            "mood_readiness_1_5": 3.0,
            "planned_duration_min": 60.0,
            "planned_intensity_1_10": 7.0,
            "workout_type": "Bench Press",
        }

        # Planned intensity > 10
        p1 = dict(base_payload, planned_intensity_1_10=11.0)
        self.assertEqual(self.client.post("/api/v1/rpe/predict", json=p1).status_code, 422)

        # Planned intensity < 1
        p2 = dict(base_payload, planned_intensity_1_10=0.0)
        self.assertEqual(self.client.post("/api/v1/rpe/predict", json=p2).status_code, 422)

        # Planned duration <= 0
        p3 = dict(base_payload, planned_duration_min=0.0)
        self.assertEqual(self.client.post("/api/v1/rpe/predict", json=p3).status_code, 422)

        # Planned duration > 360 min
        p4 = dict(base_payload, planned_duration_min=500.0)
        self.assertEqual(self.client.post("/api/v1/rpe/predict", json=p4).status_code, 422)

        # Empty workout type string
        p5 = dict(base_payload, workout_type="")
        self.assertEqual(self.client.post("/api/v1/rpe/predict", json=p5).status_code, 422)

    def test_model_loading(self):
        """Verify that the synthetic pipeline loads properly and is an sklearn Pipeline."""
        self.assertTrue(rpe_service.is_loaded)
        self.assertIsInstance(rpe_service._model, Pipeline)
        self.assertEqual(rpe_service.model_name, "ridge")

    def test_missing_model_artifacts_raises_503(self):
        """Verify that missing model artifacts return HTTP 503 and never fabricate predictions."""
        # 1. Direct service instantiation with invalid path
        dummy_service = RPEService(
            model_path=Path("non_existent_model_artifact.joblib")
        )
        self.assertFalse(dummy_service.is_loaded)

        dummy_req = RPEPredictRequest(
            temperature_c=25.0,
            humidity_pct=50.0,
            air_quality_index=20.0,
            sleep_hours=8.0,
            sleep_quality_1_5=4.0,
            mood_readiness_1_5=4.0,
            planned_duration_min=60.0,
            planned_intensity_1_10=7.0,
            workout_type="Squat",
        )

        with self.assertRaises(ModelArtifactNotFoundError):
            dummy_service.predict(dummy_req)

        # 2. Endpoint behavior when model is not available
        original_model = rpe_service._model
        try:
            rpe_service._model = None
            resp = self.client.post("/api/v1/rpe/predict", json=dummy_req.model_dump())
            self.assertEqual(resp.status_code, 503)
            self.assertIn("not available", resp.json()["detail"])
        finally:
            rpe_service._model = original_model

    def test_api_response_structure(self):
        """Verify rigorous response structure compliance across all API schemas."""
        # Check TEQS response schema
        teqs_resp = self.client.post(
            "/api/v1/teqs/evaluate",
            json={
                "temperature_c": 27.0,
                "humidity_pct": 60.0,
                "air_quality_index": 20.0,
                "sleep_hours": 8.0,
                "sleep_quality_1_5": 5.0,
                "mood_readiness_1_5": 5.0,
            },
        )
        self.assertEqual(teqs_resp.status_code, 200)
        teqs_data = teqs_resp.json()
        expected_teqs_keys = {
            "overall_teqs_score",
            "environmental_subscores",
            "readiness_subscores",
            "main_limiters",
            "short_explanations",
            "model_status",
            "research_disclaimer",
        }
        self.assertEqual(set(teqs_data.keys()), expected_teqs_keys)
        self.assertEqual(teqs_data["overall_teqs_score"], 100.0)

        # Check RPE response schema
        rpe_resp = self.client.post(
            "/api/v1/rpe/predict",
            json={
                "temperature_c": 27.0,
                "humidity_pct": 60.0,
                "air_quality_index": 20.0,
                "sleep_hours": 8.0,
                "sleep_quality_1_5": 5.0,
                "mood_readiness_1_5": 5.0,
                "planned_duration_min": 60.0,
                "planned_intensity_1_10": 5.0,
                "workout_type": "Mixed Strength",
            },
        )
        self.assertEqual(rpe_resp.status_code, 200)
        rpe_data = rpe_resp.json()
        expected_rpe_keys = {
            "predicted_session_rpe",
            "model_name",
            "model_variant",
            "model_version",
            "model_status",
            "message",
        }
        self.assertEqual(set(rpe_data.keys()), expected_rpe_keys)


if __name__ == "__main__":
    unittest.main()
