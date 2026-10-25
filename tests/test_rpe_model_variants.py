"""Consistency tests for the configurable RPE model variants.

* baseline stays the default and is unchanged.
* pp1_optimized is only used when requested explicitly; invalid variants raise.
* FastAPI (RPEService) and the Streamlit dashboard produce the same prediction for the
  same input when both use the same PP1 artifact.
"""
import warnings
from pathlib import Path

import joblib
import pandas as pd
import pytest

from app.core.config import settings
from app.schemas.rpe import RPEPredictRequest
from app.services.rpe_service import (FEATURES, RPEService, ModelArtifactNotFoundError,
                                      VARIANT_BASELINE, VARIANT_PP1, rpe_service)
from ml.pp1_features import RAW_FEATURES

PP1_ARTIFACT = settings.pp1_rpe_model_path

SAMPLES = [
    dict(temperature_c=31.0, humidity_pct=78.0, air_quality_index=60.0, sleep_hours=7.0,
         sleep_quality_1_5=3.0, mood_readiness_1_5=3.0, planned_duration_min=70.0,
         planned_intensity_1_10=7.5, workout_type="Squat"),
    dict(temperature_c=22.0, humidity_pct=45.0, air_quality_index=15.0, sleep_hours=8.5,
         sleep_quality_1_5=5.0, mood_readiness_1_5=4.0, planned_duration_min=45.0,
         planned_intensity_1_10=5.0, workout_type="Bench Press"),
    dict(temperature_c=35.0, humidity_pct=90.0, air_quality_index=120.0, sleep_hours=5.0,
         sleep_quality_1_5=2.0, mood_readiness_1_5=2.0, planned_duration_min=100.0,
         planned_intensity_1_10=9.0, workout_type="Deadlift"),
]

needs_pp1 = pytest.mark.skipif(not PP1_ARTIFACT.is_file(),
                               reason="run `python -m ml.pp1_optimize` to create the PP1 artifact")


def test_feature_order_matches_training_schema():
    assert FEATURES == RAW_FEATURES


def test_default_variant_is_baseline_ridge():
    assert settings.RPE_MODEL_VARIANT == VARIANT_BASELINE
    assert rpe_service.variant == VARIANT_BASELINE
    assert rpe_service.model_name == "ridge"
    assert rpe_service.model_version.startswith("baseline+sha256:")


def test_unsupported_variant_is_rejected_not_defaulted():
    with pytest.raises(ValueError, match="Unsupported RPE_MODEL_VARIANT"):
        RPEService(variant="best_guess")


def test_selected_variant_never_falls_back_when_artifact_missing(tmp_path):
    svc = RPEService(variant=VARIANT_PP1, model_path=tmp_path / "missing.joblib")
    assert not svc.is_loaded
    assert svc.variant == VARIANT_PP1
    with pytest.raises(ModelArtifactNotFoundError):
        svc.predict(RPEPredictRequest(**SAMPLES[0]))


@needs_pp1
def test_pp1_variant_metadata_and_baseline_untouched():
    svc = RPEService(variant=VARIANT_PP1)
    assert svc.is_loaded and svc.variant == VARIANT_PP1
    assert svc.model_name == "elasticnet_fe"
    assert svc.model_version.startswith("pp1_optimized+sha256:")
    resp = svc.predict(RPEPredictRequest(**SAMPLES[0]))
    assert resp.model_status == "DEVELOPMENT_ONLY"
    assert resp.model_variant == VARIANT_PP1 and resp.model_name == "elasticnet_fe"
    # the original baseline service object is not affected
    assert rpe_service.variant == VARIANT_BASELINE and rpe_service.model_name == "ridge"


@needs_pp1
@pytest.mark.parametrize("sample", SAMPLES)
def test_fastapi_service_matches_streamlit_dashboard_for_pp1(sample):
    """Same PP1 artifact + same raw input row => same prediction in API and dashboard."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import dashboard.app as dash  # executes the Streamlit script in bare mode

    assert Path(dash.MODEL_PATH).resolve() == PP1_ARTIFACT.resolve()
    assert list(dash.RAW_FEATURES) == FEATURES

    dash_model = dash.load_model()
    dash_pred = float(dash_model.predict(pd.DataFrame([sample], columns=dash.RAW_FEATURES))[0])

    api_resp = RPEService(variant=VARIANT_PP1).predict(RPEPredictRequest(**sample))
    assert api_resp.predicted_session_rpe == round(dash_pred, 2)

    # and against a fresh direct load of the artifact (no shared cache)
    direct = joblib.load(PP1_ARTIFACT)
    assert float(direct.predict(pd.DataFrame([sample], columns=FEATURES))[0]) == pytest.approx(dash_pred, abs=1e-12)


@needs_pp1
def test_http_endpoint_with_pp1_variant_matches_dashboard(monkeypatch):
    from fastapi.testclient import TestClient
    import app.api.v1.endpoints.rpe as rpe_endpoint
    pp1_service = RPEService(variant=VARIANT_PP1)
    monkeypatch.setattr(rpe_endpoint, "rpe_service", pp1_service, raising=False)
    from app.main import app
    client = TestClient(app)
    body = client.post("/api/v1/rpe/predict", json=SAMPLES[0]).json()
    assert body["model_variant"] == VARIANT_PP1
    direct = joblib.load(PP1_ARTIFACT)
    expected = round(float(direct.predict(pd.DataFrame([SAMPLES[0]], columns=FEATURES))[0]), 2)
    assert body["predicted_session_rpe"] == expected
