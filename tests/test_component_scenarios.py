"""
Scenario tests for the two rule-based engines (TEQS scoring and nutrition timing).

The expected values below are the outputs the engines genuinely return for these inputs
(recorded from the live API). They pin the behaviour; they are not scientific ground truth.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

TEQS_A = dict(temperature_c=27.0, humidity_pct=60.0, air_quality_index=20.0,
              sleep_hours=8.0, sleep_quality_1_5=4.5, mood_readiness_1_5=4.5)
TEQS_B = dict(temperature_c=35.0, humidity_pct=85.0, air_quality_index=40.0,
              sleep_hours=5.0, sleep_quality_1_5=2.5, mood_readiness_1_5=3.0)
TEQS_C = dict(temperature_c=27.0, humidity_pct=60.0, air_quality_index=200.0,
              sleep_hours=8.0, sleep_quality_1_5=4.0, mood_readiness_1_5=4.5)


def teqs(body):
    r = client.post("/api/v1/teqs/evaluate", json=body)
    assert r.status_code == 200
    return r.json()


def timing(body):
    r = client.post("/api/v1/timing/evaluate", json=body)
    assert r.status_code == 200
    return r.json()


# ---------------------------------------------------------------- TEQS
@pytest.mark.parametrize("body, expected", [(TEQS_A, 96.7), (TEQS_B, 60.2), (TEQS_C, 78.3)])
def test_teqs_three_scenarios_return_recorded_scores(body, expected):
    data = teqs(body)
    assert data["overall_teqs_score"] == expected
    assert data["model_status"] == "EXPERIMENTAL_UNVALIDATED"


def test_teqs_score_is_the_equal_weighted_mean_of_six_subscores():
    data = teqs(TEQS_B)
    subs = list(data["environmental_subscores"].values()) + list(data["readiness_subscores"].values())
    assert len(subs) == 6
    assert data["overall_teqs_score"] == pytest.approx(sum(subs) / 6, abs=0.1)


@pytest.mark.parametrize("field, worse, better", [
    ("sleep_hours", 4.0, 8.0),
    ("mood_readiness_1_5", 1.0, 5.0),
    ("sleep_quality_1_5", 1.0, 5.0),
    ("air_quality_index", 300.0, 10.0),
])
def test_teqs_responds_to_readiness_and_air_quality_inputs(field, worse, better):
    low = teqs(dict(TEQS_A, **{field: worse}))["overall_teqs_score"]
    high = teqs(dict(TEQS_A, **{field: better}))["overall_teqs_score"]
    assert low < high


@pytest.mark.parametrize("field, values", [("temperature_c", (20.0, 27.0, 35.0)),
                                           ("humidity_pct", (30.0, 60.0, 90.0))])
def test_teqs_comfort_curves_peak_at_the_reference_value(field, values):
    cold, ref, hot = (teqs(dict(TEQS_A, **{field: v}))["overall_teqs_score"] for v in values)
    assert ref > cold and ref > hot


def test_teqs_limiters_never_name_a_factor_that_is_already_perfect():
    data = teqs(TEQS_A)
    subs = {**data["environmental_subscores"], **data["readiness_subscores"]}
    assert data["main_limiters"], "A has imperfect sleep-quality / readiness sub-scores"
    assert all(subs[name] < 100.0 for name in data["main_limiters"])
    assert "temperature" not in data["main_limiters"]  # 100/100 in scenario A


def test_teqs_all_perfect_inputs_report_no_limiter():
    data = teqs(dict(TEQS_A, air_quality_index=10.0, sleep_quality_1_5=5.0, mood_readiness_1_5=5.0))
    assert data["overall_teqs_score"] == 100.0
    assert data["main_limiters"] == []
    assert data["short_explanations"][0].startswith("No limiting factor")


def test_teqs_explanations_match_the_subscores_that_were_returned():
    data = teqs(TEQS_C)
    text = " ".join(data["short_explanations"])
    assert data["environmental_subscores"]["air_quality"] == 0.0
    assert "Air Quality Index of 200" in text and "0.0/100" in text
    assert data["main_limiters"][0] == "air_quality"


def test_openweather_1_to_5_index_is_not_a_valid_epa_aqi_and_would_score_as_pristine():
    """Why the 1-5 OpenWeather index must never be passed in as air_quality_index."""
    for ow_index in (1.0, 3.0, 5.0):
        assert teqs(dict(TEQS_A, air_quality_index=ow_index))["environmental_subscores"]["air_quality"] == 100.0


# ---------------------------------------------------------------- nutrition timing
TIMING_A = dict(workout_type="Mixed Strength", planned_duration_min=30.0, planned_intensity_1_10=4.0,
                environment_type="INDOOR_AC", temperature_c=24.0, humidity_pct=50.0, sleep_hours=8.0,
                readiness_1_5=4.0, pre_workout_meal_timing_category="2_TO_4_HOURS")
TIMING_B = dict(workout_type="Squat", planned_duration_min=100.0, planned_intensity_1_10=9.0,
                environment_type="INDOOR_NON_AC", temperature_c=26.0, humidity_pct=55.0, sleep_hours=7.5,
                readiness_1_5=4.0, pre_workout_meal_timing_category="1_TO_2_HOURS")
TIMING_C = dict(workout_type="Mixed Strength", planned_duration_min=60.0, planned_intensity_1_10=7.0,
                environment_type="OUTDOOR", temperature_c=34.0, humidity_pct=80.0,
                environmental_data_source="OPENWEATHER", sleep_hours=7.0, readiness_1_5=3.5,
                pre_workout_meal_timing_category="LESS_THAN_1_HOUR")


@pytest.mark.parametrize("body, rules", [
    (TIMING_A, ["RULE_ENV_INDOOR_AC", "RULE_DURATION_SHORT"]),
    (TIMING_B, ["RULE_DURATION_EXTENDED", "RULE_INTENSITY_HIGH"]),
    (TIMING_C, ["RULE_ENV_HEAT_HUMIDITY", "RULE_ENV_OUTDOOR_DISCLAIMER", "RULE_MEAL_RECENT"]),
])
def test_timing_three_scenarios_trigger_the_expected_rules(body, rules):
    assert timing(body)["applied_rules"] == rules


def test_timing_guidance_changes_when_the_relevant_input_changes():
    base = timing(TIMING_A)
    assert timing(dict(TIMING_A, planned_duration_min=100.0))["applied_rules"] != base["applied_rules"]
    assert "RULE_MEAL_FASTED_OR_REMOTE" in timing(
        dict(TIMING_A, pre_workout_meal_timing_category="FASTED_OR_EMPTY"))["applied_rules"]
    assert "RULE_RECOVERY_DEPRECATED" in timing(dict(TIMING_A, sleep_hours=5.0))["applied_rules"]
    assert "RULE_TEQS_CONSTRAINED" in timing(dict(TIMING_A, experimental_teqs_score=45.0))["applied_rules"]


def test_timing_heat_rule_does_not_fire_inside_an_air_conditioned_gym():
    rules = timing(dict(TIMING_A, temperature_c=36.0, humidity_pct=90.0))["applied_rules"]
    assert "RULE_ENV_HEAT_HUMIDITY" not in rules


def test_short_session_advice_is_not_given_for_a_short_but_very_hard_session():
    rules = timing(dict(TIMING_A, planned_intensity_1_10=9.0))["applied_rules"]
    assert "RULE_DURATION_SHORT" not in rules
    assert "RULE_INTENSITY_HIGH" in rules


def test_timing_guidance_contains_no_numeric_prescriptions():
    """Educational guidance only: no millilitres, grams or per-kilogram doses."""
    data = timing(TIMING_C)
    text = " ".join(
        g for phase in ("pre_workout_guidance", "during_workout_guidance", "post_workout_guidance")
        for g in data[phase]["guidance"]
    ).lower()
    for unit in (" ml", "litre", " g/", "grams", "mg/", "g/kg"):
        assert unit not in text
