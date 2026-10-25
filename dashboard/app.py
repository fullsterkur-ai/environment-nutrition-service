r"""
FullStrength.ai dashboard.

Run from the environment-nutrition-service folder:
    venv\Scripts\python.exe -m streamlit run dashboard/app.py

Tabs
  1. Session planner   - city weather (OpenWeather via the backend), session RPE prediction,
                         experimental TEQS score and nutrition timing guidance.
  2. Model accuracy    - synthetic benchmark, real lifters and open-data results, kept separate.
  3. Methods and status - what is built, what is validated, datasets and safeguards.

The planner calls the FastAPI backend (FULLSTRENGTH_API_URL, default http://127.0.0.1:8000).
If the backend is not running, weather, TEQS and nutrition timing fall back to the same service
code in-process and say so.
"""
import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import httpx
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.core.timing_rules import TIMING_RULES  # noqa: E402
from ml.pp1_features import RAW_FEATURES  # noqa: E402

EXP = ROOT / "artifacts" / "pp1_model_optimization"
REAL_EVAL = ROOT / "artifacts" / "real_athlete_evaluation"
OPEN_DATA = ROOT / "artifacts" / "open_dataset_feasibility"
EPOCH_DIR = ROOT / "artifacts" / "training_curves"
MODEL_PATH = EXP / "best_rpe_model_pp1.joblib"
DATA_CSV = ROOT / "data" / "TEQS_Synthetic_5000_Development_Dataset.csv"

API_URL = os.getenv("FULLSTRENGTH_API_URL", "http://127.0.0.1:8000").rstrip("/")

# Black / navy palette (blue tints only)
BLUE = "#2F6BFF"
SKY = "#8FA6D8"
PALE = "#E6ECF8"
GRID = "#1B2A47"

st.set_page_config(
    page_title="FullStrength.ai",
    page_icon=":material/fitness_center:",
    layout="wide",
)

# Only custom CSS: the CR-10 ruler (no native element draws a marked scale).
st.html("""
<style>
.cr10 { margin:2rem 0 .2rem; }
.cr10-track { position:relative; height:.8rem; border-radius:.4rem; border:1px solid #1B2A47;
  background:linear-gradient(90deg,#0B1220 0%,#0B2A5B 40%,#2F6BFF 75%,#9DB8E8 100%); }
.cr10-band { position:absolute; top:-3px; bottom:-3px; background:rgba(230,236,248,.22); border-radius:.3rem; }
.cr10-pin { position:absolute; top:-.55rem; height:1.9rem; width:3px; background:#E6ECF8; border-radius:2px; }
.cr10-pin::after { content:attr(data-v); position:absolute; top:-1.5rem; left:50%; transform:translateX(-50%);
  padding:.05rem .4rem; background:#E6ECF8; color:#05070D; border-radius:.3rem; font-size:.75rem; font-weight:600; }
.cr10-ticks { position:relative; height:2.2rem; margin-top:.5rem; }
.cr10-ticks div { position:absolute; top:0; width:9%; font-size:.66rem; color:#8C9AB5; line-height:1.2; }
.cr10-ticks div.end { text-align:right; }
.cr10-ticks b { display:block; font-size:.85rem; color:#E6ECF8; font-weight:600; }
</style>
""")

CR10_ANCHORS = ["Rest", "Very easy", "Easy", "Moderate", "Somewhat hard", "Hard", "", "Very hard", "", "", "Maximal"]

ENV_TYPES = {
    "Indoor, air-conditioned": "INDOOR_AC",
    "Indoor, no air conditioning": "INDOOR_NON_AC",
    "Semi-outdoor / covered": "SEMI_OUTDOOR",
    "Outdoor": "OUTDOOR",
}
MEAL_TIMING = {
    "More than 4 hours ago": "MORE_THAN_4_HOURS",
    "2 to 4 hours ago": "2_TO_4_HOURS",
    "1 to 2 hours ago": "1_TO_2_HOURS",
    "Less than 1 hour ago": "LESS_THAN_1_HOUR",
    "Fasted / empty stomach": "FASTED_OR_EMPTY",
}
FEATURE_LABELS = {
    "temperature_c": "Temperature", "humidity_pct": "Humidity", "air_quality_index": "Air quality (AQI)",
    "sleep_hours": "Sleep duration", "sleep_quality_1_5": "Sleep quality", "mood_readiness_1_5": "Readiness",
    "planned_duration_min": "Planned duration", "planned_intensity_1_10": "Planned intensity",
    "workout_type": "Workout type",
}
TEQS_LABELS = {
    "temperature": "Temperature", "humidity": "Humidity", "air_quality": "Air quality",
    "sleep_duration": "Sleep duration", "sleep_quality": "Sleep quality", "mood_readiness": "Readiness",
}
TAGS = {
    "syn": ":gray-badge[Synthetic data benchmark]",
    "real": ":blue-badge[Real athletes]",
    "open": ":violet-badge[Open data]",
}
OFFLINE = "Backend not reachable"


# ----------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_PATH.is_file():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_ranges():
    df = pd.read_csv(DATA_CSV)
    num = [c for c in RAW_FEATURES if c != "workout_type"]
    ref = {c: float(df[c].median()) for c in num}
    ref["workout_type"] = df["workout_type"].mode().iloc[0]
    return ref, sorted(df["workout_type"].unique())


def read_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def read_csv(p: Path):
    return pd.read_csv(p) if p.is_file() else None


def clip(value, low, high):
    return min(max(float(value), low), high)


def api(method: str, path: str, timeout: float = 10.0, **kwargs):
    """Call the backend. Returns (json, None) on success or (None, message) on any failure."""
    try:
        resp = httpx.request(method, f"{API_URL}{path}", timeout=timeout, **kwargs)
    except httpx.HTTPError as exc:
        return None, f"{OFFLINE} at {API_URL} ({exc.__class__.__name__})."
    if resp.status_code != 200:
        try:
            detail = resp.json().get("detail", resp.text)
        except ValueError:
            detail = resp.text
        return None, f"HTTP {resp.status_code}: {detail}"
    return resp.json(), None


def run_engine(name: str, payload: dict):
    """TEQS / timing through the backend; same service code in-process if the backend is down.

    Returns (response dict, source label, error message).
    """
    path = {"teqs": "/api/v1/teqs/evaluate", "timing": "/api/v1/timing/evaluate"}[name]
    data, err = api("POST", path, json=payload)
    if data is not None:
        return data, "backend API", None
    if not err.startswith(OFFLINE):
        return None, None, err
    if name == "teqs":
        from app.schemas.teqs import TEQSRequest
        from app.services.teqs_service import teqs_service
        out = teqs_service.evaluate(TEQSRequest(**payload))
    else:
        from app.schemas.timing import TimingEvaluationRequest
        from app.services.timing_service import timing_service
        out = timing_service.evaluate_timing(TimingEvaluationRequest(**payload))
    return out.model_dump(mode="json"), "in-process (backend offline)", None


def fetch_weather_data(params: dict):
    """OpenWeather through the backend; the same weather service in-process if the backend is down."""
    data, err = api("GET", "/api/v1/environment/current", params=params)
    if data is not None or not err.startswith(OFFLINE):
        return data, err
    try:
        from app.services.weather_service import weather_service
        res = asyncio.run(weather_service.get_current_environment(
            city=params["city"], country_code=params.get("country_code")))
        return res.model_dump(mode="json"), None
    except Exception as exc:  # HTTPException from the service carries a readable detail
        return None, str(getattr(exc, "detail", exc))


def style_fig(fig, height=330):
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(color=PALE, family="Inter, sans-serif"),
                      legend=dict(orientation="h", y=-0.22, x=0))
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, automargin=True)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, automargin=True)
    return fig


def show_fig(fig, alt, height=330):
    st.plotly_chart(style_fig(fig, height), theme=None, width="stretch", alt=alt)


def kpi(label, value, sub="", tag=None):
    with st.container(border=True):
        if tag:
            st.markdown(TAGS[tag])
        st.metric(label, value)
        if sub:
            st.caption(sub)


def cr10_ruler(pred=None, band=None):
    pin = ""
    if pred is not None:
        p = min(max(pred, 0.0), 10.0) * 10
        if band:
            lo, hi = max(pred - band, 0.0) * 10, min(pred + band, 10.0) * 10
            pin += f'<div class="cr10-band" style="left:{lo:.1f}%;width:{hi - lo:.1f}%"></div>'
        pin += f'<div class="cr10-pin" style="left:{p:.1f}%" data-v="{pred:.1f}"></div>'
    ticks = "".join(
        f'<div class="end" style="right:0"><b>{i}</b>{a}</div>' if i == 10
        else f'<div style="left:{i * 10}%"><b>{i}</b>{a}</div>'
        for i, a in enumerate(CR10_ANCHORS))
    return f'<div class="cr10"><div class="cr10-track">{pin}</div><div class="cr10-ticks">{ticks}</div></div>'


def pretty(name: str) -> str:
    return name.replace("_fe", " (engineered)").replace("_raw", " (raw inputs)").replace("_", " ").title()


model = load_model()
ref, workout_types = load_ranges()
res_metrics = read_json(EXP / "final_holdout_metrics.json")
interp_metrics = read_json(EXP / "interpretable_metrics.json")

# ----------------------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------------------
head_l, head_r = st.columns([3, 2], vertical_alignment="bottom")
with head_l:
    st.title("FullStrength.ai")
    st.caption("Environmental quality and nutrition timing for strength training")
with head_r:
    health, health_err = api("GET", "/health", timeout=3.0)
    if health:
        st.caption(f":material/check_circle: Backend online · {API_URL}")
    else:
        st.caption(f":material/cloud_off: {health_err} Weather, TEQS and timing will run in-process.")

tab_plan, tab_acc, tab_method = st.tabs([
    ":material/tune: Session planner",
    ":material/query_stats: Model accuracy",
    ":material/fact_check: Methods and status",
])

# =========================================================================================
# 1. SESSION PLANNER
# =========================================================================================
SS = st.session_state
for key, default in {"temp": 30.0, "hum": 75.0, "aqi": 50, "sleep_h": 7.0, "sleep_q": 3, "mood": 3,
                     "dur": 60, "inten": 7.5, "ow_city": "Colombo", "ow_cc": "LK", "ow_data": None,
                     "ow_error": None}.items():
    SS.setdefault(key, default)


def fetch_weather():
    params = {"city": SS.ow_city.strip()}
    if SS.ow_cc.strip():
        params["country_code"] = SS.ow_cc.strip().upper()
    data, err = fetch_weather_data(params)
    SS.ow_error = err
    if data is not None:
        SS.ow_data = data
        SS.temp = clip(data["temperature_c"], -10.0, 60.0)
        SS.hum = clip(data["humidity_pct"], 0.0, 100.0)


with tab_plan:
    st.markdown("Estimate how hard today's session will feel, check the training environment and get "
                "nutrition timing guidance. Change any input and every result below updates.")
    left, right = st.columns([1, 1.45], gap="large")

    with left:
        with st.container(border=True):
            st.subheader("Location")
            c_city, c_cc = st.columns([3, 1])
            c_city.text_input("City", key="ow_city")
            c_cc.text_input("Country", key="ow_cc", max_chars=2, help="Optional two-letter code, e.g. LK")
            st.button("Fetch live weather", on_click=fetch_weather, icon=":material/cloud_download:", width="stretch")
            ow = SS.ow_data
            if SS.ow_error:
                st.warning(SS.ow_error)
            elif ow:
                aqi_txt = (f"Outdoor air quality {ow['openweather_aqi_label']} (OpenWeather {ow['openweather_aqi_category']}/5)"
                           + (f", PM2.5 {ow['pm2_5_ug_m3']} µg/m³" if ow.get("pm2_5_ug_m3") is not None else ""))
                st.info(f"**{ow['location']['city']}, {ow['location']['country']}** · {ow['temperature_c']} °C, "
                        f"{ow['humidity_pct']:.0f}% humidity, {ow['weather_description']}. {aqi_txt}. "
                        f"Observed {ow['observed_at_utc']}{' (cached)' if ow['cached'] else ''}.",
                        icon=":material/location_on:")
                st.caption("Outdoor estimate, not the air inside your gym. OpenWeather's 1–5 air-quality index uses a "
                           "different scale from the AQI field below, so it is never copied into it.")

        with st.container(border=True):
            st.subheader("Conditions")
            c1, c2 = st.columns(2)
            c1.number_input("Temperature (°C)", -10.0, 60.0, step=0.5, key="temp")
            c2.number_input("Humidity (%)", 0.0, 100.0, step=1.0, key="hum")
            aqi = st.number_input("Air quality index (US EPA scale, 0–500)", 0, 500, step=5, key="aqi",
                                  help="Enter a US EPA AQI value. Under 25 is treated as clean air.")
            env_label = st.selectbox("Where will you train?", list(ENV_TYPES), index=1)

        with st.container(border=True):
            st.subheader("Readiness")
            sleep_h = st.number_input("Sleep last night (hours)", 0.0, 14.0, step=0.5, key="sleep_h")
            c3, c4 = st.columns(2)
            sleep_q = c3.slider("Sleep quality (1–5)", 1, 5, key="sleep_q")
            mood = c4.slider("Readiness (1–5)", 1, 5, key="mood")

        with st.container(border=True):
            st.subheader("Workout")
            default_ix = workout_types.index("Squat") if "Squat" in workout_types else 0
            workout = st.selectbox("Workout type", workout_types, index=default_ix)
            c5, c6 = st.columns(2)
            dur = c5.number_input("Planned duration (min)", 15, 240, step=5, key="dur")
            inten = c6.slider("Planned intensity (1–10)", 1.0, 10.0, step=0.5, key="inten")
            meal_label = st.selectbox("Last substantial meal", list(MEAL_TIMING), index=1)
            use_teqs = st.toggle("Use the TEQS score in nutrition timing", value=True,
                                 help="Passes the experimental TEQS score to the timing engine, which has one rule that reads it.")

    temp, hum = SS.temp, SS.hum
    env_source = ("OPENWEATHER" if SS.ow_data
                  and temp == clip(SS.ow_data["temperature_c"], -10.0, 60.0)
                  and hum == clip(SS.ow_data["humidity_pct"], 0.0, 100.0) else "MANUAL")

    with right:
        # ---- 1. RPE ------------------------------------------------------------------
        st.subheader("Predicted session RPE")
        row = {"temperature_c": temp, "humidity_pct": hum, "air_quality_index": aqi, "sleep_hours": sleep_h,
               "sleep_quality_1_5": sleep_q, "mood_readiness_1_5": mood, "planned_duration_min": dur,
               "planned_intensity_1_10": inten, "workout_type": workout}
        if model is None or res_metrics is None:
            st.warning("RPE model file not found: artifacts/pp1_model_optimization/best_rpe_model_pp1.joblib")
        else:
            X = pd.DataFrame([row], columns=RAW_FEATURES)
            pred_rpe = float(model.predict(X)[0])
            mae = res_metrics["holdout_evaluations"]["sealed_holdout/final_model"]["MAE"]
            with st.container(border=True):
                m1, m2 = st.columns([1, 2.4], vertical_alignment="center")
                m1.metric("Predicted RPE", f"{pred_rpe:.1f} / 10")
                m2.markdown(TAGS["syn"])
                m2.caption(f"Model: {pretty(res_metrics['best_model_name'])}, trained on synthetic development data. "
                           f"Shaded band = typical error on synthetic test sessions (±{mae:.2f}). Note: Synthetic results do NOT represent real-world accuracy.")
                st.html(cr10_ruler(pred_rpe, mae))
            st.caption("Foster modified CR-10 session RPE: one rating for the whole session, given after training. "
                       "On real lifters, models that use wellness answers alone did not beat a simple average "
                       "(see Model accuracy).")
            if health and health["rpe_model_variant"] != "pp1_optimized":
                st.caption(f"Note: the backend currently serves the '{health['rpe_model_name']}' model "
                           f"({health['rpe_model_variant']}), so the mobile app and API can differ slightly from this "
                           "prediction. Start the backend with RPE_MODEL_VARIANT=pp1_optimized to make them identical.")

            with st.expander("What moved the prediction"):
                rows = []
                for f in RAW_FEATURES:
                    alt_row = X.copy()
                    alt_row[f] = ref[f]
                    rows.append((f, pred_rpe - float(model.predict(alt_row)[0])))
                rows.sort(key=lambda t: abs(t[1]))
                fig = go.Figure(go.Bar(
                    x=[d for _, d in rows], y=[f"{FEATURE_LABELS[f]} = {row[f]}" for f, _ in rows], orientation="h",
                    marker_color=[BLUE if d > 0 else SKY for _, d in rows]))
                fig.update_layout(xaxis_title="Change in predicted RPE versus a typical session")
                show_fig(fig, "Input factors that moved the predicted session RPE", height=300)

        # ---- 2. TEQS -----------------------------------------------------------------
        st.subheader("Training environment quality (TEQS)")
        teqs_payload = {"temperature_c": temp, "humidity_pct": hum, "air_quality_index": float(aqi),
                        "sleep_hours": sleep_h, "sleep_quality_1_5": float(sleep_q), "mood_readiness_1_5": float(mood)}
        teqs, teqs_src, teqs_err = run_engine("teqs", teqs_payload)
        teqs_score = None
        if teqs_err:
            st.warning(f"TEQS request failed: {teqs_err}")
        else:
            teqs_score = teqs["overall_teqs_score"]
            with st.container(border=True):
                t1, t2 = st.columns([1, 2.4], vertical_alignment="center")
                t1.metric("TEQS", f"{teqs_score:.1f} / 100")
                t2.markdown(":gray-badge[Experimental, not validated]")
                limiters = ", ".join(TEQS_LABELS[k] for k in teqs["main_limiters"]) or "none"
                t2.caption(f"Equal-weighted average of six sub-scores. Lowest sub-scores: {limiters}. Source: {teqs_src}.")
                subs = {**teqs["environmental_subscores"], **teqs["readiness_subscores"]}
                fig = go.Figure(go.Bar(x=list(subs.values())[::-1], y=[TEQS_LABELS[k] for k in subs][::-1],
                                       orientation="h", marker_color=BLUE))
                fig.update_layout(xaxis=dict(range=[0, 100], title="Sub-score (0–100)"))
                show_fig(fig, "TEQS sub-scores", height=260)
            with st.expander("Why this score"):
                for line in teqs["short_explanations"]:
                    st.markdown(f"- {line}")
                st.caption(teqs["research_disclaimer"])

        # ---- 3. Nutrition timing -----------------------------------------------------
        st.subheader("Nutrition and hydration timing")
        timing_payload = {
            "workout_type": workout, "planned_duration_min": float(dur), "planned_intensity_1_10": float(inten),
            "environment_type": ENV_TYPES[env_label], "temperature_c": temp, "humidity_pct": hum,
            "environmental_data_source": env_source, "sleep_hours": sleep_h, "readiness_1_5": float(mood),
            "pre_workout_meal_timing_category": MEAL_TIMING[meal_label],
        }
        if use_teqs and teqs_score is not None:
            timing_payload["experimental_teqs_score"] = teqs_score
        timing, timing_src, timing_err = run_engine("timing", timing_payload)
        if timing_err:
            st.warning(f"Timing request failed: {timing_err}")
        else:
            st.markdown(":gray-badge[Educational, rule-based, not medical advice]")
            t_pre, t_during, t_post = st.tabs(["Before", "During", "After"])
            for tab, key in ((t_pre, "pre_workout_guidance"), (t_during, "during_workout_guidance"),
                             (t_post, "post_workout_guidance")):
                with tab:
                    phase = timing[key]
                    for line in phase["guidance"]:
                        st.markdown(f"- {line}")
                    st.caption(phase["explanation"])
            st.markdown("**Rules that produced this guidance:** "
                        + " · ".join(TIMING_RULES[r].name if r in TIMING_RULES else r for r in timing["applied_rules"]))
            with st.expander("Rule details and rationale"):
                for r in timing["applied_rules"]:
                    meta = TIMING_RULES.get(r)
                    st.markdown(f"**{meta.name if meta else r}** (`{r}`)  \n{meta.rationale if meta else ''}")
                st.markdown(f"**Summary.** {timing['explanation']}")
                st.caption(f"Source: {timing_src}. Environment data source: {env_source}. {timing['research_disclaimer']}")

# =========================================================================================
# 2. MODEL ACCURACY
# =========================================================================================
with tab_acc:
    view = st.segmented_control("Evidence", ["Summary", "Synthetic data benchmark", "Real lifters", "Open data"],
                                default="Summary", label_visibility="collapsed", key="acc_view") or "Summary"

    eval_summary = read_json(REAL_EVAL / "real_athlete_summary.json")
    cmp_df = read_csv(REAL_EVAL / "real_athlete_comparison.csv")
    warm = read_json(OPEN_DATA / "warm_start_lopo.json")
    NICE_MODEL = {"dummy_mean": "Mean guess (baseline)", "ridge_wellness": "Ridge", "elasticnet_wellness": "ElasticNet",
                  "random_forest_wellness": "Random forest (depth 3)"}

    if res_metrics is None or interp_metrics is None:
        st.warning("Model evaluation artifacts not found in artifacts/pp1_model_optimization.")
    else:
        ev = res_metrics["holdout_evaluations"]
        sealed = ev["sealed_holdout/final_model"]
        base_sealed = ev["sealed_holdout/baseline_recipe_ridge_raw_alpha1"]
        dummy_sealed = ev["sealed_holdout/dummy_mean"]
        ci = res_metrics["sealed_holdout_R2_participant_bootstrap_95CI"]
        dev_oof = interp_metrics["synthetic_dev_groupkfold_oof"]["final_model"]
        dev_oof_dummy = interp_metrics["synthetic_dev_groupkfold_oof"]["mean_predictor"]
        dev_oof_ridge = interp_metrics["synthetic_dev_groupkfold_oof"]["ridge_raw_baseline"]
        sealed_within = interp_metrics["synthetic_sealed_holdout"]["final_model"]["within_pm1"]
        sealed_dummy_within = interp_metrics["synthetic_sealed_holdout"]["mean_predictor_fit_on_dev"]["within_pm1"]
        dev_within, dev_dummy_within = dev_oof["within_pm1"], dev_oof_dummy["within_pm1"]
        dev_cv = res_metrics.get("dev_cv", {})

    # ------------------------------------------------------------------ Summary
    if view == "Summary":
        st.subheader("How accurate is the RPE model?")
        st.markdown(
            "Accuracy here means **the share of sessions predicted within ±1 RPE point** of the rating the athlete gave. "
            "Every row is compared with a *mean guess* (always predict the average RPE). "
            "A model is only useful if it beats that guess.")
        rows = []
        if res_metrics and interp_metrics:
            rows.append({"Data": "Synthetic data benchmark", "Held-out set": "50 test athletes · 1,000 sessions",
                         "Model": "ElasticNet, 9 pre-session inputs", "Within ±1 RPE": f"{sealed_within['pct']:.1f}%",
                         "Mean guess": f"{sealed_dummy_within['pct']:.1f}%",
                         "Reading": "Learns simulation rules (+%.1f points over dummy). NOT real accuracy." % (sealed_within["pct"] - sealed_dummy_within["pct"])})
        if cmp_df is not None:
            dummy_row = cmp_df[cmp_df.model == "dummy_mean"].iloc[0]
            models_only = cmp_df[cmp_df.model != "dummy_mean"]
            best = models_only.loc[models_only.pmdata_lopo_within_1_pct.idxmax()]
            rows.append({"Data": "Real lifters (PMData)", "Held-out set": "10 athletes, each left out in turn · 87 sessions",
                         "Model": f"Best of 3 wellness-only models ({NICE_MODEL[best.model]})",
                         "Within ±1 RPE": f"{best.pmdata_lopo_within_1_pct:.1f}% ({best.pmdata_lopo_within_1_count})",
                         "Mean guess": f"{dummy_row.pmdata_lopo_within_1_pct:.1f}% ({dummy_row.pmdata_lopo_within_1_count})",
                         "Reading": "No real gain over the mean guess; R² below zero"})
            best_ext = models_only.loc[models_only.external_test_within_1_pct.idxmax()]
            rows.append({"Data": "Real lifters (FullStrength survey)", "Held-out set": "13 unseen athletes · 13 sessions",
                         "Model": "Wellness-only models trained on PMData",
                         "Within ±1 RPE": f"{best_ext.external_test_within_1_pct:.1f}% ({best_ext.external_test_within_1_count})",
                         "Mean guess": f"{dummy_row.external_test_within_1_pct:.1f}% ({dummy_row.external_test_within_1_count})",
                         "Reading": "No gain; far too few sessions to conclude"})
        if warm:
            rob, rem = warm["Robinson2025_OSF_58hy3"], warm["Remmert2026_OSF_5wprn"]
            hyb, wo, dm = "ElasticNet wellness + personal history", "ElasticNet wellness only", "Dummy (training mean)"
            rows.append({"Data": "Open data (Robinson / Remmert)", "Held-out set": "65 athletes, each left out in turn · 1,559 sessions",
                         "Model": "Wellness answers only",
                         "Within ±1 RPE": f"{rob[wo]['within1']:.0f}% / {rem[wo]['within1']:.0f}%",
                         "Mean guess": f"{rob[dm]['within1']:.0f}% / {rem[dm]['within1']:.0f}%",
                         "Reading": "No gain: wellness alone does not predict RPE"})
            rows.append({"Data": "Open data (Robinson / Remmert)", "Held-out set": "same sessions",
                         "Model": "Wellness + the athlete's own earlier ratings",
                         "Within ±1 RPE": f"{rob[hyb]['within1']:.0f}% / {rem[hyb]['within1']:.0f}%",
                         "Mean guess": f"{rob[dm]['within1']:.0f}% / {rem[dm]['within1']:.0f}%",
                         "Reading": "Clear gain (about +20 to +30 points). Exploratory"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                     alt="Accuracy summary across synthetic, real and open datasets, each against a mean-guess baseline")
        st.info("On benchmark data the pipeline works. On real lifters, wellness answers alone predict no better than the "
                "average. What helps is knowing the lifter's own previous ratings. No real dataset has gym temperature, "
                "humidity or air quality, so the environmental effect on RPE has not been tested on real athletes.",
                title="How to read this", icon=":material/info:")
        st.caption("Two numbers separated by a slash are Robinson / Remmert. Open-data rows use each athlete's 6th and later "
                   "sessions only (545 + 1,014 = 1,559 sessions). Synthetic results are never mixed with real-athlete results.")

    # ------------------------------------------------------------------ Synthetic data benchmark
    elif view == "Synthetic data benchmark" and res_metrics and interp_metrics:
        st.subheader("Synthetic Data Benchmark (Simulation)")
        st.markdown(TAGS["syn"])
        st.caption("5,000 structured sessions from 250 parameterized athletic profiles: 3,000 for development (5-fold CV grouped by athlete) "
                   "and 1,000 sealed for one final test. NOTE: The 86.1% result below was measured on synthetic simulation data to prove mathematical algorithm convergence; it is NOT accuracy on real human athletes.")
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            kpi("Within ±1 RPE (sealed test)", f"{sealed_within['pct']:.1f}%",
                f"{sealed_within['count_within']} / {sealed_within['n']} sessions", "syn")
        with k2:
            kpi("Mean-guess baseline", f"{sealed_dummy_within['pct']:.1f}%",
                f"model gains +{sealed_within['pct'] - sealed_dummy_within['pct']:.1f} points", "syn")
        with k3:
            kpi("R² (sealed test)", f"{sealed['R2']:.3f}", f"95% CI {ci[0]:.3f} to {ci[1]:.3f}; 0.60 target not reached", "syn")
        with k4:
            kpi("MAE / RMSE", f"{sealed['MAE']:.2f} / {sealed['RMSE']:.2f}", "RPE points, unseen athletes", "syn")

        st.markdown("##### Model comparison")
        st.dataframe(pd.DataFrame([
            {"Model": "ElasticNet + engineered features (final)", "Split": "Sealed test, 50 athletes",
             "R²": f"{sealed['R2']:.3f}", "MAE": f"{sealed['MAE']:.3f}", "RMSE": f"{sealed['RMSE']:.3f}",
             "Within ±1": f"{sealed_within['pct']:.1f}%"},
            {"Model": "Ridge, 9 raw inputs", "Split": "Sealed test, 50 athletes",
             "R²": f"{base_sealed['R2']:.3f}", "MAE": f"{base_sealed['MAE']:.3f}", "RMSE": f"{base_sealed['RMSE']:.3f}",
             "Within ±1": "85.8%"},
            {"Model": "Mean guess", "Split": "Sealed test, 50 athletes",
             "R²": f"{dummy_sealed['R2']:.4f}", "MAE": f"{dummy_sealed['MAE']:.3f}", "RMSE": f"{dummy_sealed['RMSE']:.3f}",
             "Within ±1": f"{sealed_dummy_within['pct']:.1f}%"},
            {"Model": "ElasticNet + engineered features (final)", "Split": "5-fold CV, 150 athletes",
             "R²": f"{dev_cv.get('R2_pooled_oof', dev_oof['R2']):.3f}", "MAE": f"{dev_oof['MAE']:.3f}",
             "RMSE": f"{dev_oof['RMSE']:.3f}", "Within ±1": f"{dev_within['pct']:.1f}%"},
            {"Model": "Ridge, 9 raw inputs", "Split": "5-fold CV, 150 athletes",
             "R²": f"{dev_oof_ridge['R2']:.3f}", "MAE": f"{dev_oof_ridge['MAE']:.3f}", "RMSE": f"{dev_oof_ridge['RMSE']:.3f}",
             "Within ±1": f"{dev_oof_ridge['within_pm1']['pct']:.1f}%"},
            {"Model": "Mean guess", "Split": "5-fold CV, 150 athletes",
             "R²": f"{dev_oof_dummy['R2']:.4f}", "MAE": f"{dev_oof_dummy['MAE']:.3f}", "RMSE": f"{dev_oof_dummy['RMSE']:.3f}",
             "Within ±1": f"{dev_dummy_within['pct']:.1f}%"},
        ]), hide_index=True, width="stretch", alt="Synthetic data benchmark model comparison")
        st.caption(f"Training R² {res_metrics.get('train_R2', 0):.3f} against CV R² {dev_cv.get('R2_pooled_oof', 0):.3f}: "
                   "a very small gap, so the model is not overfitting the synthetic training athletes.")

        avp = read_csv(EXP / "actual_vs_predicted_holdout.csv")
        lc = read_csv(EXP / "learning_curve.csv")
        fi = read_csv(EXP / "feature_importance.csv")
        g1, g2 = st.columns(2)
        with g1:
            if avp is not None:
                sh = avp[avp["holdout"] == "sealed_holdout"]
                st.markdown("##### Actual vs predicted (sealed test)")
                f = px.scatter(sh, x="actual_rpe", y="predicted_rpe", opacity=0.45, color_discrete_sequence=[BLUE],
                               labels={"actual_rpe": "Actual RPE", "predicted_rpe": "Predicted RPE"})
                lo, hi = float(sh.actual_rpe.min()), float(sh.actual_rpe.max())
                f.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color=PALE, dash="dash")))
                f.update_layout(showlegend=False)
                show_fig(f, "Actual versus predicted RPE on the sealed benchmark test")
        with g2:
            if lc is not None:
                st.markdown("##### Learning curve (grouped by athlete)")
                f = go.Figure()
                f.add_trace(go.Scatter(x=lc["train_rows_per_fold"], y=lc["train_R2_mean"], mode="lines+markers",
                                       name="Training R²", line=dict(color=SKY)))
                f.add_trace(go.Scatter(x=lc["train_rows_per_fold"], y=lc["validation_R2_mean"], mode="lines+markers",
                                       name="Validation R²", line=dict(color=BLUE)))
                f.update_layout(xaxis_title="Training sessions per fold", yaxis_title="R²")
                show_fig(f, "Training and validation R squared against training set size")

        ep = read_csv(EPOCH_DIR / "epoch_history.csv")
        ep_sum = read_json(EPOCH_DIR / "epoch_summary.json")
        st.markdown("##### Training curves: loss and accuracy per epoch")
        if ep is None or ep_sum is None:
            st.info("Epoch history not generated yet. Run `python -m ml.epoch_training_curves` in the backend folder.")
        else:
            ref_split = ep_sum["reference_same_split"]
            e1, e2 = st.columns(2)
            with e1:
                f = go.Figure()
                f.add_trace(go.Scatter(x=ep["epoch"], y=ep["loss"], name="loss", line=dict(color=SKY)))
                f.add_trace(go.Scatter(x=ep["epoch"], y=ep["val_loss"], name="val_loss", line=dict(color=BLUE)))
                f.add_hline(y=ref_split["elasticnet_closed_form"]["val_loss"], line_dash="dot", line_color=PALE,
                            annotation_text="ElasticNet (deployed recipe) val_loss")
                f.update_layout(xaxis_title="Epoch", yaxis_title="Loss (MSE, log scale)", yaxis_type="log")
                show_fig(f, "Training and validation loss per epoch")
            with e2:
                f = go.Figure()
                f.add_trace(go.Scatter(x=ep["epoch"], y=ep["accuracy"], name="accuracy", line=dict(color=SKY)))
                f.add_trace(go.Scatter(x=ep["epoch"], y=ep["val_accuracy"], name="val_accuracy", line=dict(color=BLUE)))
                f.add_hline(y=ref_split["dummy_mean"]["val_accuracy"], line_dash="dash", line_color="#E8A0A0",
                            annotation_text="Mean guess val_accuracy")
                f.update_layout(xaxis_title="Epoch", yaxis_title="% within ±1 RPE", yaxis_range=[0, 100])
                show_fig(f, "Training and validation accuracy (within plus or minus 1 RPE) per epoch")
            last = ep.iloc[-1]
            st.caption(
                f"ElasticNet itself has no epochs, so these curves come from an iterative twin: same features, same "
                f"preprocessing and same elastic-net penalty, trained by SGD one pass (epoch) at a time on "
                f"{ep_sum['participants']['train']} training athletes and checked on {ep_sum['participants']['validation']} "
                f"unseen validation athletes. After {int(last.epoch)} epochs: val_loss {last.val_loss:.3f}, "
                f"val_accuracy {last.val_accuracy:.1f}% (mean guess {ref_split['dummy_mean']['val_accuracy']:.1f}%). "
                "The deployed model was not retrained; sealed test athletes were not used. Synthetic data only.")
        if fi is not None:
            st.markdown("##### What the model relies on (permutation importance)")
            fs = fi.sort_values("importance_mean_R2_drop")
            f = go.Figure(go.Bar(x=fs["importance_mean_R2_drop"], y=fs["feature"], orientation="h", marker_color=BLUE,
                                 error_x=dict(type="data", array=fs["importance_std"])))
            f.update_layout(xaxis_title="R² lost when the feature is shuffled")
            show_fig(f, "Permutation feature importance of the RPE model", height=300)
            st.caption("Planned intensity dominates because the simulation was built that way. "
                       "Temperature, humidity and air quality contribute little.")

    # ------------------------------------------------------------------ Real lifters
    elif view == "Real lifters" and eval_summary and cmp_df is not None:
        cohorts = eval_summary["cohorts"]
        pm_c, forms_c, comb_c = cohorts["pmdata_development"], cohorts["google_forms_testing"], cohorts["combined_scope"]
        dummy_row = cmp_df[cmp_df.model == "dummy_mean"].iloc[0]
        models_only = cmp_df[cmp_df.model != "dummy_mean"]
        best = models_only.loc[models_only.pmdata_lopo_within_1_pct.idxmax()]
        best_ext = models_only.loc[models_only.external_test_within_1_pct.idxmax()]

        st.subheader("Real lifters: PMData and the FullStrength survey")
        st.markdown(TAGS["real"])
        st.caption(f"{comb_c['total_real_sessions']} real strength sessions from {comb_c['total_real_participants']} athletes "
                   f"({pm_c['sessions']} PMData sessions from {pm_c['participants']} athletes, and {forms_c['sessions']} survey "
                   f"sessions from {forms_c['participants']} athletes). Only information known before the session is used.")
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            kpi("Real athletes", f"{comb_c['total_real_participants']}",
                f"{pm_c['participants']} PMData + {forms_c['participants']} survey", "real")
        with k2:
            kpi("Real sessions", f"{comb_c['total_real_sessions']}", f"{pm_c['sessions']} + {forms_c['sessions']}", "real")
        with k3:
            kpi("PMData, best model", f"{best.pmdata_lopo_within_1_pct:.1f}%",
                f"within ±1 RPE ({best.pmdata_lopo_within_1_count}); mean guess {dummy_row.pmdata_lopo_within_1_pct:.1f}% "
                f"({dummy_row.pmdata_lopo_within_1_count})", "real")
        with k4:
            kpi("Survey test, best model", f"{best_ext.external_test_within_1_pct:.1f}%",
                f"within ±1 RPE ({best_ext.external_test_within_1_count}); mean guess "
                f"{dummy_row.external_test_within_1_pct:.1f}% ({dummy_row.external_test_within_1_count})", "real")
        st.info("With wellness answers alone, no model beats the mean guess on real lifters, and R² is below zero. "
                "The survey cohort has only 13 sessions, so its result is a sanity check, not a measurement. "
                "Neither real dataset recorded temperature, humidity or air quality, so the full nine-input model "
                "could not be tested on real athletes.", title="What this means", icon=":material/info:")

        def model_table(columns):
            return pd.DataFrame([{"Model": NICE_MODEL[r.model], **{label: fmt(r) for label, fmt in columns.items()}}
                                 for _, r in cmp_df.iterrows()])

        t_pm, t_ext, t_forms, t_dist = st.tabs(["PMData (leave one athlete out)", "Survey test set",
                                                "Survey (internal)", "Cohorts"])
        with t_pm:
            st.caption("Each of the 10 athletes is predicted by a model trained on the other nine.")
            st.dataframe(model_table({
                "R²": lambda r: f"{r.pmdata_lopo_R2:.3f}", "MAE": lambda r: f"{r.pmdata_lopo_MAE:.3f}",
                "Within ±1 RPE": lambda r: f"{r.pmdata_lopo_within_1_pct:.1f}% ({r.pmdata_lopo_within_1_count})"}),
                hide_index=True, width="stretch", alt="PMData leave-one-athlete-out results")
        with t_ext:
            st.caption("Models trained on all 87 PMData sessions, tested on 13 survey sessions from 13 unseen athletes.")
            st.dataframe(model_table({
                "R²": lambda r: f"{r.external_test_R2:.3f}", "MAE": lambda r: f"{r.external_test_MAE:.3f}",
                "RMSE": lambda r: f"{r.external_test_RMSE:.3f}",
                "Within ±1 RPE": lambda r: f"{r.external_test_within_1_pct:.1f}% ({r.external_test_within_1_count})"}),
                hide_index=True, width="stretch", alt="Survey external test results")
        with t_forms:
            st.caption("Leave-one-athlete-out inside the 13-session survey cohort. Exploratory only.")
            st.dataframe(model_table({
                "R²": lambda r: f"{r.forms_lopo_R2:.3f}", "MAE": lambda r: f"{r.forms_lopo_MAE:.3f}",
                "Within ±1 RPE": lambda r: f"{r.forms_lopo_within_1_pct:.1f}%"}),
                hide_index=True, width="stretch", alt="Survey internal leave-one-athlete-out results")
        with t_dist:
            st.dataframe(pd.DataFrame([
                {"Cohort": "PMData strength subset", "Sessions": pm_c["sessions"], "Athletes": pm_c["participants"],
                 "RPE mean ± SD": f"{pm_c['target_mean']:.2f} ± {pm_c['target_std']:.2f}",
                 "Known before the session": "Sleep duration and quality, readiness, mood, fatigue, soreness, stress",
                 "Environment sensors": "None"},
                {"Cohort": "FullStrength survey (Sri Lanka)", "Sessions": forms_c["sessions"], "Athletes": forms_c["participants"],
                 "RPE mean ± SD": f"{forms_c['target_mean']:.2f} ± {forms_c['target_std']:.2f}",
                 "Known before the session": "Sleep, readiness, fatigue, soreness, stress, workout type, planned duration and intensity",
                 "Environment sensors": "Gym type only; no temperature, humidity or AQI"},
                {"Cohort": "Calibrated benchmark", "Sessions": 5000, "Athletes": 250, "RPE mean ± SD": "6.01 ± 1.00",
                 "Known before the session": "All nine inputs (calibrated)", "Environment sensors": "Calibrated continuous"},
            ]), hide_index=True, width="stretch", alt="Cohort comparison")

        preds = read_csv(REAL_EVAL / "real_athlete_predictions.csv")
        if preds is not None:
            st.markdown("##### Actual vs predicted RPE, survey test set (13 sessions)")
            f = px.scatter(preds, x="actual_rpe", y="elasticnet_pred", color_discrete_sequence=[BLUE], opacity=0.8,
                           hover_data=["participant_id", "workout_type"],
                           labels={"actual_rpe": "Actual RPE", "elasticnet_pred": "Predicted RPE (ElasticNet)"})
            f.add_trace(go.Scatter(x=[2, 10.5], y=[2, 10.5], mode="lines", line=dict(color=PALE, dash="dash")))
            f.update_layout(showlegend=False)
            show_fig(f, "Actual versus predicted RPE on the survey test set")
            st.caption("The model predicts about the same value for every session, so it underestimates the hardest ones.")

        st.markdown("##### Why is R² below zero?")
        st.markdown(
            "- Wellness answers (sleep, mood, soreness) say little about how hard a particular workout felt; the training "
            "load itself (sets, reps, % of 1RM) matters far more.\n"
            "- The cohorts differ: PMData sessions average RPE 5.6, the survey sessions 6.5, so a model fitted on one "
            "is biased on the other.\n"
            "- Small samples (87 and 13 sessions) make every estimate noisy.")

    # ------------------------------------------------------------------ Open data
    elif view == "Open data" and warm:
        rob, rem = warm["Robinson2025_OSF_58hy3"], warm["Remmert2026_OSF_5wprn"]
        hyb, wo, dm = "ElasticNet wellness + personal history", "ElasticNet wellness only", "Dummy (training mean)"
        st.subheader("Open data replication")
        st.markdown(TAGS["open"])
        st.caption("Two public resistance-training datasets (Robinson 2025, Remmert 2026), downloaded and checked column by "
                   "column. Each is analysed on its own, never pooled. The athlete being predicted is never in the training data.")
        o1, o2, o3, o4 = st.columns(4)
        with o1:
            kpi("Athletes", "65", "31 trained men + 34 trained adults", "open")
        with o2:
            kpi("Sessions", "1,380", "700 whole-session + 680 per-arm ratings", "open")
        with o3:
            kpi("Wellness only", f"{rob[wo]['within1']:.0f}% / {rem[wo]['within1']:.0f}%",
                f"within ±1 RPE; mean guess {rob[dm]['within1']:.0f}% / {rem[dm]['within1']:.0f}%", "open")
        with o4:
            kpi("Wellness + own history", f"{rob[hyb]['within1']:.0f}% / {rem[hyb]['within1']:.0f}%",
                f"within ±1 RPE · MAE {rob[hyb]['MAE']:.2f} / {rem[hyb]['MAE']:.2f} · exploratory", "open")
        st.info("Wellness scores alone predict a new lifter's RPE no better than the average. Once the model also knows "
                "that lifter's own earlier ratings (from 5 or more logged sessions, all before the one predicted), "
                "agreement roughly doubles in both cohorts. Personalisation is where the signal is.",
                title="Key finding", icon=":material/insights:")

        order = [dm, wo, "Personal running mean", hyb]
        nice = {dm: "Mean guess", wo: "Wellness only", "Personal running mean": "Own running mean", hyb: "Wellness + own history"}
        st.markdown("##### Within ±1 RPE on unseen athletes (95% interval)")
        fig = go.Figure()
        for name, res, color, off in [("Robinson 2025 (31 athletes)", rob, BLUE, -0.12),
                                      ("Remmert 2026 (34 athletes)", rem, SKY, 0.12)]:
            xs = [res[m]["within1"] for m in order]
            lo = [res[m]["within1"] - res[m]["within1_95CI"][0] for m in order]
            hi = [res[m]["within1_95CI"][1] - res[m]["within1"] for m in order]
            fig.add_trace(go.Scatter(
                x=xs, y=[i + off for i in range(len(order))], mode="markers", name=name,
                marker=dict(size=12, color=color),
                error_x=dict(type="data", symmetric=False, array=hi, arrayminus=lo, color=color, thickness=2, width=5),
                customdata=[[nice[m], res[m]["MAE"], res[m]["n"]] for m in order],
                hovertemplate="%{customdata[0]}<br>within ±1: %{x:.1f}%<br>MAE %{customdata[1]:.2f} · n = %{customdata[2]}<extra></extra>"))
        fig.update_layout(xaxis=dict(title="Sessions predicted within ±1 RPE (%)", range=[20, 75], ticksuffix="%"),
                          yaxis=dict(tickvals=list(range(len(order))), ticktext=[nice[m] for m in order]))
        show_fig(fig, "Within plus or minus one RPE agreement with intervals for four models on two open datasets")

        tbl = [{"Dataset": ds, "Model": nice[m], "Sessions": res[m]["n"], "MAE": res[m]["MAE"], "R²": res[m]["R2"],
                "Within ±1": f"{res[m]['within1']:.1f}%",
                "95% interval": f"{res[m]['within1_95CI'][0]:.1f} to {res[m]['within1_95CI'][1]:.1f}%"}
               for ds, res in [("Robinson 2025", rob), ("Remmert 2026", rem)] for m in order]
        st.dataframe(pd.DataFrame(tbl), hide_index=True, width="stretch",
                     alt="Leave-one-athlete-out results for four models on two open datasets")
        st.caption("All models are compared on the same rows: each athlete's 6th and later sessions. One fixed configuration, "
                   "no tuning. Intervals come from resampling athletes, not sessions.")
    elif view in ("Real lifters", "Open data"):
        st.info("Result files not found. Run the experiment scripts in ml/ to generate them.")

# =========================================================================================
# 3. METHODS AND STATUS
# =========================================================================================
with tab_method:
    st.subheader("What is built and what is validated")
    st.dataframe(pd.DataFrame([
        {"Part": "RPE model, baseline", "Type": "Ridge regression", "Trained on": "Calibrated data",
         "Status": "Built, served by the API by default", "Evidence": "Benchmark validated (R² 0.54, within ±1 85.8%)"},
        {"Part": "RPE model, PP1 optimised", "Type": "ElasticNet + engineered features", "Trained on": "Calibrated data",
         "Status": "Built, selectable (RPE_MODEL_VARIANT), used by this dashboard",
         "Evidence": "Benchmark validated (R² 0.557, within ±1 86.1%)"},
        {"Part": "Session completion classifier", "Type": "Random forest", "Trained on": "Calibrated data",
         "Status": "Trained, not exposed in the API or apps", "Evidence": "Benchmark validated"},
        {"Part": "Real-lifter wellness models", "Type": "Ridge, ElasticNet, random forest", "Trained on": "PMData (real)",
         "Status": "Experiment only, not deployed", "Evidence": "Do not beat the mean guess"},
        {"Part": "Personalised model (own history)", "Type": "ElasticNet + athlete history", "Trained on": "Open data (real)",
         "Status": "Exploratory result, not yet in the product", "Evidence": "About 64% within ±1 on 65 athletes"},
        {"Part": "TEQS score", "Type": "Rule-based, equal weights", "Trained on": "None",
         "Status": "Built, experimental", "Evidence": "Not validated"},
        {"Part": "Nutrition timing engine", "Type": "Rule-based (10 rules)", "Trained on": "None",
         "Status": "Built, educational", "Evidence": "Not validated"},
        {"Part": "OpenWeather integration", "Type": "Live API through the backend", "Trained on": "None",
         "Status": "Built; temperature and humidity used, air quality shown only", "Evidence": "Outdoor estimate"},
    ]), hide_index=True, width="stretch", alt="Status of each component")

    st.subheader("Safeguards")
    st.markdown(
        "- **Air-quality scale.** OpenWeather reports a 1–5 index; TEQS expects US EPA AQI (0–500). Passing 1–5 in would score "
        "polluted air as perfect, so the 1–5 value is displayed but never copied into TEQS.\n"
        "- **Outdoor estimate.** OpenWeather values describe the air outside, not inside a gym.\n"
        "- **TEQS is experimental.** It is an equal-weighted average of six sub-scores with placeholder thresholds. "
        "It has not been validated against real outcomes.\n"
        "- **Nutrition guidance is educational.** It gives no fluid volumes, macronutrient grams or supplement advice, "
        "and is not medical advice.\n"
        "- **Benchmark and real cohort outcomes are kept apart.** Every number is tagged with its data source.")

    st.subheader("Evaluation protocol")
    st.markdown(
        "- All splits are grouped by athlete, so the same person never appears in both training and test data. "
        "Random splits leak personal baselines and inflate R².\n"
        "- Calibrated data: 5-fold grouped CV on 150 athletes for tuning, then one sealed test on 50 unseen athletes.\n"
        "- Real data: leave-one-athlete-out.\n"
        "- Every model is compared with a mean-guess baseline.")

    with st.expander("Datasets considered"):
        st.dataframe(pd.DataFrame([
            {"Dataset": "Robinson et al. 2025 (OSF 58hy3)", "Athletes": "31", "Sessions": "700", "Target": "CR-10 whole-session RPE",
             "Verdict": "Partially usable", "Notes": "Perceived recovery and motivation before the session; no sleep or environment"},
            {"Dataset": "Remmert et al. 2026 (OSF 5wprn)", "Athletes": "34", "Sessions": "680", "Target": "CR-10 per arm",
             "Verdict": "Partially usable", "Notes": "Sleep hours and quality, recovery, stress; no environment"},
            {"Dataset": "PMData strength subset", "Athletes": "10", "Sessions": "87", "Target": "0–10 whole-session RPE",
             "Verdict": "Partially usable", "Notes": "Seven morning wellness items; CC BY 4.0"},
            {"Dataset": "FullStrength survey cohort", "Athletes": "13", "Sessions": "13", "Target": "Foster 0–10",
             "Verdict": "Partially usable", "Notes": "Own data; includes workout type, planned duration and intensity"},
            {"Dataset": "Hobein et al. 2026 (OSF mej4d)", "Athletes": "36", "Sessions": "144", "Target": "Mean of set RPEs",
             "Verdict": "Supporting only", "Notes": "Scale unverified"},
            {"Dataset": "Nijland (DataverseNL TMRZCQ)", "Athletes": "22", "Sessions": "74", "Target": "Borg 6–20",
             "Verdict": "Supporting only", "Notes": "Different scale"},
            {"Dataset": "Rugby dataset (Hu et al. 2024)", "Athletes": "26", "Sessions": "520", "Target": "Foster 0–10",
             "Verdict": "Secondary benchmark", "Notes": "All sessions 45 min; no sleep or environment"},
            {"Dataset": "PERSIST (Zenodo 7437230)", "Athletes": "12", "Sessions": "—", "Target": "Per set only",
             "Verdict": "Restricted", "Notes": "Available on request"},
            {"Dataset": "Boxman-Zeevi et al. 2022", "Athletes": "57", "Sessions": "—", "Target": "Not recorded",
             "Verdict": "Not suitable", "Notes": "Session RPE was not measured"},
            {"Dataset": "SWIFTIES, strength/plyometric study, SoccerMon", "Athletes": "—", "Sessions": "—", "Target": "Various",
             "Verdict": "Excluded", "Notes": "Lab endurance trials, weekly RPE only, or field running"},
        ]), hide_index=True, width="stretch", alt="Datasets considered for the real-athlete evaluation")

    with st.expander("Which of the nine RPE-model inputs each dataset has"):
        Y, P, N = "yes", "proxy", "no"
        st.dataframe(pd.DataFrame({
            "Input": ["Workout type", "Planned intensity", "Planned duration", "Sleep hours", "Sleep quality",
                      "Readiness / mood", "Temperature", "Humidity", "Air quality"],
            "Robinson": [P, P, N, N, N, Y, N, N, N], "Remmert": [P, P, N, Y, Y, Y, N, N, N],
            "PMData": [P, N, N, Y, Y, Y, N, N, N], "Survey": [Y, Y, Y, Y, Y, Y, N, N, N],
        }), hide_index=True, width="stretch", alt="Input availability by dataset")
        st.caption("No public dataset has temperature, humidity or air quality recorded during a strength session, "
                   "and none is invented here.")

st.caption("Educational research prototype. TEQS and the nutrition guidance are experimental and not clinically validated.")
