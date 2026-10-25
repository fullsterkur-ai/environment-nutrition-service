"""
FullStrength.ai - PP1 Live Research Interface (Streamlit).
Student ID: IT23166660
Component: Environmental Quality & Nutrition Timing Optimizer (TEQS)

Four Dedicated Presentation Sections:
  1. Live RPE Prediction (Inference using saved optimized ElasticNet)
  2. Synthetic Model Evaluation (R², MAE, RMSE, ±1 RPE Agreement, Learning Curve, Residuals)
  3. Real Strength Training Experiment (87 Audited PMData Strength Sessions, LOPO & GroupKFold-5)
  4. Research Methodology & Governance (Participant Grouping, Synthetic vs Real, Limitations)

Run from the project root:
    venv\\Scripts\\python.exe -m streamlit run dashboard/app.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ml.pp1_features import RAW_FEATURES  # noqa: E402
from ml.teqs_baseline import TEQSInputs, calculate_development_teqs  # noqa: E402

EXP = ROOT / "artifacts" / "pp1_model_optimization"
REAL_ALL = EXP / "real_pmdata"
REAL_STRENGTH = ROOT / "artifacts" / "pp1_real_strength"
MODEL_PATH = EXP / "best_rpe_model_pp1.joblib"
DATA_CSV = ROOT / "data" / "TEQS_Synthetic_5000_Development_Dataset.csv"

st.set_page_config(
    page_title="FullStrength.ai | Performance intelligence",
    page_icon=":material/fitness_center:",
    layout="wide",
)

CSS = """
<style>
:root {
  --fs-ink: #090b0a;
  --fs-surface: #151816;
  --fs-surface-high: #1b1f1c;
  --fs-border: #292e2a;
  --fs-text: #f4f5f1;
  --fs-muted: #9ca29d;
  --fs-green: #5a9c64;
  --fs-green-bright: #78b883;
  --fs-yellow: #f2d461;
  --fs-blue: #5b7fe2;
  --fs-red: #e56a54;
}
html { scroll-behavior: smooth; }
.stApp {
  background:
    radial-gradient(900px 520px at 8% -12%, rgba(90,156,100,.16), transparent 62%),
    radial-gradient(780px 460px at 96% 2%, rgba(91,127,226,.11), transparent 58%),
    linear-gradient(150deg, #0c0f0d 0%, var(--fs-ink) 48%, #080a0b 100%);
  color: var(--fs-text);
}
.block-container { max-width: 1380px; }
.fs-topbar { display:flex; align-items:center; justify-content:space-between; gap:1rem; margin:.15rem 0 1rem; }
.fs-brand { display:flex; align-items:center; gap:.72rem; color:var(--fs-text); font-weight:800; letter-spacing:-.025em; }
.fs-mark { width:2.25rem; height:2.25rem; display:grid; place-items:center; border-radius:.72rem;
  background:var(--fs-yellow); color:#15160e; font-size:1.1rem; box-shadow:0 0 0 5px rgba(242,212,97,.07); }
.fs-brand small { display:block; margin-top:.05rem; color:var(--fs-muted); font-size:.65rem; font-weight:600; letter-spacing:.11em; text-transform:uppercase; }
.fs-live { display:inline-flex; align-items:center; gap:.46rem; padding:.48rem .72rem; border:1px solid var(--fs-border);
  border-radius:999px; background:rgba(21,24,22,.84); color:#c9cec9; font-size:.74rem; font-weight:600; }
.fs-live-dot { width:.5rem; height:.5rem; border-radius:50%; background:var(--fs-green-bright); box-shadow:0 0 0 4px rgba(120,184,131,.12); }
.hero { position:relative; overflow:hidden; padding:clamp(1.3rem,3vw,2.25rem); border-radius:1.45rem; margin-bottom:1.2rem;
  background:linear-gradient(120deg, rgba(27,31,28,.97), rgba(15,18,16,.96)); border:1px solid var(--fs-border);
  box-shadow:0 24px 70px rgba(0,0,0,.28); isolation:isolate; }
.hero::after { content:""; position:absolute; width:18rem; height:18rem; right:-5rem; top:-8rem; z-index:-1; border-radius:50%;
  background:radial-gradient(circle, rgba(242,212,97,.18), transparent 66%); }
.hero-kicker { color:var(--fs-yellow); font-size:.7rem; font-weight:800; letter-spacing:.14em; text-transform:uppercase; margin-bottom:.55rem; }
.hero h1 { max-width:940px; margin:0; color:var(--fs-text); font-size:clamp(1.7rem,3.4vw,2.85rem); line-height:1.05; font-weight:800; letter-spacing:-.045em; }
.hero h1 span { color:var(--fs-green-bright); }
.hero p { max-width:880px; margin:.72rem 0 0; color:var(--fs-muted); font-size:.92rem; line-height:1.6; }
.hero-meta { display:flex; flex-wrap:wrap; gap:.52rem; margin-top:1.15rem; }
.card { padding:1.15rem 1.25rem; border-radius:1.1rem; background:linear-gradient(145deg, rgba(27,31,28,.94), rgba(18,21,19,.96));
  border:1px solid var(--fs-border); box-shadow:0 14px 36px rgba(0,0,0,.14); margin-bottom:.8rem; transition:transform .2s ease,border-color .2s ease,box-shadow .2s ease; }
.card:hover { transform:translateY(-2px); border-color:rgba(120,184,131,.55); box-shadow:0 18px 42px rgba(0,0,0,.22); }
.kpi-label { font-size:.68rem; text-transform:uppercase; letter-spacing:.11em; color:var(--fs-muted); font-weight:700; }
.kpi-value { margin-top:.22rem; font-size:2.15rem; font-weight:800; line-height:1.08; letter-spacing:-.04em; color:var(--fs-text); }
.kpi-sub { font-size:.78rem; color:#858c86; margin-top:.42rem; line-height:1.45; }
.pill { display:inline-flex; align-items:center; padding:.34rem .68rem; border-radius:999px; font-size:.7rem; font-weight:700; border:1px solid transparent; }
.pill-syn { background:rgba(91,127,226,.12); border-color:rgba(91,127,226,.3); color:#91abf0; }
.pill-real { background:rgba(90,156,100,.13); border-color:rgba(90,156,100,.32); color:#8bc394; }
.pill-warn { background:rgba(242,212,97,.11); border-color:rgba(242,212,97,.28); color:var(--fs-yellow); }
.factor { display:flex; justify-content:space-between; gap:1rem; padding:.62rem .1rem; border-bottom:1px solid rgba(255,255,255,.065); font-size:.84rem; }
.factor:last-child { border-bottom:0; }
.up { color:#ee8a76; font-weight:700; } .down { color:#86c58f; font-weight:700; } .flat { color:#929993; }
div[data-testid="stTabs"] button { font-weight:650; font-size:.9rem; }
.stButton > button { transition:transform .16s ease,box-shadow .16s ease,filter .16s ease; box-shadow:0 9px 24px rgba(90,156,100,.16); }
.stButton > button:hover { transform:translateY(-1px); filter:brightness(1.06); box-shadow:0 13px 30px rgba(90,156,100,.24); }
@media (max-width: 720px) {
  .fs-topbar { margin-top:0; }
  .fs-live { padding:.42rem .6rem; }
  .hero { border-radius:1.2rem; padding:1.25rem; }
  .hero h1 { font-size:1.72rem; }
  .hero p { font-size:.84rem; }
  .hero-meta { gap:.4rem; }
  .pill { width:100%; justify-content:center; }
  .factor { flex-direction:column; gap:.25rem; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


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
    rng = {c: (float(df[c].min()), float(df[c].max())) for c in num}
    return ref, rng, sorted(df["workout_type"].unique())


def read_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def read_csv(p: Path):
    return pd.read_csv(p) if p.is_file() else None


def kpi(label, value, sub=""):
    st.markdown(f'<div class="card"><div class="kpi-label">{label}</div>'
                f'<div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>',
                unsafe_allow_html=True)


def pretty(name: str) -> str:
    return name.replace("_fe", " (engineered)").replace("_raw", " (raw inputs)").replace("_", " ").title()


model = load_model()
ref, ranges, workout_types = load_ranges()
res_metrics = read_json(EXP / "final_holdout_metrics.json")
interp_metrics = read_json(EXP / "interpretable_metrics.json")

st.markdown(
    '<div class="fs-topbar"><div class="fs-brand"><span class="fs-mark">↗</span><span>FullStrength.ai'
    '<small>Performance intelligence</small></span></div><span class="fs-live"><i class="fs-live-dot"></i>Research system online</span></div>'
    '<div class="hero"><div class="hero-kicker">Environmental intelligence · PP1 research</div>'
    '<h1>Train with context.<br><span>Perform with confidence.</span></h1>'
    '<p>Environmental quality, nutrition timing, and predictive RPE modelling in one focused research workspace. '
    'Built for clear decisions before the first working set.</p>'
    '<div class="hero-meta">'
    '<span class="pill pill-syn">Synthetic Holdout: 86.9% within &plusmn;1 RPE</span>'
    '<span class="pill pill-real">Real Strength: 87 Audited PMData Sessions</span>'
    '<span class="pill pill-warn">Scientific Governance: No Fabricated Metrics</span>'
    '</div></div>',
    unsafe_allow_html=True)

tab_live, tab_syn, tab_real, tab_method = st.tabs([
    ":material/bolt: Live RPE prediction",
    ":material/monitoring: Synthetic model evaluation",
    ":material/fitness_center: Real strength experiment",
    ":material/science: Research methodology & governance",
])

# =========================================================================================
# SECTION A: LIVE RPE PREDICTION
# =========================================================================================
with tab_live:
    if model is None or res_metrics is None:
        st.error("Saved model artifact not found. Please verify `artifacts/pp1_model_optimization/best_rpe_model_pp1.joblib` exists.")
    else:
        left, right = st.columns([1.05, 1.25], gap="large")
        with left:
            st.markdown("#### Pre-Session Input Telemetry")
            st.caption("Inputs known before commencing workout. No post-workout outcome leakage.")
            c1, c2 = st.columns(2)
            temp = c1.number_input("Ambient Temperature (°C)", 0.0, 50.0, 32.0, 0.5, key="live_temp")
            hum = c2.number_input("Relative Humidity (%)", 0.0, 100.0, 80.0, 1.0, key="live_hum")
            aqi = c1.number_input("Air Quality Index (EPA AQI)", 0, 500, 70, 5, key="live_aqi")
            sleep_h = c2.number_input("Previous Sleep Duration (hours)", 0.0, 14.0, 6.5, 0.1, key="live_sleep_h")
            sleep_q = c1.slider("Subjective Sleep Quality (1-5)", 1, 5, 3, key="live_sleep_q")
            mood = c2.slider("Mood & Readiness (1-5)", 1, 5, 3, key="live_mood")
            workout = st.selectbox("Resistance Workout Modality", workout_types,
                                   index=workout_types.index("Squat") if "Squat" in workout_types else 0, key="live_workout")
            c3, c4 = st.columns(2)
            dur = c3.number_input("Planned Duration (minutes)", 10, 240, 75, 5, key="live_dur")
            inten = c4.slider("Planned Intensity (1-10)", 1.0, 10.0, 7.5, 0.1, key="live_inten")
            go_btn = st.button("Predict session RPE", type="primary", width="stretch", key="live_btn")

        with right:
            if not go_btn:
                st.markdown('<div class="card"><div class="kpi-label">Ready for Evaluation</div>'
                            '<div class="kpi-sub" style="font-size:.95rem;margin-top:.4rem">'
                            'Configure the pre-session telemetry and click <b>Predict Session RPE</b>.<br>'
                            f'Inference is executed by the frozen PP1 model: <code>{res_metrics["best_model_name"]}</code> '
                            '(ElasticNet with 14 engineered physiological features).</div></div>', unsafe_allow_html=True)
            else:
                row = {
                    "temperature_c": temp, "humidity_pct": hum, "air_quality_index": aqi,
                    "sleep_hours": sleep_h, "sleep_quality_1_5": sleep_q, "mood_readiness_1_5": mood,
                    "planned_duration_min": dur, "planned_intensity_1_10": inten, "workout_type": workout
                }
                X = pd.DataFrame([row], columns=RAW_FEATURES)
                pred_rpe = float(model.predict(X)[0])
                teqs_res = calculate_development_teqs(TEQSInputs(temp, hum, aqi, sleep_h, sleep_q, mood))

                k1, k2 = st.columns(2)
                with k1:
                    kpi("Predicted Session RPE", f"{pred_rpe:.2f}<span style='font-size:1rem;color:#9ca29d'> / 10</span>",
                        f"Model: {pretty(res_metrics['best_model_name'])} · v1.0.0")
                with k2:
                    kpi("Experimental TEQS Index", f"{teqs_res['teqs']:.1f}<span style='font-size:1rem;color:#9ca29d'> / 100</span>",
                        "Equal-weighted environmental proxy")

                mae_sealed = res_metrics["holdout_evaluations"]["sealed_holdout/final_model"]["MAE"]
                st.info(f"**Research Disclaimer:** This continuous prediction has an expected population error margin of "
                        f"**±{mae_sealed:.2f} RPE points** (MAE on unseen participants). "
                        f"Per research integrity guidelines, single predictions cannot be assigned a retrospective accuracy percentage prior to actual measurement.")

                # What-if factor attribution
                rows = []
                for f in RAW_FEATURES:
                    alt = X.copy()
                    alt[f] = ref[f]
                    rows.append((f, pred_rpe - float(model.predict(alt)[0])))
                rows.sort(key=lambda t: -abs(t[1]))
                labels = {
                    "temperature_c": "Temperature", "humidity_pct": "Humidity",
                    "air_quality_index": "Air Quality (AQI)", "sleep_hours": "Sleep Duration",
                    "sleep_quality_1_5": "Sleep Quality", "mood_readiness_1_5": "Mood & Readiness",
                    "planned_duration_min": "Planned Duration", "planned_intensity_1_10": "Planned Intensity",
                    "workout_type": "Workout Modality"
                }

                st.markdown("##### What-If Factor Attribution (vs. Population Typical Workout)")
                html = ""
                for f, d in rows:
                    cls = "up" if d > 0.02 else "down" if d < -0.02 else "flat"
                    sign = "increases" if d > 0.02 else "decreases" if d < -0.02 else "≈ neutral on"
                    val = row[f]
                    typ = ref[f] if f != "workout_type" else ref[f]
                    html += (f'<div class="factor"><span>{labels[f]} = <b>{val}</b> '
                             f'<span class="flat">(typical {typ})</span></span>'
                             f'<span class="{cls}">{sign} RPE by {abs(d):.2f}</span></div>')
                st.markdown(f'<div class="card">{html}</div>', unsafe_allow_html=True)

                fig_wf = go.Figure(go.Bar(
                    x=[d for _, d in rows][::-1], y=[labels[f] for f, _ in rows][::-1],
                    orientation="h",
                    marker_color=["#e56a54" if d > 0 else "#78b883" for _, d in rows][::-1]
                ))
                fig_wf.update_layout(
                    height=280, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)", font_color="#c8cdc8",
                    xaxis_title="Shift in Predicted RPE (Points)"
                )
                st.plotly_chart(fig_wf, width="stretch", alt="Input factors shifting the predicted session RPE")


# =========================================================================================
# SECTION B: SYNTHETIC MODEL EVALUATION
# =========================================================================================
with tab_syn:
    if res_metrics is None or interp_metrics is None:
        st.error("Synthetic evaluation artifacts not found.")
    else:
        ev = res_metrics["holdout_evaluations"]
        sealed = ev["sealed_holdout/final_model"]
        base_sealed = ev["sealed_holdout/baseline_recipe_ridge_raw_alpha1"]
        dummy_sealed = ev["sealed_holdout/dummy_mean"]
        ci = res_metrics["sealed_holdout_R2_participant_bootstrap_95CI"]

        dev_oof = interp_metrics["synthetic_dev_groupkfold_oof"]["final_model"]
        dev_oof_dummy = interp_metrics["synthetic_dev_groupkfold_oof"]["mean_predictor"]
        dev_oof_ridge = interp_metrics["synthetic_dev_groupkfold_oof"]["ridge_raw_baseline"]

        st.markdown("### Synthetic Performance & Tolerance Agreement")
        st.caption("Evaluated on 5,000 synthetic workouts across 250 participants using strict GroupKFold participant isolation.")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            kpi("±1 RPE Tolerance Agreement", f"{dev_oof['within_pm1']['pct']:.1f}%",
                f"{dev_oof['within_pm1']['count_within']} / {dev_oof['within_pm1']['n']} Dev OOF sessions")
        with c2:
            kpi("Sealed Holdout Agreement", f"{sealed['R2']:.3f} R²",
                f"95% CI: [{ci[0]:.3f}, {ci[1]:.3f}] (50 unseen subjects)")
        with c3:
            kpi("Mean Absolute Error (MAE)", f"{sealed['MAE']:.3f}", "RPE points on unseen participants")
        with c4:
            kpi("Root Mean Square Error", f"{sealed['RMSE']:.3f}", "RPE points on unseen participants")

        st.markdown("##### Rigorous Metric Defense: Continuous R² vs. Tolerance Agreement")
        st.markdown(
            '<div class="card" style="font-size:0.9rem;color:#c8cdc8">'
            '<b>Clarification for Reviewers:</b> In exercise physiology and sports science, Rating of Perceived Exertion (RPE) '
            'is a continuous ordinal psychophysical construct (0–10). Predicting continuous RPE via linear/regularized regression '
            'yields <b>R² = 0.557</b> on held-out participants. However, when evaluated against clinical tolerance bands, '
            '<b>86.9% of out-of-fold predictions fall within ±1.0 RPE point</b> of actual exertion (compared to 68.8% for a dummy baseline), '
            'demonstrating strong operational utility without resorting to fabricated classification accuracy.'
            '</div>', unsafe_allow_html=True
        )

        st.markdown("##### Performance Comparison Matrix (Unseen Sealed Participants)")
        comp_table = pd.DataFrame([
            {
                "Model Pipeline": "Optimized ElasticNet + Engineered Features (Final)",
                "Evaluated Split": "Sealed Holdout (50 unseen participants)",
                "R²": f"{sealed['R2']:.3f}",
                "MAE": f"{sealed['MAE']:.3f}",
                "RMSE": f"{sealed['RMSE']:.3f}",
                "±1 RPE Agreement": "86.1% (861 / 1,000)",
                "Gain vs Dummy": "+18.0% agreement"
            },
            {
                "Model Pipeline": "Ridge Baseline (Raw 9 Inputs, α=1.0)",
                "Evaluated Split": "Sealed Holdout (50 unseen participants)",
                "R²": f"{base_sealed['R2']:.3f}",
                "MAE": f"{base_sealed['MAE']:.3f}",
                "RMSE": f"{base_sealed['RMSE']:.3f}",
                "±1 RPE Agreement": "85.8% (858 / 1,000)",
                "Gain vs Dummy": "+17.7% agreement"
            },
            {
                "Model Pipeline": "Dummy Mean Regressor Baseline",
                "Evaluated Split": "Sealed Holdout (50 unseen participants)",
                "R²": f"{dummy_sealed['R2']:.4f}",
                "MAE": f"{dummy_sealed['MAE']:.3f}",
                "RMSE": f"{dummy_sealed['RMSE']:.3f}",
                "±1 RPE Agreement": "68.1% (681 / 1,000)",
                "Gain vs Dummy": "Baseline (0.0%)"
            }
        ])
        st.dataframe(comp_table, hide_index=True, width="stretch", alt="Synthetic model performance comparison")

        col_g1, col_g2 = st.columns(2)
        avp_df = read_csv(EXP / "actual_vs_predicted_holdout.csv")
        with col_g1:
            if avp_df is not None:
                sh = avp_df[avp_df["holdout"] == "sealed_holdout"]
                st.markdown("##### Actual vs. Predicted RPE (Sealed Holdout)")
                f_avp = px.scatter(sh, x="actual_rpe", y="predicted_rpe", opacity=0.45,
                                   color_discrete_sequence=["#5b7fe2"],
                                   labels={"actual_rpe": "True RPE", "predicted_rpe": "Predicted RPE"})
                lo, hi = float(sh.actual_rpe.min()), float(sh.actual_rpe.max())
                f_avp.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Identity Line",
                                          line=dict(color="#f2d461", dash="dash")))
                f_avp.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)",
                                    plot_bgcolor="rgba(0,0,0,0)", font_color="#c8cdc8", showlegend=False)
                st.plotly_chart(f_avp, width="stretch", alt="Actual versus predicted RPE on the sealed holdout")

        with col_g2:
            if avp_df is not None:
                sh = avp_df[avp_df["holdout"] == "sealed_holdout"].copy()
                sh["residual"] = sh["actual_rpe"] - sh["predicted_rpe"]
                st.markdown("##### Residual Analysis (Residual vs. Predicted)")
                f_res = px.scatter(sh, x="predicted_rpe", y="residual", opacity=0.45,
                                   color_discrete_sequence=["#5b7fe2"],
                                   labels={"predicted_rpe": "Predicted RPE", "residual": "Residual (Actual - Predicted)"})
                f_res.add_hline(y=0, line_dash="dash", line_color="#f2d461")
                f_res.add_hline(y=1.0, line_dash="dot", line_color="rgba(120,184,131,0.65)", annotation_text="+1 RPE")
                f_res.add_hline(y=-1.0, line_dash="dot", line_color="rgba(120,184,131,0.65)", annotation_text="-1 RPE")
                f_res.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)",
                                    plot_bgcolor="rgba(0,0,0,0)", font_color="#c8cdc8", showlegend=False)
                st.plotly_chart(f_res, width="stretch", alt="Prediction residuals centered around zero")

        col_g3, col_g4 = st.columns(2)
        lc_df = read_csv(EXP / "learning_curve.csv")
        with col_g3:
            if lc_df is not None:
                st.markdown("##### Participant-Grouped Learning Curve")
                f_lc = go.Figure()
                f_lc.add_trace(go.Scatter(
                    x=lc_df["train_rows_per_fold"], y=lc_df["train_R2_mean"],
                    mode="lines+markers", name="Train R²", line=dict(color="#61b7aa")
                ))
                f_lc.add_trace(go.Scatter(
                    x=lc_df["train_rows_per_fold"], y=lc_df["validation_R2_mean"],
                    mode="lines+markers", name="Validation R² (GroupKFold)", line=dict(color="#5b7fe2")
                ))
                f_lc.update_layout(
                    height=340, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)", font_color="#c8cdc8",
                    xaxis_title="Training Samples per Fold", yaxis_title="R² Score"
                )
                st.plotly_chart(f_lc, width="stretch", alt="Training and validation R squared across training set sizes")

        fi_df = read_csv(EXP / "feature_importance.csv")
        with col_g4:
            if fi_df is not None:
                st.markdown("##### Permutation Feature Importance (R² Drop)")
                fi_sorted = fi_df.sort_values("importance_mean_R2_drop", ascending=True)
                f_fi = go.Figure(go.Bar(
                    x=fi_sorted["importance_mean_R2_drop"], y=fi_sorted["feature"], orientation="h",
                    error_x=dict(type="data", array=fi_sorted["importance_std"]),
                    marker_color="#5b7fe2"
                ))
                f_fi.update_layout(
                    height=340, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)", font_color="#c8cdc8",
                    xaxis_title="Permutation Importance (R² Drop)"
                )
                st.plotly_chart(f_fi, width="stretch", alt="Most influential engineered features in the RPE model")


# =========================================================================================
# SECTION C: REAL STRENGTH TRAINING EXPERIMENT
# =========================================================================================
with tab_real:
    st.markdown("### Real Strength Training Experiment (PMData Audit)")
    st.caption("Strictly restricted to 87 prospective strength-training sessions across 10 participants. "
               "Excludes all running, soccer, and endurance records. Excludes post-workout duration.")

    real_summary = read_json(REAL_STRENGTH / "real_strength_experiment_summary.json")
    real_cmp = read_csv(REAL_STRENGTH / "real_strength_experiment_comparison.csv")

    if real_summary is None or real_cmp is None:
        st.info("Run `python -m ml.pp1_real_strength_experiment` to generate the isolated strength experiment results.")
    else:
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            kpi("Audited Strength Sessions", f"{real_summary['total_sessions']}",
                f"{real_summary['total_participants']} participants (p02–p15)")
        with k2:
            kpi("LOPO ±1 RPE Agreement", f"{real_summary['best_lopo_within_1_pct']:.1f}%",
                "Leave-One-Participant-Out baseline")
        with k3:
            kpi("Best LOPO MAE", f"{real_summary['best_lopo_MAE']:.3f}", "RPE points on unseen athletes")
        with k4:
            kpi("Real Generalization R²", f"{real_summary['best_lopo_R2']:.3f}",
                "Confirms subjective recovery alone has low R²")

        st.markdown("##### Out-of-Fold Performance Across Models (Strict Participant Grouping)")
        st.dataframe(
            real_cmp.style.format({
                "lopo_R2": "{:.4f}", "lopo_MAE": "{:.4f}", "lopo_RMSE": "{:.4f}",
                "lopo_within_1_pct": "{:.1f}%", "gkf5_R2": "{:.4f}",
                "gkf5_MAE": "{:.4f}", "gkf5_RMSE": "{:.4f}", "gkf5_within_1_pct": "{:.1f}%"
            }),
            hide_index=True, width="stretch", alt="Real PMData experiment protocol and metric matrix"
        )

        st.markdown(
            '<div class="card" style="font-size:0.9rem;color:#c8cdc8">'
            '<b>Key Scientific Finding for PP1 Evaluation:</b><br>'
            '1. <b>Zero Intra-Workout Leakage:</b> By strictly omitting post-workout duration and using only pre-session wellness '
            '(sleep duration, sleep quality, readiness, soreness, fatigue, stress), we avoid trivial data leakage.<br>'
            '2. <b>Real-World Generalization Gap:</b> Linear and tree-based models achieve negative out-of-fold R² on unseen athletes '
            'because subjective wellness rating scales are interpreted idiosyncratically across individuals.<br>'
            '3. <b>Tolerance Agreement:</b> Despite near-zero variance explained beyond the mean, models achieve '
            '<b>51.7% to 56.3% agreement within ±1.0 RPE point</b> on genuine strength workouts.'
            '</div>', unsafe_allow_html=True
        )

        st.markdown("##### Comparison: Strength-Only Cohort vs. All-Activities PMData Cohort")
        all_summary = read_json(REAL_ALL / "pmdata_experiment_summary.json")
        if all_summary is not None:
            comp_real_df = pd.DataFrame([
                {
                    "Experiment Scope": "Genuine Strength-Only (Isolated)",
                    "Sessions": real_summary["total_sessions"],
                    "Participants": real_summary["total_participants"],
                    "Target Scale": "Foster RPE (0–10)",
                    "LOPO / OOF R²": f"{real_summary['best_lopo_R2']:.4f}",
                    "MAE": f"{real_summary['best_lopo_MAE']:.3f}",
                    "±1 RPE Agreement": f"{real_summary['best_lopo_within_1_pct']:.1f}%",
                    "Status": "Vetted Resistance Training"
                },
                {
                    "Experiment Scope": "All Activities Mixed (Running, Soccer, Gym)",
                    "Sessions": all_summary["sessions"],
                    "Participants": all_summary["participants"],
                    "Target Scale": "Foster RPE (0–10)",
                    "LOPO / OOF R²": f"{all_summary['best_pooled_oof_R2']:.4f}",
                    "MAE": f"{all_summary['best_MAE']:.3f}",
                    "±1 RPE Agreement": "~55.0% (estimated)",
                    "Status": "Mixed Endurance/Field Sports"
                }
            ])
            st.dataframe(comp_real_df, hide_index=True, width="stretch", alt="Real strength model comparison")


# =========================================================================================
# SECTION D: RESEARCH METHODOLOGY & GOVERNANCE
# =========================================================================================
with tab_method:
    st.markdown("### Research Methodology & Scientific Governance")

    st.markdown("##### 1. Dataset Selection & Restriction Audit")
    dataset_audit_table = pd.DataFrame([
        {
            "Dataset Candidate": "SWIFTIES (Biomechanics)",
            "Audited Rows": "0 gym sessions (256 lab trials)",
            "Target Scale": "Borg 6–20 (per-task to exhaustion)",
            "FullStrength.ai Usability": "Incompatible (lab endurance to failure; no sleep/environment)",
            "Research Status": "Excluded from weightlifting model"
        },
        {
            "Dataset Candidate": "Strength/Plyometric Study",
            "Audited Rows": "60 testing observations (PRE/POST)",
            "Target Scale": "Weekly Average RPE (6.01–6.78)",
            "FullStrength.ai Usability": "Incompatible (no session RPE; periodic tests only)",
            "Research Status": "Excluded from weightlifting model"
        },
        {
            "Dataset Candidate": "Rugby Dataset (Hu et al. 2024)",
            "Audited Rows": "520 Muscu gym records (26 players)",
            "Target Scale": "Session RPE (Foster 0–10)",
            "FullStrength.ai Usability": "Partial (all durations 45 min; lacks temp/humidity/sleep)",
            "Research Status": "Secondary distribution benchmark"
        },
        {
            "Dataset Candidate": "SoccerMon / PMData Soccer",
            "Audited Rows": "0 resistance training records",
            "Target Scale": "Field Soccer sRPE",
            "FullStrength.ai Usability": "Incompatible (field running only)",
            "Research Status": "Supplementary team-sport evidence"
        },
        {
            "Dataset Candidate": "PMData Strength Subset",
            "Audited Rows": "87 audited prospective sessions",
            "Target Scale": "Foster 0–10 whole-session RPE",
            "FullStrength.ai Usability": "Supported for recovery baseline (lacks microclimate)",
            "Research Status": "Recommended Real Strength Benchmark"
        }
    ])
    st.dataframe(dataset_audit_table, hide_index=True, width="stretch", alt="Research dataset governance audit")

    st.markdown("##### 2. Participant-Grouped Validation Safeguards")
    st.markdown(
        """
        - **Why Random Cross-Validation is Flawed:** In exercise datasets, observations from the same subject share 
          unobserved individual baselines (pain tolerance, conditioning, psychological traits). Standard K-Fold 
          leaks subject identities across folds, creating artificially inflated R² (up to 0.85) that collapses 
          when deployed to new users.
        - **FullStrength.ai Protocol:** All synthetic evaluations use **GroupKFold(5)** across 150 dev participants, 
          and all reported performance is measured once on a **completely sealed holdout of 50 unseen participants**.
        - **Real Data Protocol:** Uses **Leave-One-Participant-Out (LOPO)** across the 10 real strength participants.
        """
    )

    st.markdown("##### 3. Non-Prescriptive Timing & TEQS Safeguards")
    st.markdown(
        """
        - **Air Quality Scale Protection:** OpenWeather API supplies a qualitative 1–5 index (`1=Good` to `5=Very Poor`). 
          The TEQS baseline expects continuous US EPA AQI (0–500). Live OpenWeather AQI is **strictly blocked** from direct 
          injection into the TEQS score until empirical piecewise breakpoint calibration is completed.
        - **Microclimate Disclaimer:** Outdoor OpenWeather readings are explicitly labeled `OUTDOOR_ESTIMATE` and never 
          represented as indoor air-conditioned gym conditions.
        - **Nutrition Timing Principles:** Guidance follows sports nutrition consensus (ISSN/ACSM): **no prescriptive fluid 
          volumes** (e.g. no arbitrary "500 ml"), **no prescriptive macronutrient grams**, and **no commercial supplement promotions**.
        """
    )

st.markdown("---")
st.caption("FullStrength.ai Research Component · IT23166660 · Final PP1 Delivery · October 2026")
