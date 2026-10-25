<!-- FS-COLOR-HEADER-START -->
<div align="center">

# 🏋️‍♂️ FullStrength.ai — TEQS Service

### 🌡️ Environmental Quality & Nutrition Timing Optimizer

<p>
<img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
<img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"/>
<img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn"/>
<img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas"/>
<img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy"/>
</p>

<p>
<img src="https://img.shields.io/badge/Status-Development_Only-FF8C00?style=flat-square" alt="Status"/>
<img src="https://img.shields.io/badge/Stage-3_Timing_Engine-6A5ACD?style=flat-square" alt="Stage"/>
<img src="https://img.shields.io/badge/Research-PP1_Oct_2026-0A66C2?style=flat-square" alt="Research"/>
<img src="https://img.shields.io/badge/License-Research_Only-success?style=flat-square" alt="License"/>
</p>

<kbd>🌿 branch: <b>feat/teqs-02-ml-pipeline</b></kbd>  &nbsp; <kbd>🎓 IT23166660</kbd>  &nbsp; <kbd>🤖 FastAPI + ML</kbd>

</div>

---

### ✨ What this service does

> 🌡️ **Environment** → 🥗 **Nutrition timing** → 💧 **Hydration windows** → 🧠 **RPE prediction** → 📊 **TEQS score**

The **Environmental Quality & Nutrition Timing Optimizer (TEQS)** scores how ambient training conditions (temperature, humidity, air quality) combined with athlete recovery state (sleep, readiness) shape resistance-training readiness, session RPE, and nutrition / hydration timing windows.

### 📈 Model Performance Visuals

<p align="center">
  <img src="https://raw.githubusercontent.com/fullsterkur-ai/environment-nutrition-service/feature/teqs-model-pipeline-v2/artifacts/pp1_model_optimization/actual_vs_predicted.png" alt="Actual vs Predicted" width="45%"/>
  &nbsp;
  <img src="https://raw.githubusercontent.com/fullsterkur-ai/environment-nutrition-service/feature/teqs-model-pipeline-v2/artifacts/pp1_model_optimization/feature_importance.png" alt="Feature Importance" width="45%"/>
</p>
<p align="center">
  <img src="https://raw.githubusercontent.com/fullsterkur-ai/environment-nutrition-service/feature/teqs-model-pipeline-v2/artifacts/pp1_model_optimization/learning_curve.png" alt="Learning Curve" width="45%"/>
  &nbsp;
  <img src="https://raw.githubusercontent.com/fullsterkur-ai/environment-nutrition-service/feature/teqs-model-pipeline-v2/artifacts/pp1_model_optimization/model_comparison.png" alt="Model Comparison" width="45%"/>
</p>

### 🧭 Quick Start

```powershell
cd environment-nutrition-service
python -m venv venv
.\venv\Scripts\pip install -r requirements.txt
.\venv\Scripts\uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Then open 👉 `http://127.0.0.1:8000/docs`

### 🧩 Features at a glance

| 🎯 Feature | 📝 What it does |
|---|---|
| 🌡️ **Environment ingest** | Live OpenWeather data: temperature, humidity, AQI |
| 📊 **TEQS score** | Six sub-scores → single training-environment index |
| 🧠 **RPE prediction** | sklearn pipeline on 5,000-session synthetic dataset |
| 💧 **Timing engine** | Rule-based nutrition / hydration windows |
| 📈 **Dashboard** | Streamlit view for the model artifacts |

<!-- FS-COLOR-HEADER-END -->

---

**Student ID:** IT23166660  
**Service:** `environment-nutrition-service`  
**Stage:** 3 — Context-Aware Nutrition & Hydration Timing Engine  
**System Status:** `DEVELOPMENT_ONLY` / `EXPERIMENTAL_UNVALIDATED` / `EXPERIMENTAL_RULE_BASED`

---

## 1. Component Overview & Scientific Context

The **Environmental Quality & Nutrition Timing Optimizer (TEQS)** is a core research component of **FullStrength.ai**. It assesses how ambient training conditions (temperature, humidity, air quality) and athlete baseline recovery states (sleep duration, sleep quality, subjective readiness) influence resistance training readiness, perceived exertion (Session RPE), and nutritional/hydration timing windows.

### Critical Research Notice & Empirical Grounding
* **PMData Real-World Evaluation:** GroupKFold cross-validation on the prospective PMData strength cohort (N = 539 audited prospective sessions, 15 participants; N = 87 strength sessions across 10 participants) demonstrated weak predictive generalization (Ridge regression wellness-only MAE = 1.414 vs mean baseline MAE = 1.466; pooled out-of-fold $R^2 = 0.013$). No strength-specific ML model outperformed the mean baseline.
* **Deployment Policy:** PMData experimental models are **not** deployed as validated scientific models.
* **Synthetic Development Pipeline:** The RPE prediction model is trained on the 5,000-session synthetic development pipeline (`artifacts/best_rpe_model.joblib`). All inferences are explicitly labeled **`DEVELOPMENT_ONLY`** and accompanied by non-clinical advisories.
* **TEQS Baseline:** The TEQS scoring algorithm uses an exploratory, unvalidated development baseline (`ml/teqs_baseline.py`) tagged **`EXPERIMENTAL_UNVALIDATED`**.
* **Timing Engine:** The context-aware nutrition and hydration timing engine is an educational research prototype tagged **`EXPERIMENTAL_RULE_BASED`**. It provides general guidance, **not** clinical prescriptions.

---

## 2. Backend Architecture (`app/`)

The backend is built with FastAPI and organized into modular layers:

```
environment-nutrition-service/
├── .env.example                    # Template for environment variables (OpenWeather API key)
├── .env                            # Local git-ignored secrets configuration
├── app/
│   ├── __init__.py
│   ├── main.py                     # FastAPI application factory, CORS, health endpoint
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py               # Settings, OpenWeather configuration & artifact paths
│   │   └── timing_rules.py         # Timing engine rules, metadata, and scientific rationales
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── health.py               # Health check response schema
│   │   ├── teqs.py                 # TEQS evaluate request & subscore response schemas
│   │   ├── rpe.py                  # RPE prediction request & response schemas
│   │   ├── environment.py          # LocationInfo & EnvironmentCurrentResponse schemas
│   │   └── timing.py               # TimingEvaluationRequest & Response schemas
│   ├── services/
│   │   ├── __init__.py
│   │   ├── teqs_service.py         # TEQS baseline evaluation service (ml/teqs_baseline.py)
│   │   ├── rpe_service.py          # Synthetic model inference service (artifacts/best_rpe_model.joblib)
│   │   ├── weather_service.py      # Async OpenWeather client with 10-min cache & reliability guards
│   │   └── timing_service.py       # Context-aware rule-based nutrition and hydration timing service
│   └── api/
│       ├── __init__.py
│       └── v1/
│           ├── __init__.py
│           ├── router.py           # v1 router aggregation (teqs, rpe, environment, timing)
│           └── endpoints/
│               ├── __init__.py
│               ├── teqs.py         # POST /api/v1/teqs/evaluate
│               ├── rpe.py          # POST /api/v1/rpe/predict
│               ├── environment.py  # GET /api/v1/environment/current
│               └── timing.py       # POST /api/v1/timing/evaluate
├── artifacts/                      # Model checkpoints and experiment artifacts
│   ├── best_rpe_model.joblib       # Synthetic Ridge regression pipeline
│   └── model_summary.json          # Metrics metadata
├── ml/                             # ML development pipelines and baseline scripts
│   ├── teqs_baseline.py            # TEQS scoring scaffold
│   ├── predict.py                  # Pre-session feature inference CLI
│   └── train_models.py             # Synthetic pipeline training script
├── tests/
│   ├── test_api_backend.py         # Automated API tests (Health, TEQS, RPE, Validation)
│   ├── test_environment_api.py     # Automated mock tests for OpenWeather integration & caching
│   ├── test_timing_api.py          # Automated tests for timing engine & context rules
│   └── test_pmdata_pipeline.py     # PMData research integrity & audit tests
├── timing_engine_methodology.md    # In-depth timing engine methodology & research framework
├── pytest.ini                      # Pytest runner configuration
└── requirements.txt                # Production and development dependencies
```

---

## 3. Configuration & API Keys

### OpenWeather API Key (`.env`)
1. Obtain an API key from [https://openweathermap.org/api](https://openweathermap.org/api).
2. Copy `.env.example` to `.env` in `environment-nutrition-service`:
   ```powershell
   Copy-Item .env.example .env
   ```
3. Set your key in `.env`:
   ```ini
   OPENWEATHER_API_KEY=your_actual_key_here
   ```
4. `.env` is git-ignored and never committed or exposed in responses.

---

## 4. Installation & Running the Server

### Working Directory
Ensure you are in the `environment-nutrition-service` directory:

```powershell
cd C:\Users\rivinma\Desktop\ResearchProject\environment-nutrition-service
```

### Activate Virtual Environment & Start Server
```powershell
.\venv\Scripts\activate
.\venv\Scripts\uvicorn.exe app.main:app --host 127.0.0.1 --port 8000 --reload
```

---

## 5. API Endpoints & Usage Demonstration

Interactive Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)  
ReDoc: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

### 5.1 `GET /health`
Inspects operational health, semantic version, and ML artifact status.

---

### 5.2 `POST /api/v1/teqs/evaluate`
Evaluates exploratory pre-session TEQS baseline score (0–100) and domain subscores (`EXPERIMENTAL_UNVALIDATED`).

---

### 5.3 `POST /api/v1/rpe/predict`
Predicts session RPE using the synthetic development Ridge pipeline (`DEVELOPMENT_ONLY`).

---

### 5.4 `GET /api/v1/environment/current`
Fetches real-time outdoor temperature, humidity, and OpenWeather categorical air quality (1–5) with 10-minute in-memory caching (`OUTDOOR_ESTIMATE`).

---

### 5.5 `POST /api/v1/timing/evaluate`
Evaluates context-aware nutrition and hydration timing guidance across pre-workout, during-workout, and post-workout phases.

#### Input Fields:
| Field | Type | Required | Valid Range / Options | Description |
|---|---|---|---|---|
| `workout_type` | string | No (default `Mixed Strength`) | 1–100 chars | Workout modality |
| `planned_duration_min` | float | **Yes** | `1.0` to `360.0` | Planned duration in minutes |
| `planned_intensity_1_10` | float | **Yes** | `1.0` to `10.0` | Planned intensity rating |
| `environment_type` | string | No (default `INDOOR_NON_AC`) | `INDOOR_AC`, `INDOOR_NON_AC`, `SEMI_OUTDOOR`, `OUTDOOR` | Training environment setting |
| `temperature_c` | float | No | `-10.0` to `60.0` | Ambient temperature in °C |
| `humidity_pct` | float | No | `0.0` to `100.0` | Relative humidity percentage |
| `environmental_data_source` | string | No | `OPENWEATHER`, `SENSOR`, `MANUAL`, `NOT_SPECIFIED` | Data source tag |
| `sleep_hours` | float | No | `0.0` to `24.0` | Prior sleep duration in hours |
| `readiness_1_5` | float | No | `1.0` to `5.0` | Subjective readiness rating |
| `pre_workout_meal_timing_category` | string | No | `FASTED_OR_EMPTY`, `LESS_THAN_1_HOUR`, `1_TO_2_HOURS`, `2_TO_4_HOURS`, `MORE_THAN_4_HOURS` | Meal timing proximity |
| `experimental_teqs_score` | float | No | `0.0` to `100.0` | Exploratory TEQS development score |

#### Example Request:
```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/v1/timing/evaluate" `
  -H "Content-Type: application/json" `
  -d '{
    "workout_type": "Mixed Strength",
    "planned_duration_min": 75.0,
    "planned_intensity_1_10": 8.0,
    "environment_type": "INDOOR_NON_AC",
    "temperature_c": 31.0,
    "humidity_pct": 78.0,
    "pre_workout_meal_timing_category": "2_TO_4_HOURS",
    "experimental_teqs_score": 52.0
  }'
```

#### Example Response (200 OK):
```json
{
  "validated_input_context": {
    "workout_type": "Mixed Strength",
    "planned_duration_min": 75.0,
    "planned_intensity_1_10": 8.0,
    "environment_type": "INDOOR_NON_AC",
    "environmental_data_source": "MANUAL",
    "temperature_c": 31.0,
    "humidity_pct": 78.0,
    "sleep_hours": null,
    "readiness_1_5": null,
    "pre_workout_meal_timing_category": "2_TO_4_HOURS",
    "experimental_teqs_score": 52.0,
    "teqs_validation_status": "EXPERIMENTAL_UNVALIDATED"
  },
  "pre_workout_guidance": {
    "phase": "PRE_WORKOUT",
    "title": "Pre-Workout Preparation & Fluid Readiness",
    "guidance": [
      "Maintain baseline fluid readiness before training by drinking water comfortably in the hours leading to your workout to avoid starting in a fluid deficit.",
      "In warm or humid non-air-conditioned conditions, prioritize pre-session fluid availability to support early sweat onset and evaporative heat regulation.",
      "For high-intensity exertion (intensity ≥ 8/10), ensure pre-session energy availability to support elevated anaerobic glycolytic demand."
    ],
    "contextual_factors": [
      "baseline_hydration",
      "heat_humidity_strain",
      "high_intensity"
    ],
    "explanation": "Pre-workout guidance focuses on establishing adequate fluid readiness and gastrointestinal comfort conditioned on meal proximity, thermal environment, and planned workout intensity.",
    "evidence_status": "EDUCATIONAL_PROTOTYPE_UNVALIDATED"
  },
  "during_workout_guidance": {
    "phase": "DURING_WORKOUT",
    "title": "Intra-Workout Hydration & Session Comfort",
    "guidance": [
      "Keep water or fluids readily accessible and drink according to thirst and comfort to maintain training stamina.",
      "In warm or humid settings, sweat rates increase significantly. Drink to thirst regularly between exercise sets and seek airflow or shade between working intervals to support cooling comfort.",
      "For extended training exceeding 75 minutes, muscular glycogen depletion and cumulative fluid loss become more pronounced. Consider access to easily digestible carbohydrate fluids or foods to sustain training output.",
      "Because cumulative environmental and readiness indicators are sub-optimal, rate of perceived exertion (RPE) may elevate faster than usual. Take slightly longer rest intervals between heavy working sets if necessary."
    ],
    "contextual_factors": [
      "baseline_fluid_access",
      "elevated_sweat_rate",
      "extended_duration",
      "teqs_limiters_active"
    ],
    "explanation": "During-workout guidance balances thermal cooling comfort, session duration demands, and hydration availability without imposing rigid fluid volume mandates.",
    "evidence_status": "EDUCATIONAL_PROTOTYPE_UNVALIDATED"
  },
  "post_workout_guidance": {
    "phase": "POST_WORKOUT",
    "title": "Post-Workout Recovery & Rehydration",
    "guidance": [
      "Plan a balanced recovery meal or snack containing protein and carbohydrates according to your regular daily meal schedule to support muscle protein synthesis and recovery.",
      "Rehydrate gradually after exercise by drinking fluids to quench thirst and restoring fluid balance across subsequent hours.",
      "Following workouts in warm or humid conditions with high sweat loss, include sodium/electrolytes through regular dietary foods or beverages to facilitate effective fluid retention.",
      "Given the high intensity or extended duration, prioritize carbohydrate-containing whole foods in your post-training meal to replenish glycogen stores."
    ],
    "contextual_factors": [
      "protein_synthesis",
      "fluid_restoration",
      "electrolyte_restoration",
      "glycogen_replenishment"
    ],
    "explanation": "Post-workout recovery emphasizes gradual rehydration to thirst and balanced dietary intake supporting muscle remodeling and energy store replenishment.",
    "evidence_status": "EDUCATIONAL_PROTOTYPE_UNVALIDATED"
  },
  "applied_rules": [
    "RULE_ENV_HEAT_HUMIDITY",
    "RULE_DURATION_EXTENDED",
    "RULE_INTENSITY_HIGH",
    "RULE_TEQS_CONSTRAINED"
  ],
  "explanation": "Guidance tailored for a 75-minute Mixed Strength session at planned intensity 8.0/10 in a INDOOR_NON_AC setting. Elevated thermal conditions (31.0°C, 78.0% humidity) increase sweat loss, making fluid availability and cooling comfort primary priorities. An exploratory TEQS score of 52.0/100 reflects active situational limiters. (Note: TEQS is strictly an EXPERIMENTAL_UNVALIDATED research baseline, not a clinical threshold).",
  "evidence_status": "EXPERIMENTAL_RULE_BASED",
  "research_disclaimer": "This timing engine is an educational research prototype providing general sports nutrition guidance. It does NOT offer medical nutrition therapy, clinical diagnoses, or individualized dietary prescriptions. Exploratory TEQS scores are unvalidated development placeholders and must not be treated as clinical safety thresholds."
}
```

---

## 6. Critical Scientific Principles & Constraints

For full methodology and rule catalog, see [`timing_engine_methodology.md`](file:///C:/Users/rivinma/Desktop/ResearchProject/environment-nutrition-service/timing_engine_methodology.md).

### Non-Prescriptive Research Guardrails:
* **No Fabricated Fluid Volumes:** Avoids arbitrary milliliter mandates (e.g., no "drink 450 ml every 15 min").
* **No Exact Gram Targets:** Avoids prescriptive macronutrient claims (e.g., no "consume 30g protein and 60g carbohydrates").
* **No Supplement Prescriptions:** Avoids prescribing creatine, stimulants, or commercial sports products.
* **AQI Compatibility:** OpenWeather AQI (1–5 qualitative index) is kept distinct from continuous US EPA AQI (0–500) and is never passed directly into the TEQS formula.
* **Outdoor vs Indoor Caveat:** Outdoor weather estimates are explicitly labeled `OUTDOOR_ESTIMATE` and never claimed as indoor gym measurements.

---

## 7. Automated Test Suite (29 / 29 Passing)

Run tests via pytest:

```powershell
.\venv\Scripts\pytest.exe -v
```

### Test Breakdown:
* **Backend Core & ML Pipeline Tests (`tests/test_api_backend.py` — 8 tests):**
  Health endpoint, TEQS baseline evaluation, invalid inputs rejection, RPE synthetic model inference, 503 missing artifact handling, response schemas.
* **OpenWeather Environment API Tests (`tests/test_environment_api.py` — 7 tests):**
  Weather/AQI parsing, in-memory caching, 503 missing key handling, 404 invalid city, 504 timeout, graceful air pollution fallback, 1–5 AQI mapping.
* **Nutrition Timing Engine Tests (`tests/test_timing_api.py` — 9 tests):**
  Baseline valid requests, missing optional input handling, validation rejections (422), indoor AC vs outdoor heat/humidity context, duration & intensity differentiation, meal proximity & sleep rules, TEQS score tagging, strict absence of precise quantitative prescriptions.
* **PMData Audited Research Pipeline Tests (`tests/test_pmdata_pipeline.py` — 5 tests):**
  Prospective timestamp verification, dataset preservation, group split isolation, feature leakage prevention.

---

## 8. Readiness for Flutter Mobile Client Integration

Both endpoints are ready for mobile consumption:
1. **`GET /api/v1/environment/current?city=...`**: Fetches weather, temperature, humidity, and AQI category for pre-session display.
2. **`POST /api/v1/timing/evaluate`**: Consumes pre-session workout context + environment data to present pre-workout, during-workout, and post-workout timing cards.
3. **CORS:** Enabled globally.
4. **Structured Typed JSON:** Well-defined schemas with descriptive titles and bullet points.
