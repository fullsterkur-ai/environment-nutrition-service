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

<kbd>🌿 branch: <b>main</b></kbd>  &nbsp; <kbd>🎓 IT23166660</kbd>  &nbsp; <kbd>🤖 FastAPI + ML</kbd>

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

> Nutrition and hydration coach for Fullsterkur.ai. Produces a Training-Environment Quality Score (TEQS) and timing guidance from environmental and readiness data.

<br />
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#tech-stack">Tech Stack</a></li>
    <li><a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
        <li><a href="#training-the-model">Training the Model</a></li>
      </ul>
    </li>
    <li><a href="#project-structure">Project Structure</a></li>
  </ol>
</details>

---

## Tech Stack

| <img src="https://raw.githubusercontent.com/github/explore/main/topics/python/python.png" width="60" /> | <img src="https://raw.githubusercontent.com/github/explore/main/topics/scikit-learn/scikit-learn.png" width="60" /> | <img src="https://raw.githubusercontent.com/github/explore/main/topics/pandas/pandas.png" width="60" /> | <img src="https://raw.githubusercontent.com/github/explore/main/topics/fastapi/fastapi.png" width="60" /> |
|:---:|:---:|:---:|:---:|
| **Python**<br>Language the service and model pipeline are built in. | **scikit-learn**<br>Trains the TEQS regression and completion-classification models. | **pandas / NumPy**<br>Data loading, cleaning, and feature preparation for training. | **FastAPI** *(planned)*<br>Will serve the trained model to the Mobile app over HTTP. |

---

## Getting Started

### Prerequisites

- [Python](https://www.python.org/downloads/)

### Installation

1. Clone the repo
   ```
   git clone https://github.com/fullsterkur-ai/environment-nutrition-service.git
   cd environment-nutrition-service
   ```
2. Create and activate a virtual environment
   ```
   python -m venv .venv
   ```
   Windows:
   ```
   .venv\Scripts\activate
   ```
   macOS/Linux:
   ```
   source .venv/bin/activate
   ```
3. Install dependencies
   ```
   pip install -r requirements.txt
   ```

### Training the Model

```
python train_models.py --csv data/TEQS_Synthetic_5000_Development_Dataset.csv --outdir artifacts
```

With Random Forest tuning:

```
python train_models.py --csv data/TEQS_Synthetic_5000_Development_Dataset.csv --outdir artifacts --tune
```

Outputs are written to `artifacts/`: `best_rpe_model.joblib`, `best_completion_model.joblib`, `model_summary.json`, `rpe_feature_importance.csv`, and the train/test participant lists.

See [README_MODEL_PIPELINE.md](README_MODEL_PIPELINE.md) for the research methodology, leakage rules, and model strategy behind this pipeline.

---

## Project Structure

```
environment-nutrition-service/
├── artifacts/   # Trained models and training outputs
├── data/        # Datasets
├── ml/          # Model training code
└── requirements.txt
```

---

## See Also

- [Main repository README](https://github.com/fullsterkur-ai/J26-SE-342): project overview, architecture, components, and team
- [DEVELOPMENT.md](https://github.com/fullsterkur-ai/J26-SE-342/blob/main/DEVELOPMENT.md): coding conventions and branching/merging strategy

<p align="right">(<a href="#environment-nutrition-service">back to top</a>)</p>
