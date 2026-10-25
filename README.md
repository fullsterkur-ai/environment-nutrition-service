# Environment Nutrition Service

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
