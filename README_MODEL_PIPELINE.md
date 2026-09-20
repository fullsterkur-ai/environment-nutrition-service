
# TEQS Model Development Pipeline

This starter package is for **development and research prototyping** of the Environmental Quality & Nutrition Timing Optimizer.

## Critical research rule

The included 5,000-row dataset is synthetic.  
Metrics obtained from it are **not scientific evidence** and must not be reported as final model performance.

## Recommended research architecture

### A. TEQS layer
Environmental + readiness variables:
- temperature
- humidity
- AQI
- sleep duration / quality
- mood/readiness

produce an explainable TEQS.

### B. Session-outcome model
Use only information available **before the workout** to predict an observable post-session outcome, for example:
- session RPE (regression)
- completion status (classification)

This supports TEQS calibration.

### C. Timing engine
TEQS + workout context:
- workout type
- planned duration
- planned intensity

produce pre-, intra-, and post-workout timing guidance.

---

## Why some columns must NOT be model inputs

For a pre-session model, these are leakage:
- actual_duration_min
- session_rpe_1_10
- completion_status
- performance_deviation_pct

They are only known during or after the session.

Using them to predict a pre-session outcome would make the test unrealistically easy.

---

## Why participant-grouped splitting matters

The synthetic dataset has repeated sessions per participant.

A normal random row split can put sessions from the same person in both train and test data.  
That can make performance appear better than real generalisation.

`train_models.py` therefore splits by `participant_id`.

For real research, keep this idea:
**unseen participants should appear in the test set whenever the goal is generalisation to new users.**

---

## Run

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install:

```bash
pip install -r requirements.txt
```

Train:

```bash
python train_models.py --csv data/TEQS_Synthetic_5000_Development_Dataset.csv --outdir artifacts
```

Optional advanced Random Forest tuning:

```bash
python train_models.py --csv data/TEQS_Synthetic_5000_Development_Dataset.csv --outdir artifacts --tune
```

Outputs:
- `best_rpe_model.joblib`
- `best_completion_model.joblib`
- `model_summary.json`
- `rpe_feature_importance.csv`
- train/test participant lists

---

## Model strategy

### Regression task
Target:
`session_rpe_1_10`

Candidates:
1. Ridge regression — interpretable baseline
2. Random Forest — non-linear model
3. Histogram Gradient Boosting — non-linear model

Metrics:
- MAE
- RMSE
- R²

### Classification task
Target:
`completion_status`

Candidates:
1. Logistic Regression
2. Random Forest Classifier
3. Histogram Gradient Boosting Classifier

Primary metric:
- macro F1

Why macro F1?
Because completion classes may be imbalanced.

---

## Advanced concerns for the REAL dataset

1. **Data leakage**
   Do not use post-session outcomes as pre-session inputs.

2. **Participant leakage**
   Split by participant.

3. **Indoor vs outdoor environment**
   Weather API location data may not equal actual gym temperature/humidity.

4. **Timestamp alignment**
   Weather observation must correspond closely to workout start time.

5. **Missing data**
   Decide how to handle missing sleep/mood/environment values.

6. **Self-report bias**
   Mood, sleep quality and RPE are subjective.

7. **Class imbalance**
   Most sessions may be "completed", making naive accuracy misleading.

8. **Small sample size**
   Complex models can overfit. Prefer simple baselines first.

9. **Feature drift**
   Different seasons, gyms, and athletes may change the data distribution.

10. **Calibration**
    A model predicting RPE is not automatically a scientifically validated TEQS.
    TEQS still needs a defensible mapping/calibration strategy.

11. **Explainability**
    Keep factor contributions visible. Use permutation importance or SHAP after a final model is selected.

12. **Model versioning**
    Save model version, training-data version, feature list and metrics for every experiment.

---

## Recommended next research milestone

Do not optimise the synthetic-data model for weeks.

Use it only until this pipeline works end-to-end:

CSV -> cleaning -> training -> saved model -> FastAPI -> Flutter result

Then start real data collection.

