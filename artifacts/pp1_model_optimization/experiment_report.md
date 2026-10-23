# PP1 Model Optimization Report (IT23166660)

All numbers come from `final_holdout_metrics.json`, `experiment_comparison.csv` and `real_pmdata/*`,
produced by `python -m ml.pp1_optimize` and `python -m ml.pp1_pmdata_experiment`.

## Protocol / holdout provenance
- The original baseline (`ml/train_models.py`) chose Ridge by **test MAE**, so its 50 test participants
  were already used for model selection ("legacy holdout").
- PP1 splits (disjoint participants): **dev 150** (all tuning, GroupKFold(5)), **sealed holdout 50**
  (new participants drawn from the original training set; evaluated once after freezing),
  **legacy holdout 50** (comparison only).
- Selection rule fixed in advance: highest mean GroupKFold R2 on dev. Baseline artifacts untouched.
- Only pre-session features are used; leakage columns are excluded (see `training_configuration.json`).

## Synthetic development result (NOT real-athlete validation)
| Model | Sealed holdout R2 | MAE | RMSE |
|---|---|---|---|
| Baseline recipe (Ridge, raw, fit on dev) | 0.542 | 0.541 | 0.675 |
| **Optimized: ElasticNet + engineered features** | **0.557** (95% CI 0.518-0.593) | 0.531 | 0.663 |
| Mean predictor | -0.000 | 0.797 | 0.997 |

Dev GroupKFold R2 = 0.554 +/- 0.021; train R2 0.563 (gap 0.009). Legacy holdout: original artifact 0.556,
optimized 0.567.

**R2 >= 0.60 was NOT achieved.** Best validated value: 0.557 (sealed), 0.554 (CV).

## Why the ceiling is ~0.55-0.57
- 13 tuned candidates (Ridge, ElasticNet, RF, ExtraTrees, HistGB, XGBoost; raw vs engineered features)
  all land in CV R2 0.518-0.554; non-linear models do not beat linear ones and overfit more
  (train R2 up to 0.69).
- Residual std 0.67 vs target std 1.00; participant means explain only ~5% of residual variance and
  adding participant identity did not help -> remaining error is session-level noise not explained by
  the 9 pre-session inputs. Planned intensity carries almost all the signal (permutation R2 drop 0.88).
- Reaching 0.60 would require information the data lacks (or changing labels, which was prohibited).
- CatBoost not installed; not used.

## Real PMData (separate, 539 sessions, 15 participants, no environment data)
Nested participant-grouped CV (5-fold x5 repeats + leave-one-participant-out), 13 candidates.
Dummy mean pooled R2 -0.018 (MAE 1.459); best candidate (ElasticNet, wellness-only) R2 +0.006,
MAE 1.422 (selection-optimistic). No candidate exceeded R2 0.10. Real performance remains very weak.
The synthetic R2 must not be cited as real-world performance.
