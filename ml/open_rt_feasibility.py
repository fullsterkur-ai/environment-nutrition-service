"""Exploratory participant-grouped feasibility check on two open resistance-training datasets.
Pre-session inputs only. Not a final result."""
import sys, json
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def metrics(y, p):
    e = p - y
    ss = ((y - y.mean()) ** 2).sum()
    return dict(n=int(len(y)), MAE=round(float(np.abs(e).mean()), 3), RMSE=round(float(np.sqrt((e ** 2).mean())), 3),
                R2=round(float(1 - (e ** 2).sum() / ss), 3), within1=round(float((np.abs(e) <= 1).mean() * 100), 1))

def boot_ci(y, p, g, B=1000, seed=0):
    rng = np.random.default_rng(seed); ids = np.unique(g); out = []
    for _ in range(B):
        pick = rng.choice(ids, len(ids), replace=True)
        idx = np.concatenate([np.where(g == i)[0] for i in pick])
        out.append((np.abs(p[idx] - y[idx]) <= 1).mean() * 100)
    return [round(float(np.percentile(out, 2.5)), 1), round(float(np.percentile(out, 97.5)), 1)]

def lopo(df, num, cat, target, group):
    X, y, g = df[num + cat], df[target].to_numpy(float), df[group].to_numpy()
    pre = ColumnTransformer([("n", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
                             ("c", OneHotEncoder(handle_unknown="ignore"), cat)])
    models = {"Dummy mean": DummyRegressor(),
              "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5),
              "RandomForest": RandomForestRegressor(n_estimators=300, max_depth=4, min_samples_leaf=5, random_state=42)}
    res = {}
    for name, m in models.items():
        pred = np.zeros(len(y))
        for tr, te in LeaveOneGroupOut().split(X, y, g):
            pipe = make_pipeline(pre, m).fit(X.iloc[tr], y[tr]); pred[te] = pipe.predict(X.iloc[te])
        r = metrics(y, pred); r["within1_95CI_participant_bootstrap"] = boot_ci(y, pred, g); res[name] = r
    # Warm-start personal baseline: mean of the athlete's own PREVIOUS sessions (sessions 6+ only)
    d = df.sort_values([group, "order"]).copy()
    d["prev_mean"] = d.groupby(group)[target].transform(lambda s: s.shift(1).expanding().mean())
    d["k"] = d.groupby(group).cumcount()
    w = d[(d.k >= 5) & d.prev_mean.notna() & d[target].notna()]
    r = metrics(w[target].to_numpy(float), w.prev_mean.to_numpy(float)); r["note"] = "evaluated on each athlete's 6th+ session"
    res["Personal running mean (warm start)"] = r
    return res

out = {}
# Remmert et al. 2026, OSF 5wprn: one row per participant x session x limb
r = pd.read_excel(sys.argv[1], sheet_name="session data").dropna(subset=["session.rpe"])
r["order"] = r["session"]; r["pid"] = r["id"].astype(str)
r["grp"] = r["pid"] + "_" + r["limb"]
num = ["perceived.recovery", "soreness", "mtt", "stress", "sleep.quality", "sleep.quantity", "nutrition", "session"]
res = lopo(r, num, ["condition"], "session.rpe", "pid")
# warm start must be per limb series
d = r.sort_values(["grp", "order"]).copy()
d["prev_mean"] = d.groupby("grp")["session.rpe"].transform(lambda s: s.shift(1).expanding().mean())
d["k"] = d.groupby("grp").cumcount(); w = d[(d.k >= 5) & d.prev_mean.notna()]
res["Personal running mean (warm start)"] = {**metrics(w["session.rpe"].to_numpy(float), w.prev_mean.to_numpy(float)), "note": "per limb, 6th+ session"}
out["Remmert2026_OSF_5wprn"] = {"rows": len(r), "participants": r.pid.nunique(), "target_mean": round(r["session.rpe"].mean(), 2),
                                "target_sd": round(r["session.rpe"].std(), 2), "sets_performed_by_condition": r.groupby("condition")["sets.performed"].describe()[["mean", "std", "min", "max"]].round(2).to_dict("index"), "results": res}
# Robinson et al. 2023/2025, OSF 58hy3: one row per participant x session
b = pd.read_excel(sys.argv[2], sheet_name="Subjective Scales").dropna(subset=["srpe"])
b["pid"] = b["condition"] + "_" + b["id"].astype(str); b["order"] = b["timepoint"]
b["day_in_week"] = ((b["timepoint"] - 1) % 3).astype(str)
res = lopo(b, ["prs", "mtt", "timepoint"], ["condition", "day_in_week"], "srpe", "pid")
out["Robinson2025_OSF_58hy3"] = {"rows": len(b), "participants": b.pid.nunique(), "target_mean": round(b.srpe.mean(), 2), "target_sd": round(b.srpe.std(), 2), "results": res}
print(json.dumps(out, indent=2))
json.dump(out, open(sys.argv[3], "w"), indent=2)
