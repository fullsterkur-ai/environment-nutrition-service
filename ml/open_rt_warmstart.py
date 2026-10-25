"""Warm-start comparison on identical rows (each series' 6th+ session), participant-grouped."""
import sys, json
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
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


def prep(df, series, target):
    d = df.sort_values([series, "order"]).copy()
    g = d.groupby(series)[target]
    d["prev_mean"] = g.transform(lambda s: s.shift(1).expanding().mean())
    d["prev_last"] = g.shift(1)
    d["k"] = d.groupby(series).cumcount()
    return d[(d.k >= 5) & d.prev_mean.notna()].reset_index(drop=True)

def run(d, num, cat, target, group):
    y, g = d[target].to_numpy(float), d[group].to_numpy()
    res = {"Personal running mean": d.prev_mean.to_numpy(float)}
    for name, feats in {"ElasticNet wellness only": num, "ElasticNet wellness + personal history": num + ["prev_mean", "prev_last"]}.items():
        pre = ColumnTransformer([("n", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), feats),
                                 ("c", OneHotEncoder(handle_unknown="ignore"), cat)])
        pred = np.zeros(len(y))
        for tr, te in LeaveOneGroupOut().split(d, y, g):
            pred[te] = make_pipeline(pre, ElasticNet(alpha=0.05, l1_ratio=0.5)).fit(d.iloc[tr], y[tr]).predict(d.iloc[te])
        res[name] = pred
    res["Dummy (training mean)"] = np.array([y[g != gi].mean() for gi in g])
    return {k: {**metrics(y, p), "within1_95CI": boot_ci(y, p, g)} for k, p in res.items()}

out = {}
r = pd.read_excel(sys.argv[1], sheet_name="session data").dropna(subset=["session.rpe"])
r["order"] = r["session"]; r["pid"] = r["id"].astype(str); r["series"] = r["pid"] + "_" + r["limb"]
d = prep(r, "series", "session.rpe")
out["Remmert2026_OSF_5wprn"] = run(d, ["perceived.recovery", "soreness", "mtt", "stress", "sleep.quality", "sleep.quantity", "nutrition", "session"], ["condition"], "session.rpe", "pid")
b = pd.read_excel(sys.argv[2], sheet_name="Subjective Scales").dropna(subset=["srpe"])
b["pid"] = b["condition"] + "_" + b["id"].astype(str); b["order"] = b["timepoint"]; b["day_in_week"] = ((b["timepoint"] - 1) % 3).astype(str)
d = prep(b, "pid", "srpe")
out["Robinson2025_OSF_58hy3"] = run(d, ["prs", "mtt", "timepoint"], ["condition", "day_in_week"], "srpe", "pid")
print(json.dumps(out, indent=1)); json.dump(out, open(sys.argv[3], "w"), indent=2)
