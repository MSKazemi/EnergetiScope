#!/usr/bin/env python3
"""On the EXACT paper dataset (n=711, 10 groups): which protocol yields
MAE=240 J, R2=0.979, MAPE=2.7%?"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.neighbors import KNeighborsRegressor

ROOT = "/home/mohsen/scratch/none-project/EnergetiScope"
T = "energy_step_j"
f = pd.read_parquet(f"{ROOT}/data/bench/features.parquet")
l = pd.read_parquet(f"{ROOT}/data/bench/labels.parquet")
df = f.merge(l, on=["namespace", "workload_kind", "workload_name"], how="inner")
df = df[df[T].notna() & np.isfinite(df[T]) & (df[T] > 0)]
df = df.groupby("workload_name").filter(lambda x: len(x) >= 5)   # -> n=711, 10 groups

X = np.stack(df["features"].to_numpy())
y = df[T].astype(float).to_numpy()
g = df["workload_name"].astype(str).to_numpy()
print(f"PAPER DATASET: n={len(y)}  groups={df['workload_name'].nunique()}  "
      f"y mean={y.mean():.1f} J  std={y.std():.1f}")


def mape(yt, yp, eps=1e-6):
    m = np.abs(yt) > eps
    return 100.0 * np.mean(np.abs((yt[m] - yp[m]) / yt[m]))


def report(label, y_true, y_pred, foldr2=None):
    fr = f"{foldr2:>9.3f}" if foldr2 is not None else "      n/a"
    hit = "  <== PAPER" if (abs(mean_absolute_error(y_true, y_pred) - 240) < 40
                            and abs(r2_score(y_true, y_pred) - 0.979) < 0.02) else ""
    print(f"{label:<44} MAE={mean_absolute_error(y_true,y_pred):>8.1f} "
          f"pooledR2={r2_score(y_true,y_pred):>7.3f} meanfoldR2={fr} "
          f"MAPE={mape(y_true,y_pred):>6.2f}%{hit}")


def cv(label, splitter, factory, grouped):
    oof = np.zeros_like(y)
    r2s = []
    it = splitter.split(X, y, g) if grouped else splitter.split(X, y)
    for tr, va in it:
        m = factory().fit(X[tr], y[tr])
        p = m.predict(X[va])
        oof[va] = p
        r2s.append(r2_score(y[va], p))
    report(label, y, oof, np.mean(r2s))


knn = lambda: KNeighborsRegressor(n_neighbors=5, metric="cosine")
print("\n--- KNN k=5 cosine, various protocols ---")
cv("GroupKFold (protocol claimed in paper)", GroupKFold(n_splits=5), knn, True)
cv("KFold shuffle=True (LEAKY)", KFold(5, shuffle=True, random_state=42), knn, False)
cv("KFold shuffle=False (LEAKY)", KFold(5, shuffle=False), knn, False)

m = knn().fit(X, y)
report("IN-SAMPLE fit (train==test, no CV)", y, m.predict(X))

print("\n--- Reviewer-requested baselines, GroupKFold (honest protocol) ---")
for name, fac in [("Linear Regression (OLS)", LinearRegression),
                  ("Gradient Boosted Trees", lambda: GradientBoostingRegressor(random_state=0)),
                  ("Random Forest", lambda: RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1))]:
    cv(name, GroupKFold(n_splits=5), fac, True)
