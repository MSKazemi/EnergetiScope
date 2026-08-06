#!/usr/bin/env python3
"""Reviewer-requested stronger baseline + overhead numbers.

Runs on the EXISTING benchmark data (data/bench/train_rows_filtered.parquet).
No new data collection. Same GroupKFold protocol as train_power.py.
"""
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import (GradientBoostingRegressor,
                              HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsRegressor

TRAIN = "/home/mohsen/scratch/none-project/EnergetiScope/data/bench/train_rows_filtered.parquet"
TARGET = "energy_step_j"

df = pd.read_parquet(TRAIN)
X = np.stack(df["features"].to_numpy())
y = df[TARGET].astype(float).to_numpy()
keep = np.isfinite(y)
X, y, df = X[keep], y[keep], df.loc[keep]
groups = df["workload_name"].astype(str).to_numpy()

print(f"n={len(y)}  feat_dim={X.shape[1]}  groups={df['workload_name'].nunique()}")
print(f"target {TARGET}: mean={y.mean():.1f} J  min={y.min():.1f}  max={y.max():.1f}")

# how many feature columns are actually informative (non-constant)?
nonconst = int((X.std(axis=0) > 0).sum())
print(f"non-constant feature columns: {nonconst} / {X.shape[1]}")

n_splits = min(5, df["workload_name"].nunique())
gkf = GroupKFold(n_splits=n_splits)


def mape(yt, yp):
    m = yt != 0
    return float(np.mean(np.abs((yt[m] - yp[m]) / yt[m])) * 100)


def evaluate(name, make_model):
    maes, r2s, mapes, fit_t = [], [], [], []
    for tr, va in gkf.split(X, y, groups):
        model = make_model()
        t0 = time.perf_counter()
        model.fit(X[tr], y[tr])
        fit_t.append(time.perf_counter() - t0)
        p = model.predict(X[va])
        maes.append(mean_absolute_error(y[va], p))
        r2s.append(r2_score(y[va], p))
        mapes.append(mape(y[va], p))
    print(f"{name:<28} MAE {np.mean(maes):7.1f} +/- {np.std(maes):6.1f} J | "
          f"R2 {np.mean(r2s):6.3f} +/- {np.std(r2s):5.3f} | "
          f"MAPE {np.mean(mapes):5.2f}% | fit {np.mean(fit_t)*1000:6.1f} ms")
    return dict(mae=np.mean(maes), mae_sd=np.std(maes), r2=np.mean(r2s),
                r2_sd=np.std(r2s), mape=np.mean(mapes))


print("\n--- GroupKFold CV (workload_name as group) ---")
res = {}
res["knn"] = evaluate("KNN (k=5, cosine)", lambda: KNeighborsRegressor(n_neighbors=5, metric="cosine"))
res["lr"] = evaluate("Linear Regression (OLS)", LinearRegression)
res["gbt"] = evaluate("Gradient Boosted Trees", lambda: GradientBoostingRegressor(random_state=0))
res["hgb"] = evaluate("HistGradientBoosting", lambda: HistGradientBoostingRegressor(random_state=0))
res["rf"] = evaluate("Random Forest", lambda: RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1))

# ---------------- overhead ----------------
print("\n--- Inference overhead (KNN, full training set) ---")
model = KNeighborsRegressor(n_neighbors=5, metric="cosine").fit(X, y)
one = X[:1]
for _ in range(50):
    model.predict(one)
lat = []
for _ in range(1000):
    t0 = time.perf_counter()
    model.predict(one)
    lat.append((time.perf_counter() - t0) * 1000)
lat = np.array(lat)
print(f"single-query latency: mean {lat.mean():.3f} ms | p50 {np.percentile(lat,50):.3f} | "
      f"p95 {np.percentile(lat,95):.3f} | p99 {np.percentile(lat,99):.3f} ms")

t0 = time.perf_counter()
model.predict(X)
batch = (time.perf_counter() - t0) * 1000
print(f"batch of {len(X)}: {batch:.1f} ms  ({batch/len(X):.3f} ms/query)")

t0 = time.perf_counter()
KNeighborsRegressor(n_neighbors=5, metric="cosine").fit(X, y)
print(f"full training time: {(time.perf_counter()-t0)*1000:.1f} ms")

import os
p = "/home/mohsen/scratch/none-project/EnergetiScope/app/artifacts"
for f in ("knn_energy.joblib", "encoder.joblib"):
    print(f"artifact {f}: {os.path.getsize(os.path.join(p,f))/1024:.1f} KiB")
