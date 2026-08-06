#!/usr/bin/env python3
"""Serving overhead (Reviewer 1) + honest re-run of the external check.

Overhead is measured on the artifacts the service actually loads.
The external dataset contains a single workload, so no group-aware split
is possible; we report both a shuffled and a contiguous (temporal) split
so the leakage-sensitivity of that number is visible rather than hidden.
"""
import os
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def mape(yt, yp, eps=1e-6):
    m = np.abs(yt) > eps
    return 100.0 * np.mean(np.abs((yt[m] - yp[m]) / yt[m]))


# ---------------------------------------------------------------- external
print("=" * 70)
print("External dataset (Zenodo:14332659) - single workload, no grouping possible")
print("=" * 70)
files = [f"{ROOT}/data/external/zenodo_14332659/data/"
         f"_{ts}-BootstrapInitializer-containerUtilization.csv"
         for ts in ["1729087088", "1729099594", "1729113884"]]
frames = [pd.read_csv(f) for f in files if os.path.exists(f)]
df = pd.concat(frames, ignore_index=True).rename(
    columns={"cpuSeconds": "cpu_seconds", "powerWatts": "power_w"})
df = df.dropna(subset=["cpu_seconds", "power_w"])
df = df[(df["cpu_seconds"] > 0) & (df["power_w"] > 0)]
Xe = StandardScaler().fit_transform(df[["cpu_seconds"]].values)
ye = df["power_w"].values
print(f"n={len(ye)}  mean power={ye.mean():.1f} W")

for label, splitter in [("KFold shuffle=True  (leaky in time)", KFold(5, shuffle=True, random_state=42)),
                        ("KFold shuffle=False (contiguous)", KFold(5, shuffle=False))]:
    oof = np.zeros_like(ye)
    for tr, va in splitter.split(Xe):
        m = KNeighborsRegressor(n_neighbors=5).fit(Xe[tr], ye[tr])
        oof[va] = m.predict(Xe[va])
    print(f"  KNN {label:<38} MAE={mean_absolute_error(ye,oof):6.2f} W  "
          f"R2={r2_score(ye,oof):6.3f}  MAPE={mape(ye,oof):5.2f}%")

# ---------------------------------------------------------------- overhead
print("\n" + "=" * 70)
print("Serving overhead on the benchmark training set")
print("=" * 70)
f = pd.read_parquet(f"{ROOT}/data/bench/features.parquet")
l = pd.read_parquet(f"{ROOT}/data/bench/labels.parquet")
d = f.merge(l, on=["namespace", "workload_kind", "workload_name"], how="inner")
d = d[d["energy_step_j"].notna() & np.isfinite(d["energy_step_j"]) & (d["energy_step_j"] > 0)]
d = d.groupby("workload_name").filter(lambda x: len(x) >= 5)
X = np.stack(d["features"].to_numpy())
y = d["energy_step_j"].astype(float).to_numpy()

for name, factory in [("KNN (k=5, cosine)", lambda: KNeighborsRegressor(n_neighbors=5, metric="cosine")),
                      ("Gradient Boosted Trees", lambda: GradientBoostingRegressor(random_state=0))]:
    t0 = time.perf_counter()
    model = factory().fit(X, y)
    fit_ms = (time.perf_counter() - t0) * 1000

    one = X[:1]
    for _ in range(50):
        model.predict(one)
    lat = []
    for _ in range(2000):
        t0 = time.perf_counter()
        model.predict(one)
        lat.append((time.perf_counter() - t0) * 1000)
    lat = np.array(lat)
    import io
    import joblib
    buf = io.BytesIO()
    joblib.dump(model, buf)
    print(f"{name:<24} train={fit_ms:7.1f} ms | single-query p50={np.percentile(lat,50):.2f} ms "
          f"p95={np.percentile(lat,95):.2f} ms p99={np.percentile(lat,99):.2f} ms | "
          f"artifact={len(buf.getvalue())/1024:.0f} KiB")

enc = f"{ROOT}/app/artifacts/encoder.joblib"
if os.path.exists(enc):
    print(f"encoder artifact: {os.path.getsize(enc)/1024:.1f} KiB")
