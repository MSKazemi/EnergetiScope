#!/usr/bin/env python3
"""Diagnose the gap between committed data and the paper's reported numbers."""
import re

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsRegressor

ROOT = "/home/mohsen/scratch/none-project/EnergetiScope"
f = pd.read_parquet(f"{ROOT}/data/bench/features.parquet")
l = pd.read_parquet(f"{ROOT}/data/bench/labels.parquet")
print("features cols:", f.columns.tolist())
print("labels cols  :", l.columns.tolist())
print("features rows:", len(f), " labels rows:", len(l))

df = f.merge(l, on=["namespace", "workload_kind", "workload_name"], how="inner")
print("\nmerged rows:", len(df))
for c in ["energy_step_j", "avg_power_w", "total_energy_j"]:
    if c in df.columns:
        v = df[c]
        print(f"  {c}: valid>0 = {int((v.notna() & np.isfinite(v) & (v>0)).sum())}, "
              f"mean={v.mean():.1f}")

t = "energy_step_j"
df = df[df[t].notna() & np.isfinite(df[t]) & (df[t] > 0)]

print("\nrows per workload group:")
vc = df["workload_name"].value_counts()
for k, v in vc.items():
    print(f"  {k:<45} {v}")


def norm(n):
    """Collapse CronJob run instances to their parent CronJob."""
    return re.sub(r"-\d{6,}$", "", n)


df["grp_norm"] = df["workload_name"].astype(str).map(norm)
print(f"\ngroups raw={df['workload_name'].nunique()}  normalized={df['grp_norm'].nunique()}")


def run(label, d, gcol):
    X = np.stack(d["features"].to_numpy())
    y = d[t].astype(float).to_numpy()
    g = d[gcol].astype(str).to_numpy()
    ng = d[gcol].nunique()
    if ng < 2:
        return
    gkf = GroupKFold(n_splits=min(5, ng))
    maes, r2s, oof = [], [], np.zeros_like(y)
    for tr, va in gkf.split(X, y, g):
        m = KNeighborsRegressor(n_neighbors=5, metric="cosine").fit(X[tr], y[tr])
        p = m.predict(X[va])
        oof[va] = p
        maes.append(mean_absolute_error(y[va], p))
        r2s.append(r2_score(y[va], p))
    print(f"{label:<46} n={len(y):>4} grp={ng:>3} "
          f"MAE={np.mean(maes):>8.1f} meanfoldR2={np.mean(r2s):>10.3f} "
          f"pooledOOF_R2={r2_score(y, oof):>7.3f}")


print()
run("raw workload_name", df, "workload_name")
run("normalized cronjob groups", df, "grp_norm")

# drop singleton groups (cronjob one-off rows)
big = df.groupby("workload_name").filter(lambda x: len(x) >= 5)
run("drop groups with <5 rows", big, "workload_name")
big2 = df.groupby("grp_norm").filter(lambda x: len(x) >= 5)
run("normalized + drop <5 rows", big2, "grp_norm")
