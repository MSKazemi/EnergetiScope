#!/usr/bin/env python3
"""Try to reproduce the paper's headline numbers: n=711, 10 groups,
MAE=240 J, R2=0.979, MAPE=2.7%."""
import itertools
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.neighbors import KNeighborsRegressor

warnings.filterwarnings("ignore")
ROOT = "/home/mohsen/scratch/none-project/EnergetiScope"


def mape(yt, yp, eps=1e-6):
    m = np.abs(yt) > eps
    return 100.0 * np.mean(np.abs((yt[m] - yp[m]) / yt[m]))


def load_merged(target):
    f = pd.read_parquet(f"{ROOT}/data/bench/features.parquet")
    l = pd.read_parquet(f"{ROOT}/data/bench/labels.parquet")
    df = f.merge(l, on=["namespace", "workload_kind", "workload_name"], how="inner")
    return df[df[target].notna() & np.isfinite(df[target]) & (df[target] > 0)]


def load_parquet(name, target):
    df = pd.read_parquet(f"{ROOT}/data/bench/{name}")
    return df[df[target].notna() & np.isfinite(df[target]) & (df[target] > 0)]


SOURCES = {
    "merged(features+labels)": load_merged,
    "train_rows.parquet": lambda t: load_parquet("train_rows.parquet", t),
    "train_rows_filtered.parquet": lambda t: load_parquet("train_rows_filtered.parquet", t),
}

print(f"{'source':<30} {'target':<16} {'protocol':<20} {'n':>5} {'grp':>4} "
      f"{'MAE':>9} {'R2':>9} {'MAPE':>7}")
print("-" * 108)

for sname, loader, target in itertools.product(
        SOURCES, [None], ["energy_step_j", "avg_power_w", "total_energy_j"]):
    try:
        df = SOURCES[sname](target)
    except Exception as e:
        continue
    if len(df) < 10:
        continue
    X = np.stack(df["features"].to_numpy())
    y = df[target].astype(float).to_numpy()
    groups = df["workload_name"].astype(str).to_numpy()
    ngrp = df["workload_name"].nunique()

    protocols = {
        "KFold shuffle (leaky)": KFold(n_splits=5, shuffle=True, random_state=42).split(X, y),
        "KFold no-shuffle": KFold(n_splits=5, shuffle=False).split(X, y),
    }
    if ngrp >= 2:
        protocols["GroupKFold"] = GroupKFold(n_splits=min(5, ngrp)).split(X, y, groups)

    for pname, splitter in protocols.items():
        maes, r2s, mapes = [], [], []
        for tr, va in splitter:
            m = KNeighborsRegressor(n_neighbors=5, metric="cosine").fit(X[tr], y[tr])
            p = m.predict(X[va])
            maes.append(mean_absolute_error(y[va], p))
            r2s.append(r2_score(y[va], p))
            mapes.append(mape(y[va], p))
        flag = ""
        if abs(np.mean(maes) - 240) < 30 and abs(np.mean(r2s) - 0.979) < 0.02:
            flag = "  <== MATCHES PAPER"
        print(f"{sname:<30} {target:<16} {pname:<20} {len(y):>5} {ngrp:>4} "
              f"{np.mean(maes):>9.1f} {np.mean(r2s):>9.3f} {np.mean(mapes):>6.2f}%{flag}")
