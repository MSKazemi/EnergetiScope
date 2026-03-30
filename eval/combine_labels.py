#!/usr/bin/env python3
"""
Combine all per-run label Parquet files from bench collection into one file.

Usage:
    python eval/combine_labels.py --input-dir data/bench/labels --out data/bench/labels.parquet
"""
import argparse
import glob
import os
import pandas as pd

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.input_dir, "*.parquet")))
    if not files:
        raise SystemExit(f"[ERROR] No parquet files found in {args.input_dir}")

    frames = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            df["source_file"] = os.path.basename(f)
            frames.append(df)
        except Exception as e:
            print(f"[WARN] skipping {f}: {e}")

    combined = pd.concat(frames, ignore_index=True)

    # Normalize energy column: bench collection uses "total_energy_j" (job mode),
    # while window mode uses "energy_step_j". Support both.
    if "total_energy_j" in combined.columns and "energy_step_j" not in combined.columns:
        combined["energy_step_j"] = combined["total_energy_j"]

    # Keep only rows with valid energy
    valid = combined[combined["energy_step_j"].notna() & (combined["energy_step_j"] > 0)]
    print(f"[OK] Combined {len(files)} files → {len(combined)} rows ({len(valid)} with valid energy_step_j)")
    print(f"     Workloads: {valid['workload_name'].nunique()} unique")
    print(f"     Kinds:     {valid['workload_kind'].value_counts().to_dict()}")

    combined.to_parquet(args.out, index=False)
    print(f"[OK] Written to {args.out}")

if __name__ == "__main__":
    main()
