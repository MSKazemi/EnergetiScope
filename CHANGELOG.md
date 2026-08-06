# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Apache-2.0 `LICENSE` and `NOTICE`.
- Community health files: `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`,
  `SUPPORT.md`, `CITATION.cff`, issue forms, and a pull-request template.
- `AGENTS.md` describing the repository for coding agents.
- `ci` GitHub Actions workflow: lint, tests on Python 3.11/3.12, a check that the README
  quickstart still reproduces the benchmark table, and a syntax check over every shipped
  module.
- `eval/eval_bench_grouped.py`, `eval/eval_overhead_and_external.py`, and `eval/repro_*.py`
  reproduction scripts.
- `--model {gbt,rf,linear,knn}` on `app/train_power.py`; gradient-boosted trees are now the
  default.

### Fixed
- **`app/predict_k8s.py` was not valid Python.** A stray line of pasted text after `main()`
  meant the batch-prediction CLI failed to parse and could never have run.
- **`make test` did not work on a clean clone.** `uv run pytest` did not place the repository
  root on `sys.path`, so the test module's imports failed; fixed with
  `[tool.pytest.ini_options] pythonpath`.
- **Cross-validation leaked across folds.** `eval/eval_all.py::cv5()` used ungrouped shuffled
  `KFold`, so repeated per-interval rows of the same workload appeared on both sides of a
  split; it now uses `GroupKFold` with metrics computed on pooled out-of-fold predictions.
- `eval/eval_all.py` no longer allocates an unused `results` list in `run_dataset_a()`.
- Docker workflow pushed to an image named after a previous project (`k8spodpcp`).

### Changed
- README rewritten: leads with the result, quotes only reproducible numbers, documents the
  working `uvicorn` invocation and the real `InferenceRequest` schema, and states the
  project's limitations explicitly.
- Ruff configured with per-file ignores that reflect deliberate patterns (evaluation scripts
  call `warnings.filterwarnings()` before importing the scientific stack) rather than
  suppressing findings wholesale.

### Known issues
- `join_features_labels.py` over-joins: its `train_rows*.parquet` outputs contain many-to-many
  duplicate rows (14,906 rows where the correct inner merge yields 1,422), and models trained
  on them score worse than a mean predictor. Use `eval/eval_bench_grouped.py` meanwhile.
