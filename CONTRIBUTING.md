# Contributing to EnergetiScope

Thanks for considering a contribution. This is a research project, so the most valuable
contributions are probably not the ones you'd expect — see [What helps most](#what-helps-most).

## Setup — clone to green tests

```bash
git clone https://github.com/MSKazemi/EnergetiScope.git
cd EnergetiScope

# uv is the supported workflow (https://docs.astral.sh/uv/)
make install-dev        # uv sync --extra dev

make test               # uv run pytest        -> 8 passed
make lint               # uv run ruff check .  -> All checks passed!
```

Python **3.11 or 3.12**. CI runs on both. If `make test` fails on a clean clone, that is a
bug — please open an issue, it is not your setup.

Verify the headline claim while you're here:

```bash
python eval/eval_bench_grouped.py     # ~1 minute, prints the benchmark table
```

## What helps most

Ranked by how much it actually moves the project:

1. **Energy measurements from hardware that isn't ours.** Every number in this repo comes
   from a single bare-metal node. A second machine — different CPU, different vendor, ARM,
   a different Kepler version — is worth more than any code change. See
   [`COLLECTION_RUNBOOK.md`](COLLECTION_RUNBOOK.md) and open an issue with the
   `hardware-measurement` template.
2. **Reproduction reports.** Run the quickstart and tell us whether you got our numbers.
   A "this didn't reproduce" issue is a gift, not a complaint.
3. **Bug fixes**, especially in the pipeline stages (`app/`).
4. **Documentation** that fixes something that confused you. If you were confused, the docs
   were wrong.
5. **New workload manifests** for `manifests/bench/` that broaden the benchmark beyond
   stress-ng and the five application images currently covered.

## Before you write code

- **Open an issue first** for anything that changes the feature schema, the model interface,
  the API request/response shape, or adds a dependency. These are hard to un-decide, and I'd
  rather discuss than reject your work.
- **No issue needed** for typo fixes, doc corrections, added tests, or anything labelled
  [`good first issue`](https://github.com/MSKazemi/EnergetiScope/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).
- Comment on the issue to claim it, so two people don't do the same work.

## Pull requests

- **Keep them under ~300 changed lines.** Split larger work into a sequence — reviewing 1,000
  lines of research code well is not something I can do in one sitting, so big PRs sit longer.
- **Link the issue** the PR closes.
- **Add or update a test** for behaviour changes. Tests live in `app/test_*.py`.
- **Run `make test` and `make lint` before pushing** — the same commands CI runs.
- **Don't reformat unrelated code.** A diff that is 90% whitespace hides the 10% that matters.
- **If you change a number that appears in the README**, update the README in the same PR.
  Documented numbers must be reproducible by the command shown next to them.

Commit messages follow `type(scope): summary` — e.g. `fix(encode): handle missing memory
limits`. Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`.

## Notes for coding agents (and anyone new to the codebase)

Four things will trip you up, and none are guessable from the code:

1. **`app/` modules import each other as top-level modules** (`from k8s_encode import ...`),
   while the tests import them as `app.<module>`. So running the service needs
   `PYTHONPATH=app uvicorn predict_service:app` — `uvicorn app.predict_service:app`
   **does not work**.
2. **`join_features_labels.py` over-joins.** Its `train_rows*.parquet` outputs contain
   many-to-many duplicates (14,906 rows where the correct inner merge gives 1,422), and
   models trained on them score *worse than a mean predictor*. Don't use them to judge model
   quality, and don't "fix" a model that is really being starved by this join. The correct
   merge is in `eval/eval_bench_grouped.py`.
3. **`avg_power_w` is entirely NaN** in the bundled `train_rows*` files. Use `energy_step_j`
   or `total_energy_j`.
4. **Never evaluate with an ungrouped split** — see the rule below.

## Scientific-accuracy rules

This code backs an academic paper, so a few rules are stricter than usual:

- **Never commit a metric you haven't reproduced.** State the command that produces it.
- **Never change an evaluation protocol and keep the old numbers.** If you change how a
  split, fold, or metric is computed, regenerate every number that depended on it.
- **Grouped splits are not optional.** Repeated measurements of one workload must never
  appear on both sides of a train/test split; use `GroupKFold` on `workload_name`. An
  ungrouped split reports memorization as accuracy — this has bitten this project before.
- **State hardware and versions** with any measurement you report.

## Response times — the honest version

This is maintained by one person alongside a research job. Realistically:

- **First response: within about a week.** Often faster; occasionally slower during
  conference deadlines.
- If something has sat for **two weeks**, please bump it. That is not rude, it means it fell
  through and I'd like the reminder.

If you'd rather ask a question than file a bug, use
[Discussions](https://github.com/MSKazemi/EnergetiScope/discussions) — Issues are kept for
actionable work.

## Repeat contributors

If you land a few PRs and want more responsibility, say so. Triage rights and review rights
are given to people who show up consistently; there's no committee. Contributions of all
kinds — measurements, docs, reviews, reproduction reports — count.

## License

Contributions are licensed under [Apache-2.0](LICENSE), matching the project.
