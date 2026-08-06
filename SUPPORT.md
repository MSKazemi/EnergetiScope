# Getting help

## Where to go

| I want to… | Go to |
|---|---|
| Ask how something works, or why a number looks odd | [Discussions → Q&A](https://github.com/MSKazemi/EnergetiScope/discussions/categories/q-a) |
| Report something broken | [Issues → Bug report](https://github.com/MSKazemi/EnergetiScope/issues/new/choose) |
| Share energy measurements from your hardware | [Issues → Hardware measurement](https://github.com/MSKazemi/EnergetiScope/issues/new/choose) |
| Propose a feature or change | [Discussions → Ideas](https://github.com/MSKazemi/EnergetiScope/discussions/categories/ideas) first, then an issue |
| Report a security problem | **Not** an issue — see [`SECURITY.md`](SECURITY.md) |

Issues are kept for actionable work, so questions get moved to Discussions. That is not a
brush-off — it keeps the issue tracker honest about what still needs doing.

## Before asking

Please include:

- What you ran (the exact command).
- What you expected, and what happened instead.
- `python --version` and your OS.
- Whether `make test` passes on your clone.

For anything involving energy numbers, also include your CPU model, whether RAPL is
available, and your Kepler version — energy results are meaningless without the hardware
context.

## Response time

One maintainer, working on this alongside a research job. Expect a first response **within
about a week**. If two weeks pass, please bump the thread — it fell through the cracks.

## Things that are known and don't need a report

- **Kepler reports zero energy on cloud VMs.** Most cloud instances don't expose RAPL MSRs.
  This is a Kepler/hardware limitation, not an EnergetiScope bug.
- **`train_rows*.parquet` produce poor models.** `join_features_labels.py` currently
  over-joins; use `eval/eval_bench_grouped.py`. Tracked in the issue list.
- **The bundled model doesn't predict your hardware well.** Expected — retrain on your own
  cluster. Energy is hardware-specific.
