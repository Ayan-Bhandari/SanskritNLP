# Contributing

Thanks for helping with the Sanskrit NLP Toolkit. The repository is organized
as numbered, self-contained modules. Start from `modules/README.md` and work
inside the module relevant to your contribution.

## Workflow

1. Create a feature branch from the team's shared default branch.
2. Keep changes focused and preserve reproducibility: record dataset revision,
   random seeds, package versions, and output locations for experiments.
3. Keep raw input alongside derived output when working locally. Never
   overwrite original corpus records with normalized or corrected text.
4. Update the module's README and docs when behavior or results change. Put
   project-wide setup and data policy in root `docs/`.
5. Before opening a pull request, review `git status` and `git diff` to ensure
   local data, model binaries, credentials, and generated files are absent.

## Data and model artifacts

Large datasets, local experiment files, and checkpoints are intentionally
ignored by Git. Do not force-add them. Share reproducible download and
preparation instructions, source revisions, and aggregate metrics instead.
Check source terms and attribution before using or redistributing any data.

## Code and evaluation

- Keep normalization deterministic and preserve raw text.
- Separate formatting cleanup from OCR correction and linguistic analysis.
- Report CER and exact match with split definitions and baselines.
- Keep held-out evaluation data out of model tuning.
- Label synthetic examples and results as synthetic.

There is no automated test suite configured yet. For Module 1 normalizer
changes, run `modules/module_01_normalization/scripts/normalization_examples.py`
and add focused checks as part of the proposed change.
