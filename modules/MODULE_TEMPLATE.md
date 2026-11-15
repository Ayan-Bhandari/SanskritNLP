# Module N: <name>

## Goal

Describe the user or research problem this module addresses and how it builds
on prior modules.

## Inputs and outputs

- Inputs and their source/license:
- Output format and location:
- What raw data is preserved:

## Structure

```text
src/       Module implementation
scripts/   Reproducible preparation, training, and evaluation commands
notebooks/ Exploratory analysis only
docs/      Method, decisions, results, and limitations
```

## Setup and reproduction

List commands from the repository root, required dependencies, dataset
revision, and expected local output paths. Keep large data and checkpoints in
the shared local-only `datasets/`, `experiments/`, and `models/` directories.

## Evaluation and status

Report baselines, split definitions, metrics, and qualitative review. State
clearly what has and has not been validated, known risks, and the next
concrete step.
