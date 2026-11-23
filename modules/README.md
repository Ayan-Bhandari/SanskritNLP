# Module index

Each numbered module owns its implementation, scripts, notebooks, and
module-specific documentation. Shared environment setup, data policy, and
project conventions stay at the repository root.

## Current modules

| Module | Scope | Status |
|---|---|---|
| [Module 1: Text normalization](module_01_normalization/README.md) | Sanskrit Unicode/whitespace normalization, corpus preparation, and OCR correction research | Normalization complete for the pinned Sansar `default/train` split; OCR model pilot unsuccessful |
| [Module 2: Tokenization](module_02_tokenization/README.md) | Compare character, whitespace, and subword token units for Sanskrit | Data audit and whitespace/code-point/SentencePiece baselines complete; CPU only so far |
| [Module 3: Word segmentation](module_03_word_segmentation/README.md) | Restore Sanskrit word boundaries and sandhi-aware segmentation | Hackathon DCS audit, full character Transformer baseline, inference script, and held-out evaluation complete |

## Adding a module

For the next planned module, create a sibling directory such as
`module_03_word_segmentation/` and use this structure:

```text
module_02_topic_name/
  README.md
  src/
  scripts/
  notebooks/
  docs/
```

Document the objective, input/output contract, data sources, reproduction
steps, evaluation, and limitations in that module's README. Keep shared
dependencies in the root `requirements.txt` unless a module-specific
environment becomes necessary. Store downloaded datasets, generated output,
and checkpoints in the shared local-only root directories described in
`docs/DATASETS.md`.
