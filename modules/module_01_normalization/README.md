# Module 1: Text normalization

## Goal

Normalize Sanskrit text representation consistently without losing the source
string, verse line breaks, punctuation, or corpus metadata. A separate
research extension evaluates post-OCR correction; it has not yielded a usable
learned model.

## Contents

```text
src/normalization/   Deterministic NFC and whitespace normalizer
scripts/             Corpus preparation, baseline evaluation, and pilot code
notebooks/           Bounded exploratory Sansar analysis
docs/research/       Dataset audit, protocols, results, and limits
```

## Run

From the repository root after following `docs/SETUP.md`:

```powershell
python modules\module_01_normalization\scripts\normalization_examples.py
```

To regenerate the full pinned Sansar `default/train` split, use
`scripts/normalize_sansar_full_corpus.py`. It downloads and processes the
complete split, writes local Parquet shards to `datasets/processed/`, and may
require significant disk space and time. Read `docs/DATASETS.md` first.

## Status and results

The normalizer is complete and the pinned Sansar train split was processed into
1,829,785 rows. The original text is retained as `raw_text` alongside
`normalized_text`. This does not mean every upstream corpus split/version was
downloaded.

The OCR benchmark split audit and bounded Transformer pilot are documented in
[`docs/research/04_real_ocr_baseline_review.md`](docs/research/04_real_ocr_baseline_review.md)
and [`docs/research/05_character_transformer_pilot.md`](docs/research/05_character_transformer_pilot.md).
The pilot model was substantially worse than the identity baseline and should
not be used to correct OCR text.
