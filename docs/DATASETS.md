# Datasets and local files

## Sansar Sanskrit Corpus

The completed normalization output was generated from the pinned
`MuseMesh/Sansar Sanskrit Corpus` `default/train` split at revision
`a268a87223c38575883c7ee6af25b1d2e65fd0d7`. It contains 1,829,785 rows in
183 Parquet shards under `datasets/processed/sansar_normalized/`. The output
preserves the original record fields, stores the original text as `raw_text`,
and adds `normalized_text`.

This is the entire pinned training split used by the project, not a claim that
all source splits, versions, or related datasets are downloaded. The generated
data stays on the local machine and is ignored by Git.

Recreate it with:

```powershell
python modules\module_01_normalization\scripts\normalize_sansar_full_corpus.py
```

The pinned revision and resumable output manifest are configured/recorded by
that script in the local data directory. Check the upstream dataset card and
license/attribution conditions before using or sharing derived material.

## Sanskrit OCR post-correction benchmark

The research pilot used `acomquest/sanskrit-ocr-post-correction` at revision
`35de91d045dfd8c9bd875b44f2bfda52f09f72ce`. Files are stored locally under
`datasets/raw/sanskrit_ocr_post_correction/`; leakage-filtered working splits
are under `datasets/splits/ocr_target_disjoint/`. The local snapshot's counts
differ from the paper and current dataset card. Canonicalized targets
overlapped across original partitions, so the prepared split builder reserves
OOD rows first and removes exact target overlaps from lower-priority splits.
See [the full audit](../modules/module_01_normalization/docs/research/04_real_ocr_baseline_review.md).

The mirror declares MIT, but this does not establish that every underlying
book or image is redistributable. Keep benchmark data out of this repository;
retain attribution and review original source rights separately.

## Data policy for this repository

`datasets/`, `experiments/`, and `models/` are excluded from Git by default.
These directories contain downloaded corpus rows, text-bearing predictions,
large generated files, or binary checkpoints. Their README files explain how
to recreate them. Commit scripts, documentation, source revisions, and
aggregate results instead of raw text or generated datasets.
