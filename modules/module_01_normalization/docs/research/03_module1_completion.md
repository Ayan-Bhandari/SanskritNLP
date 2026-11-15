# Module 1 completion: Sanskrit text normalization baseline

## Deliverables

- The deterministic normalizer is documented at
  [`src/normalization/README.md`](../../src/normalization/README.md) and
  implemented in `src/normalization/text_normalizer.py`.
- The initial 5,000-row sample and the later 5,000-row source-aware sample
  support inspection and grouped evaluation. They were not the full-corpus
  normalization output.
- A source/work-group split contains 3,997 train, 500 validation, and 500
  test rows. Exact normalized text and groups do not cross partitions.
- The synthetic formatting and generic character-edit baseline pilot is
  documented in
  [`02_synthetic_noise_protocol.md`](02_synthetic_noise_protocol.md).
- The entire pinned Sansar `default/train` split is normalized under
  `datasets/processed/sansar_normalized/`.

## Full-corpus output

The streaming job completed at revision
`a268a87223c38575883c7ee6af25b1d2e65fd0d7` with 1,829,785 records in 183
Zstandard-compressed Parquet shards. The first 182 shards contain 10,000 rows
each; the final shard contains 9,785. Parquet metadata was summed and matched
the manifest's `records_written` count. The schema preserves all source
columns, renames the original `text` field to `raw_text`, and adds
`normalized_text`. Thus the original strings remain available for comparison
and future correction work. The manifest records dataset revision, package
versions, schema, status, and shard count.

The data is intentionally excluded from Git; it remains local in the processed
directory. If regenerated, use
`modules/module_01_normalization/scripts/normalize_sansar_full_corpus.py` and the pinned revision in its
configuration.

## What the pilot established

The deterministic rules consistently clean Unicode and horizontal spacing
while preserving punctuation and line breaks. On held-out synthetic formatting
variants, the rule baseline restored exact text for all 500 test examples. On
generic akshara-like insertions, deletions, and substitutions, it did not
reduce CER or improve exact match. This is expected for a formatter.

A Sanskrit OCR benchmark was downloaded and audited. Its split files contain
target overlap after canonical whitespace/Unicode cleanup, so filtered,
target-disjoint partitions were prepared before training. A 10,000-example,
10-epoch character Transformer pilot completed, but its decoded CER was much
worse than identity and deterministic baselines on validation, test, and OOD.
It is an unsuccessful research pilot, not a usable correction model. See
[04_real_ocr_baseline_review.md](04_real_ocr_baseline_review.md) and
[05_character_transformer_pilot.md](05_character_transformer_pilot.md).

## Module 1 status and limits

The deterministic normalization and full pinned Sansar train-split output are
complete. The research extension has a documented real OCR baseline, leakage
audit, filtered splits, training script, and completed Transformer pilot. That
pilot failed to produce useful correction quality. It is not an accepted
post-correction system. GPU-backed training and a better-designed model remain
future research work; the current environment used CPU-only PyTorch.

The downloaded Sansar data is the pinned `default/train` split, not a claim
that every split or version of the upstream corpus is present locally.
