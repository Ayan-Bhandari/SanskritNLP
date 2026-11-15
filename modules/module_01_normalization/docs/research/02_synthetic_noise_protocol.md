# Module 1: Synthetic-noise experiment protocol

## Status

The bounded source-aware sample, work-group split, synthetic baseline pilot,
full training-split normalization, real OCR baseline and leakage audit, and a
bounded learned-model pilot have been completed. The learned model did not
improve on identity or deterministic baselines. The clean Sansar sample is not
OCR ground truth.

The full `default/train` split at pinned revision
`a268a87223c38575883c7ee6af25b1d2e65fd0d7` was normalized in 10,000-row
Parquet shards. The run wrote 1,829,785 records in 183 shards, preserving
`raw_text` and all input metadata beside `normalized_text`. Its resumable
manifest is `datasets/processed/sansar_normalized/manifest.json`; the data
directory is excluded from Git because of its size. Parquet metadata was
checked after completion: the shards sum to 1,829,785 rows, with 10,000 rows
in each of the first 182 shards and 9,785 in the last.

The 5,000-row files are analysis/evaluation samples, not the full training
corpus. Full-corpus normalization has now run; the result is available under
`datasets/processed/sansar_normalized/`.

## Source-aware sample and limits

The initial first 5,000-row sample contains one `source` value (`itihasa`), no
non-missing `work` values, and one licence value (`Apache-2.0`). It cannot
support a source-held-out or work-held-out evaluation.

A bounded source-aware sample has now been created with
[`sample_source_diverse_sansar.py`](../../scripts/sample_source_diverse_sansar.py).
It used the pinned dataset revision, a 10,000-record shuffle buffer, and seed
42. The scan stopped after 18,969 yielded rows once the 5,000-row sample cap
was reached; each source/work group contributed at most 300 rows. The sample
is saved at
`datasets/processed/sansar_source_aware_sample.parquet`.

The sample has 207 source/work groups: 206 non-missing `work` values under
`vagartha` (4,700 rows), and one source-level `sangraha` group (300 rows).
All 5,000 rows have `CC-BY-4.0` licence metadata. This is enough for a
work-held-out split, but only two sources are present, so a source-held-out
result would not be a useful cross-source evaluation.

There were three duplicate normalized texts, each repeated across two
source/work groups. The split candidate retained the first occurrence of each
text and recorded the omitted and representative IDs in a duplicate ledger.
The original source-aware sample remains unchanged. This left 4,997 unique
text rows for partitioning.

The split was created by
[`create_sansar_group_splits.py`](../../scripts/create_sansar_group_splits.py).
It assigns complete source/work groups with seed `42` and targets 80/10/10 by
row count. The result is 3,997 train rows, 500 validation rows, and 500 test
rows across 170, 18, and 19 groups respectively. All 300 `sangraha` rows are
in train; validation and test contain `vagartha` works. No group or normalized
text crosses partitions. This is a work-held-out split, not a source-held-out
evaluation.

Outputs are `datasets/splits/sansar_source_aware_group_splits.parquet` and
`datasets/splits/sansar_duplicate_text_exclusions.csv`.

The normalizer changed 0 of these sample texts; all were already NFC. It found
line breaks in 143 rows, which remain preserved. Keep each record's ID,
source, work, licence, attribution, and raw text. Check the terms for every
included record before using or sharing derived data.

## Synthetic baseline pilot results

[`generate_synthetic_noise_baselines.py`](../../scripts/generate_synthetic_noise_baselines.py)
generated deterministic formatting variants and generic akshara-like edits
after splitting. It did not use a Sanskrit OCR confusion table. Metrics are
saved in `experiments/normalization/generated/synthetic_baseline_metrics.csv`;
generated text pairs are stored beside them. These generated artifacts are
excluded from Git.

On the 500-row test split, formatting corruption had mean CER 0.0642 for the
identity baseline and 0 for the deterministic normalizer; exact match rose
from 0 to 1. Generic edit mean CER did not improve: it was 0.0370 vs 0.0371
at 1% target unit edits, 0.0378 vs 0.0380 at 3%, and 0.0740 vs 0.0742 at 5%.
Exact match was unchanged at each generic-edit severity. The rule normalizer
is intentionally a formatting cleanup pass, so these results do not imply
OCR-character correction. The 1/3/5% labels are requested edit rates over
akshara-like units, not measured CER; short-text lengths can make realized CER
non-monotonic.

## Held-out real OCR baseline check

A separate Sanskrit post-OCR correction dataset was identified from
Maheshwari et al. (Findings of EMNLP 2022). Its maintainers describe the
"ood_test" split as 500 manually post-edited, out-of-domain examples. The
Hugging Face mirror declares MIT and provides "input_text" and "target_text"
fields. Only this held-out split was downloaded; it was not used to tune rules
or train a model. The source file is pinned to revision
"35de91d045dfd8c9bd875b44f2bfda52f09f72ce" and its SHA-256 is recorded in
"experiments/normalization/real_ocr/ood_test_baseline_metrics.json".

The fixed normalizer changed 0 of 500 OCR inputs. Identity and rule baselines
therefore have the same micro-CER (0.059719) and exact-match rate (55.2%). The
later learned-model pilot scored OOD CER 1.1831 and 0% exact match, worse than
both baselines. Split overlap, filtering, and model results are recorded in
`04_real_ocr_baseline_review.md`.

The OCR dataset card reports this benchmark's scope as classical Sanskrit in
Devanagari and lists known limitations around its selected source texts and
error types. Retain attribution and source metadata and review the original
book/image rights before redistributing any material. See
"04_real_ocr_baseline_review.md" for the dataset review and interpretation.

## Split protocol

1. Apply the documented clean-text normalization to each source record, while
   retaining the raw text and metadata.
2. Remove repeated `normalized_text` values from the split candidate, keeping
   the first record in sample order. Record omitted IDs and representative
   IDs so duplicate removal is auditable; leave the source-aware sample
   unchanged. **Completed** in the split manifest.
3. Group remaining records by `source` plus `work` when `work` is present.
   When `work` is missing, use `source` as the group.
4. Assign whole groups to train, validation, and test, targeting 80/10/10 with
   split seed `42`. Group sizes may make the final proportions differ.
   **Completed**; the split reached 80/10/10 by row count.
5. Check that neither group IDs nor exact clean texts cross partitions.
   **Completed**; both overlap counts are zero.
6. Only after assignment, generate noisy variants inside each partition. All
   variants of a clean record stay in that record's partition. Use fixed seeds
   for validation and test so repeated runs produce the same inputs.
7. If the bounded sample does not contain enough distinct groups for this
   split, gather a larger but still capped source-aware sample. Do not fall
   back to a random row split and claim source-level generalization.

If enough groups are available, reserve one or more complete source/work
groups for an additional out-of-domain test. This is separate from the main
test split.

## Corruption tracks

Keep these tracks separate so results show what the normalizer can already
solve and what may require a learned correction model.

| Track | Controlled changes | Expected baseline behavior |
|---|---|---|
| Unicode and formatting | NFC text represented as NFD; CRLF/CR line-ending variants; tabs, repeated horizontal spaces, and edge spaces | The deterministic normalizer should repair these changes. Preserve verse line breaks and punctuation. |
| Generic character edits | Character insertion, deletion, and substitution | Report as generic synthetic edits until a Sanskrit OCR confusion table supports calling them OCR-like. |
| Glyph-confusion edits | Confusions between visually similar Devanagari glyphs, if supported by paired Sanskrit OCR evidence or a manually reviewed table | Evaluate separately from generic edits and record the evidence for each confusion. |

Do not remove line breaks or alter danda punctuation in the initial format
track. The current sample suggests line breaks may mark verse structure, and
the baseline deliberately preserves them.

Use noise seed `43` and record the noise type, severity, and realized
character error rate (CER) for each pair. Choose severity ranges in a small pilot and
report the realized CER distribution; do not copy rates from another language
without evidence. Keep single-track runs separate from combined-noise runs so
an ablation can identify which corruption types matter.

## Baselines and evaluation

- **Identity baseline:** return the noisy input unchanged.
- **Rule baseline:** apply `src/normalization/text_normalizer.py`.
- **Learned model:** consider only after the synthetic pairs and splits are
  fixed; compare against both baselines.
- Report character error rate after applying NFC consistently to prediction
  and reference, plus exact-match rate. Include a qualitative review of
  corrections that change Sanskrit content or verse structure.
- Report formatting-noise results separately from character-edit results.
  Good results on synthetic edits do not establish performance on real OCR.

## Research basis and limits

OCR post-correction studies have investigated multiple synthetic noise levels
and CER-based selection of useful synthetic examples ([Guan et al., EMNLP
2024](https://aclanthology.org/2024.emnlp-main.862/)). Another study compares
synthetic-data volume, augmentation, and generation methods, including a
glyph-similarity method ([Guan and Greene, Findings of ACL 2024](https://aclanthology.org/2024.findings-acl.361/)). An Icelandic study used both real
and artificially generated OCR errors and reported that model variants did
not benefit identically from the generated errors ([Jasonarson et al.,
NoDaLiDa 2023](https://aclanthology.org/2023.nodalida-1.29/)). These are
method references, not evidence that their noise distributions transfer to
Sanskrit.

The current Sansar sample has no verified noisy-to-clean pairs. A separate
manually post-edited Sanskrit OCR benchmark has now been identified and its
500-example out-of-domain split evaluated; see
04_real_ocr_baseline_review.md. Synthetic results must still be labeled as
synthetic. Any learned model should use the benchmark's training/validation
splits and keep its out-of-domain test set untouched.
