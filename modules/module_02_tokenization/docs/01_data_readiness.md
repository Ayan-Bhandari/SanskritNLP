# Module 2, Stage 1: sample readiness

## Purpose

Before comparing tokenizers, verify which local text and evaluation split are
available. This stage uses Module 1's source/work-group split so tokenizer
choices can be fitted on train and later compared without tuning on held-out
text.

## Input

`datasets/splits/sansar_source_aware_group_splits.parquet`, generated from a
5,000-row source-aware Sansar sample. Deduplication left 4,997 unique texts.
The split contains 3,997 train, 500 validation, and 500 test rows. It assigns
whole source/work groups to partitions (170/18/19 groups, respectively).
Validation and test groups are held out from train.

The initial 5,000-row pilot file is not used here because it has only one
source value and no populated `work` field. The work-group split is more
appropriate for this comparison.

## Audit findings

All 4,997 split rows have nonempty normalized text. The upstream `num_words`
field equals a whitespace split count for all 4,997 rows in this sample. This
makes whitespace tokenization a useful reproducible baseline, but it does not
prove that those boundaries are linguistically correct or undo Sandhi.

Token-length distribution by whitespace baseline:

| Partition | Rows | Groups | Median tokens | 90th percentile | 95th percentile | Maximum |
|---|---:|---:|---:|---:|---:|---:|
| Train | 3,997 | 170 | 11 | 17 | 176 | 2,523 |
| Validation | 500 | 18 | 11 | 14 | 15 | 25 |
| Test | 500 | 19 | 13 | 15 | 16 | 29 |

The training partition has a long tail of unusually long records; its maximum
is far larger than validation/test. Later tokenizer fitting and comparisons
must inspect this skew and use consistent length handling. Do not silently
truncate or discard long records.

The audit found zero exact normalized-text overlaps and zero group overlaps
across the three partitions.

## Tokenization is not word segmentation

The whitespace baseline returns source-delimited chunks. A subword tokenizer
may split those chunks into frequent pieces, and a character tokenizer breaks
them further. Neither method should be described as recovering the hidden
words inside a Sandhi-joined form. That boundary-recovery task belongs to
Module 3 and will need its own annotated evaluation data.

## Proposed comparison measures

For each tokenizer, record:

- Vocabulary size and unknown-token behavior.
- Token counts and sequence-length distributions by split.
- Coverage of held-out characters/units and rare forms.
- Encode/decode round-trip preservation of the input string.
- Runtime and peak memory on the same sample.

These are tokenizer diagnostics, not word-segmentation accuracy. To measure
downstream usefulness, a later module should compare tokenizers under the
same language-model training and held-out evaluation setup.

## Current recommendation

Begin with whitespace and Unicode character baselines, then fit a small
SentencePiece BPE/Unigram tokenizer on train only. Keep validation and test
held out until vocabulary choices and normalization are fixed. Use CPU; no
neural model or GPU training is required for this stage.

Reproduce the aggregate inspection with
`scripts/inspect_tokenization_sample.py`. Its JSON report contains no text
examples and is written to the ignored local `experiments/` directory.
