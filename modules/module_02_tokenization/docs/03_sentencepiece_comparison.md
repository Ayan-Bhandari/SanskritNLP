# Module 2, Stage 3: SentencePiece subword comparison

## Purpose

Subword units are a possible middle ground between the very large whitespace
vocabulary and long code-point sequences. This small experiment compares
SentencePiece BPE and Unigram models at requested vocabulary sizes 512 and
1,024. SentencePiece supports both methods and provides Python training and
encode/decode APIs ([official documentation](https://google.github.io/sentencepiece/)).

No neural NLP model was trained. Tokenizer vocabulary fitting ran on CPU.

## Data and safeguards

- Input: the same 4,997-row work-group split used by the earlier baselines.
- Training: 3,997 train rows only. Validation and test text did not contribute
  to vocabulary fitting.
- Text normalization: identity inside SentencePiece; the project normalizer's
  output is used without additional NFKC or whitespace cleanup.
- Line breaks: represented during fitting and encoding with a reserved
  user-defined marker, then restored after decoding. The script first checks
  that the marker does not already occur in the source text.
- Long rows: SentencePiece's configured maximum was set above the longest
  UTF-8 training record so the trainer would not silently skip the long-tail
  records.
- Artifacts: model and vocabulary files plus aggregate metrics remain in the
  ignored local `experiments/tokenization/` directory.

## Results

All four models reached their requested vocabulary size. No unknown tokens
were produced. Exact reconstruction succeeded for every train, validation,
and test row.

| Model | Vocabulary | Mean train pieces/row | Mean validation pieces/row | Mean test pieces/row | Test p95 / max | Test exact round-trip |
|---|---:|---:|---:|---:|---:|---:|
| BPE | 512 | 160.42 | 45.10 | 45.34 | 53 / 98 | 500/500 |
| BPE | 1,024 | 135.89 | 38.53 | 38.72 | 45 / 88 | 500/500 |
| Unigram | 512 | 161.40 | 45.76 | 46.64 | 56 / 107 | 500/500 |
| Unigram | 1,024 | 137.64 | 39.36 | 39.50 | 48 / 90 | 500/500 |

On this sample, 1,024-entry BPE gives the shortest test sequences of the
subword candidates. It averages 38.72 pieces per test row, between the earlier
whitespace baseline (12.46) and code-point baseline (86.62). It also has a
smaller training vocabulary than whitespace tokens (1,024 versus 75,829)
without producing OOVs on these held-out rows.

The longest training row still produces 11,512 BPE pieces at vocabulary size
1,024. Module 5 batching should account for this long tail instead of silently
discarding it.

## Interpretation and limits

These are intrinsic tokenizer measurements, not language-model scores or
tokenization accuracy. Exact reconstruction only says the input can be
recovered. Zero OOV is specific to this sample, the configured character
coverage, and the special unknown-token behavior; it is not a guarantee for
future Sanskrit sources. The character vocabulary was already known to cover
the held-out code points in this split.

The data is a small work-held-out sample, not the full 1.8-million-row Sansar
training split. The work-group split also has a strong record-length skew.
Subword pieces are not Sanskrit word boundaries and do not undo Sandhi; that
is Module 3's problem. We should not select the final tokenizer for the whole
project solely from these results. A later downstream language-model
comparison should hold data, model size, and training budget constant.

## Reproduction

Install the dependency in the project environment if needed:

```powershell
python -m pip install -r requirements.txt
```

Then run from the repository root:

```powershell
python modules\module_02_tokenization\scripts\compare_sentencepiece_tokenizers.py
```

This experiment used SentencePiece 0.2.3 and four CPU threads. Aggregate
output is written to the ignored local file
`experiments/tokenization/module2_sentencepiece_comparison.json`.
