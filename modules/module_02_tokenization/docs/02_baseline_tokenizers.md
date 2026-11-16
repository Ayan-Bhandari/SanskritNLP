# Module 2, Stage 2: whitespace and code-point baselines

## Purpose and protocol

This first comparison measures vocabulary scale, held-out coverage, token
sequence lengths, and exact reconstruction for two simple tokenizers. It uses
the existing 4,997-row work-group split. Token vocabularies are counted from
the 3,997 training rows only; validation and test remain held out.

The whitespace tokenizer identifies each non-whitespace span as a token and
stores the original whitespace separators for an exact reconstruction
check. The code-point tokenizer emits each Python Unicode code point,
including spaces and line breaks, as one unit. Both are deterministic and
require no learned neural model.

## Results

| Split | Rows | Whitespace mean units | Code-point mean units | Whitespace exact round-trip | Code-point exact round-trip |
|---|---:|---:|---:|---:|---:|
| Train | 3,997 | 37.36 | 308.84 | 3,997/3,997 | 3,997/3,997 |
| Validation | 500 | 10.79 | 86.26 | 500/500 | 500/500 |
| Test | 500 | 12.46 | 86.62 | 500/500 | 500/500 |

Vocabulary and held-out token coverage:

| Tokenizer | Train vocabulary types | Validation unknown token instances | Test unknown token instances |
|---|---:|---:|---:|
| Whitespace | 75,829 | 2,394/5,397 (44.36%) | 2,269/6,229 (36.43%) |
| Unicode code point | 170 | 0/43,130 (0%) | 0/43,311 (0%) |

Among training whitespace tokens, 23,869 contain danda or double danda.
Because punctuation remains attached to many whitespace tokens, punctuation
handling is a tokenizer design question for the next comparison; normalization
currently preserves punctuation unchanged.

## Interpretation

- The whitespace baseline is short and readable, but the held-out work groups
  contain many word forms absent from the training vocabulary. These OOV
  counts describe this small held-out split and do not establish the quality
  of a future model.
- The code-point vocabulary is compact and has no held-out OOV code points in
  this sample. It produces much longer sequences. It also splits a Devanagari
  orthographic form such as `कि` into the consonant code point `क` and vowel
  sign `ि`; code points should not be treated as akshara units.
- Exact reconstruction only confirms that tokenization can preserve source
  strings. It does not measure Sanskrit word-boundary correctness.
- Whitespace boundaries do not recover words hidden by Sandhi. That remains
  Module 3's task.

Neither baseline is selected as the final tokenizer. The next comparison
should fit a modest BPE or Unigram subword vocabulary on the training
partition only, inspect its vocabulary size and held-out OOV/sequence lengths,
and verify exact encode/decode reconstruction. This is CPU-feasible; no GPU
was used.

## Reproduction

From the repository root, run:

```powershell
python modules\module_02_tokenization\scripts\compare_baseline_tokenizers.py
```

Aggregate output is written to the ignored local file
`experiments/tokenization/module2_baseline_tokenizers.json`. It contains no
Sanskrit text examples.
