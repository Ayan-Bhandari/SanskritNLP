# Module 2: Sanskrit tokenization

## Goal

Compare practical ways to represent normalized Sanskrit text as model tokens.
Tokenization prepares text for later NLP models. It does **not** recover
Sanskrit word boundaries obscured by Sandhi; that is Module 3.

## Candidate approaches

1. **Whitespace baseline:** one token per space-delimited unit. This is simple
   and reversible, but punctuation may stay attached and boundaries reflect
   the source text.
2. **Character baseline:** Unicode code points. It gives a small vocabulary,
   but code points can split Devanagari combining sequences.
3. **Akshara-like baseline:** group Devanagari consonants, virama, and
   dependent signs into orthographic units. Define and inspect the rules before
   treating these as linguistic syllables.
4. **Subword tokenizers:** compare learned BPE and/or Unigram vocabularies.
   Fitting a tokenizer vocabulary is light CPU work, not neural-model
   training.

The candidates and metrics will be refined as experiments progress. The
module now uses the lightweight SentencePiece dependency for its BPE/Unigram
comparison.

## Stage 1: data readiness

The initial audit uses the existing 4,997-row work-group split produced in
Module 1. It does not download data or alter the source files. Run from the
repository root:

```powershell
python modules\module_02_tokenization\scripts\inspect_tokenization_sample.py
```

The script writes aggregate-only results to
`experiments/tokenization/module2_sample_audit.json`; no Sanskrit text is
copied into that report. See `docs/01_data_readiness.md` for the findings.

## Stage 2: baseline comparison

Run the whitespace and Unicode-code-point baselines:

```powershell
python modules\module_02_tokenization\scripts\compare_baseline_tokenizers.py
```

The script counts vocabularies from train only and writes aggregate metrics to
`experiments/tokenization/module2_baseline_tokenizers.json`. It preserves
whitespace separators during the exact reconstruction check. Current findings
are documented in `docs/02_baseline_tokenizers.md`.

The initial results show why a subword candidate is worth evaluating:
whitespace has high held-out OOV rates, while code points have a small
vocabulary but much longer sequences and can split Devanagari combining
marks. Neither baseline is selected as the final tokenizer.

## Stage 3: subword comparison

Install the added dependency with `pip install -r requirements.txt` if it is
not already present, then run:

```powershell
python modules\module_02_tokenization\scripts\compare_sentencepiece_tokenizers.py
```

This trains BPE and Unigram vocabularies of 512 and 1,024 entries from the
train split only. Validation and test are only used for reporting. The script
keeps line breaks through an explicit marker and configures SentencePiece to
use identity text normalization. Model files and aggregate metrics stay under
the ignored local `experiments/tokenization/` directory. See
`docs/03_sentencepiece_comparison.md` for results and interpretation.

The package's official [documentation](https://google.github.io/sentencepiece/)
describes SentencePiece's BPE and Unigram training and its Python interface.

## Stage 4: full-corpus subword models

The complete local Module 1 `default/train` corpus is used to fit byte-level
BPE and Unigram models with an 8,192-entry target vocabulary. The script
streams all 183 Parquet shards through Hugging Face Tokenizers; it does not
create an intermediate text file or a second tokenized copy of the corpus.
The tokenizer models and aggregate-only reports are written to ignored local
`models/` and `experiments/` folders.

Install dependencies with `python -m pip install -r requirements.txt`, then
run from the repository root:

```powershell
python modules\module_02_tokenization\scripts\train_full_corpus_tokenizers.py --vocab-size 8192
```

The run fits each model on every row in the complete training split. It then
reports in-corpus metrics and exact reconstruction on 5,000 deterministic,
evenly spaced rows. These are not held-out accuracy or generalization results,
because the tokenizer was fitted on the full split. See
`docs/04_full_corpus_tokenizers.md` for the completed run and its limits.
Tokenizer training is CPU work; a GPU is not used.

## Stage 5: tokenize the complete split

The selected BPE tokenizer has now generated IDs for all 1,829,785 training
rows. Run or resume the shard-based job with:

```powershell
python modules\module_02_tokenization\scripts\tokenize_full_corpus.py
```

To process a fixed number of new shards per session, use
`--max-shards 10`, then rerun the same command on a later day. Completed
shards are skipped using a revision- and model-checked progress manifest.
Outputs are local under `datasets/tokenized/`; each Parquet shard contains
`id`, `source_row_index`, `token_ids`, and `token_count`. Token IDs are stored
as `uint16` and can be cast to `int64` when creating model tensors. See
`docs/04_full_corpus_tokenizers.md` for the output totals and verification.

## GPU

No GPU is needed for corpus inspection, whitespace/character baselines, or
fitting a small SentencePiece vocabulary. CPU is sufficient for this module's
initial comparison. GPU may matter later when a neural downstream model is
trained to measure the effect of tokenization, especially in Module 5.
