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

The candidates and metrics will be refined after the dataset readiness report.
No tokenizer package or new dependency has been added yet.

## Current stage: data readiness

The initial audit uses the existing 4,997-row work-group split produced in
Module 1. It does not download data or alter the source files. Run from the
repository root:

```powershell
python modules\module_02_tokenization\scripts\inspect_tokenization_sample.py
```

The script writes aggregate-only results to
`experiments/tokenization/module2_sample_audit.json`; no Sanskrit text is
copied into that report. See `docs/01_data_readiness.md` for the recorded
findings and the next stage.

## GPU

No GPU is needed for corpus inspection, whitespace/character baselines, or
fitting a small SentencePiece vocabulary. CPU is sufficient for this module's
initial comparison. GPU may matter later when a neural downstream model is
trained to measure the effect of tokenization, especially in Module 5.
