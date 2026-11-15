# Module 1: Sansar sample quality baseline

## Purpose

Record the initial inspection and deterministic normalization baseline for the
Sansar Sanskrit Corpus. This note describes the sample that was actually
loaded; it does not claim that the sample represents the entire corpus.

Dataset: [MuseMesh/Sansar Sanskrit Corpus](https://huggingface.co/datasets/MuseMesh/sansar-sanskrit-corpus)

## Reproducible sample details

- Dataset revision: `a268a87223c38575883c7ee6af25b1d2e65fd0d7`
- Configuration: `default`
- Split: `train`
- Sampling method: first 5,000 records from a streaming iterator
- Dataset library: `datasets` 4.8.5
- Python: 3.12.10
- Fields returned: `id`, `text`, `source`, `source_url`, `source_record`,
  `licence`, `attribution`, `work`, `title`, `register`, and `num_words`

The dataset repository also exposed licence-named configurations during the
inspection. The default configuration was used for this sample; per-record
licence and attribution fields were retained in the processed output.

## Sample findings

| Check | Records |
|---|---:|
| Loaded | 5,000 |
| Missing text | 0 |
| Non-string text | 0 |
| Empty or whitespace-only text | 0 |
| Exact duplicate text rows after first occurrence | 0 |
| Text not in Unicode NFC | 0 |
| Text containing Devanagari letters | 5,000 |
| Text with mixed-script letters | 2 |
| Text containing danda (`।`) | 2,431 |
| Text containing double danda (`॥`) | 2,392 |
| Text with tabs | 0 |
| Text with line breaks | 12 |
| Text with nonstandard whitespace | 0 |
| Text with suspicious control or unassigned characters | 0 |

Punctuation counts can overlap: a row can contain both danda forms.

The sample contains one `source` value (`itihasa`), no non-missing `work`
values, and one licence value (`Apache-2.0`). It cannot support a
source-held-out or work-held-out model evaluation.

The 12 line breaks appeared between verse passages, often following danda
punctuation. That suggests they may carry layout or verse structure. The
normalizer therefore preserves line breaks rather than flattening them. This
is a conservative interpretation, not a claim that every source line break is
linguistically meaningful.

## Baseline transformation

The shared function in `src/normalization/text_normalizer.py`:

1. Applies Unicode NFC.
2. Converts CRLF and CR line endings to LF.
3. Collapses horizontal whitespace runs to one ordinary space within each
   line, then trims horizontal whitespace at each line edge.
4. Preserves line breaks, blank lines, danda punctuation, and other
   punctuation.
5. Passes non-string values through unchanged.

The raw text remains in `raw_text`; the normalized value is stored in
`normalized_text`. The sample file is
`datasets/processed/sansar_train_sample_5000_normalized.parquet`.

After adopting the line-preserving policy, the normalizer changed 0 of the
5,000 sample texts. This is consistent with the sample having no NFC, tab,
repeated-space, or edge-whitespace issues. It does not show that the rule set
has no value on other data, and it is not an OCR correction result.

## Limitations and next research decision

- The sample is the beginning of a split, not a random or stratified sample.
  Counts must not be generalized to the full corpus.
- The corpus is already described as normalized; it is useful for inspecting
  clean Sanskrit text, not as a supervised noisy-to-clean corpus.
- No real OCR-derived noisy sample was evaluated. Synthetic corruption can be
  used for an initial controlled experiment, but synthetic results alone
  cannot establish real OCR correction performance.
- Keep the deterministic normalizer as the baseline. Before building a neural
  correction model, define corruption types, hold out clean source texts
  between splits, and seek a small manually verified real-noise sample for an
  out-of-domain check if one is available.
