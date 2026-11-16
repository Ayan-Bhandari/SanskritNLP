# Module 2, Stage 4: full-corpus subword tokenizers

## Purpose and input

This stage fits BPE and Unigram token vocabularies on the complete local
Sansar normalized training split. Tokenizer fitting is not neural-model
training and does not use the GPU. Both models use byte-level processing so
the full byte alphabet is available and source text can be reconstructed.

Input provenance is checked against the Module 1 manifest:

- Dataset: `MuseMesh/sansar-sanskrit-corpus`
- Revision: `a268a87223c38575883c7ee6af25b1d2e65fd0d7`
- Config/split: `default/train`
- Manifest count: 1,829,785 records in 183 Parquet shards
- Vocabulary target: 8,192 entries per model

## Memory and data handling

The script streams `normalized_text` from Parquet batches into the Hugging
Face Tokenizers iterator trainer. The source shards are not modified and no
temporary 5.5 GiB text copy or per-row tokenized dataset is created. Models
are saved under the ignored local directory
`models/module_02_tokenization/full_corpus/`; aggregate reports are under
`experiments/tokenization/full_corpus/`.

One normalized record is about two million Unicode code points. Training
consumes the complete record. For evaluation, selected long records are
processed in codepoint-safe 1,024-character chunks to bound memory. Text is
not truncated. Since chunk edges can affect subword merges, token counts for
the exceptionally long rows may be slightly higher than whole-row encoding.
Chunks are decoded and joined to check exact reconstruction.

## Evaluation scope

The models are fitted on every row in `default/train`. For practical
verification, aggregate token counts and round-trip checks are computed on
5,000 deterministic, evenly spaced row positions across that same split.
This is an in-corpus sample, not a held-out evaluation: these rows contributed
to vocabulary fitting. It establishes tokenizer coverage and reversibility
for the sampled rows; it does not measure downstream NLP accuracy or
generalization to new Sanskrit sources. A held-out source or split must be
reserved before fitting for that purpose.

## Results

This section records the actual run output. Per-model details are in the
ignored JSON reports alongside the local model files.

| Model | Vocabulary | Fit rows | Sample rows | Mean tokens / row | Unknown pieces | Exact reconstruction |
|---|---:|---:|---:|---:|---:|---:|
| BPE | 8,192 | 1,829,785 | 5,000 | 868.72 | 0 | 5,000 / 5,000 |
| Unigram | 8,192 | 1,829,785 | 5,000 | 1,357.70 | 0 | 5,000 / 5,000 |

The 868.72 BPE mean comes from evenly spaced rows across the full corpus and
includes its long-record tail. It is not comparable to the earlier sample
comparison's mean because that work-group split had a different record-length
distribution.

On this evenly spaced in-corpus sample, BPE produces fewer pieces per row
than Unigram. This is a sequence-length comparison only; it does not establish
which tokenizer improves a downstream task. The sample covers 6,043,713
Unicode code points and 16,541,320 UTF-8 bytes. Its median row has 66 BPE
pieces and 104 Unigram pieces; the sampled maximum has 275,400 and 424,581
pieces respectively. This large tail should be accounted for in later model
batching.

The BPE model was fit in about 8.8 minutes and Unigram in about 28.2 minutes
on this laptop's CPU. Each model reached the requested 8,192 entries. These
durations are specific to this hardware and run.

## Reproduction

From the repository root, with the project `.venv` active:

```powershell
python -m pip install -r requirements.txt
python modules\module_02_tokenization\scripts\train_full_corpus_tokenizers.py --vocab-size 8192
```

The experiment uses Hugging Face Tokenizers' iterator training interface
([official API reference](https://huggingface.co/docs/tokenizers/python/latest/api/reference.html)).
Use `--models bpe` or `--models unigram` to run one model. Existing local
model JSON files are reused to resume evaluation; delete the ignored model
file only if you intentionally want to retrain it.
