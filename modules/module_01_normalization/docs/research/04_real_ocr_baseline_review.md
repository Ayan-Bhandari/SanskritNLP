# Module 1: Real Sanskrit OCR dataset review

## Candidate and scope

Maheshwari, Singh, Krishna, and Ramakrishnan released a Sanskrit post-OCR
correction benchmark in Findings of EMNLP 2022. The paper describes manually
verified OCR/correction pairs from multiple classical books. Its repository
documents train, validation, and test CSVs plus a separate 500-sentence
out-of-domain test file.

The Hugging Face mirror is
acomquest/sanskrit-ocr-post-correction, pinned here to revision
35de91d045dfd8c9bd875b44f2bfda52f09f72ce. The mirror declares an MIT
license. Its data card describes two fields: OCR input text and manually
corrected target text; it reports 500 examples in the out-of-domain split and
notes that the data is limited to selected classical Devanagari sources and
may not cover every OCR error type.

The project downloaded the train, validation, test, and OOD files into
`datasets/raw/sanskrit_ocr_post_correction/`. The OOD CSV has 500 rows and is
semicolon-delimited UTF-8. The pinned mirror's local counts are 208,137 train,
5,000 validation, 5,000 test, and 500 OOD. These differ from the current card
and published paper; see `split_audit.json`. The raw files are kept unchanged.

The Hugging Face mirror's license declaration is recorded for reproducibility.
Before redistributing examples or source scans, retain the paper and dataset
attribution and check the rights associated with the underlying books/images.

## Fixed-rule baseline

The existing normalizer was compared with an identity baseline on all 500
pairs. Both prediction and target were compared in NFC; CER is Levenshtein
distance divided by reference length. The rule baseline was not modified
after seeing this split.

| Measure | Identity | Deterministic normalizer |
|---|---:|---:|
| Micro CER | 0.059719 | 0.059719 |
| Exact match | 55.2% | 55.2% |
| Inputs changed by normalizer | — | 0 |

Since the normalizer changed no input, these metrics are equal. The check
confirms that NFC and whitespace cleanup do not correct the OCR character
errors in this split. A later learned-model pilot is reported below.

Re-run from the project root with:

    .\.venv\Scripts\python.exe modules\module_01_normalization\scripts\evaluate_real_ocr_normalizer.py

Outputs:

- experiments/normalization/real_ocr/ood_test_baseline_metrics.json
- experiments/normalization/real_ocr/ood_test_pair_metrics.csv

## Research sources

- Maheshwari et al. (2022), [paper](https://aclanthology.org/2022.findings-emnlp.466/)
- Authors' [dataset and code repository](https://github.com/ayushbits/pe-ocr-sanskrit)
- Pinned [Hugging Face dataset card](https://huggingface.co/datasets/acomquest/sanskrit-ocr-post-correction/tree/35de91d045dfd8c9bd875b44f2bfda52f09f72ce)

## Split audit and leakage control

`modules/module_01_normalization/scripts/inspect_ocr_correction_splits.py` recorded schemas, counts, Unicode
normalization changes, and overlap. Canonicalized targets overlapped across
every pair of partitions, including 428 unique target strings shared by train
and OOD. The original partitions therefore cannot be treated as independent
for this experiment. `modules/module_01_normalization/scripts/prepare_ocr_correction_splits.py` created a
filtered working set with priority OOD > test > validation > train. It removed
3,398 train, 50 validation, and 8 test rows, yielding 204,739 train, 4,950
validation, 4,992 test, and 500 OOD rows. Canonical target intersections are
zero in the prepared splits. The manifest is
`datasets/splits/ocr_target_disjoint/manifest.json`.

## Character Transformer pilot

`modules/module_01_normalization/scripts/train_module1_character_transformer.py` trained a 202,768-parameter
character-codepoint encoder-decoder Transformer on a seeded 10,000-row subset
of prepared training data. It ran 10 epochs on CPU with PyTorch 2.15.0+cpu;
best validation token loss was 2.5930 at epoch 10. Decoded output quality
was poor:

| Split | Rows | Identity CER | Normalizer CER | Model CER | Model exact match |
|---|---:|---:|---:|---:|---:|
| Validation | 4,876 | 0.0425 | 0.0413 | 1.2574 | 0.0% |
| Test | 4,933 | 0.0435 | 0.0425 | 1.2322 | 0.0% |
| OOD | 500 | 0.0597 | 0.0597 | 1.1831 | 0.0% |

The model is substantially worse than copying OCR input and must not be used
for correction. Artifacts include history, per-row predictions, metrics,
sample row numbers, and checkpoint under
`experiments/normalization/real_ocr/character_transformer_10k/` and
`models/module1_character_transformer_10k/`. Rows longer than 256 code points
were excluded; the model uses Unicode code points, not Sanskrit akshara units.

The OOD rows were prioritized in filtering before the prepared validation and
test sets were generated. Exact canonical target leakage was removed, though
this does not rule out source-document or near-duplicate leakage.

## Reproduction and limits

Run `modules/module_01_normalization/scripts/inspect_ocr_correction_splits.py`, then
`modules/module_01_normalization/scripts/prepare_ocr_correction_splits.py`, then
`modules/module_01_normalization/scripts/train_module1_character_transformer.py`. The script is configured
for a bounded 10k CPU pilot; the installed PyTorch wheel is CPU-only. Choosing
dGPU mode on the laptop alone will not enable CUDA in this environment. A GPU
run requires CUDA-enabled PyTorch and confirming that the runtime detects it.
