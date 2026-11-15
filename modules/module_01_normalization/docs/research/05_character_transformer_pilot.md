# Module 1 character Transformer pilot

## Question

Can a small character-level encoder-decoder learn Sanskrit post-OCR
correction from the benchmark's paired OCR and manually corrected texts? This
is a feasibility pilot; it is not intended as a production model.

## Data and leakage controls

Source: `acomquest/sanskrit-ocr-post-correction` at revision
`35de91d045dfd8c9bd875b44f2bfda52f09f72ce`. The raw source files remain under
`datasets/raw/sanskrit_ocr_post_correction/`. The audit found canonical target
overlap between source splits, so the prepared partitions give priority to
OOD, then test, validation, and train, removing duplicate canonical targets
from lower-priority splits. See `04_real_ocr_baseline_review.md` and
`datasets/splits/ocr_target_disjoint/manifest.json` for counts and caveats.

The model training sample is a seeded random sample of 10,000 rows from the
prepared training partition. Rows over 256 Unicode code points are excluded.
The model has 202,768 parameters and a 144-token vocabulary. It was trained
for 10 epochs on CPU using PyTorch 2.15.0+cpu. Best validation token loss was
2.5930 at epoch 10. Training loss fell from 4.0877 to 2.6366, while validation
loss fell from 3.3645 to 2.5930.

## Results

| Evaluation split | Rows | Identity CER | Normalizer CER | Model CER | Model exact match |
|---|---:|---:|---:|---:|---:|
| Validation | 4,876 | 0.04252 | 0.04134 | 1.25741 | 0.0% |
| Test | 4,933 | 0.04355 | 0.04251 | 1.23217 | 0.0% |
| OOD test | 500 | 0.05972 | 0.05972 | 1.18311 | 0.0% |

The model worsened every validation and test row versus both baselines. On OOD,
it improved only 6 rows and worsened 491 versus identity; CER was still far
worse. This experiment is a clear failure as an OCR corrector. Falling
teacher-forced token loss was not enough to produce useful autoregressive
decoding. Do not deploy this checkpoint or describe it as a successful
correction model.

## Reproduction and artifacts

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe modules\module_01_normalization\scripts\train_module1_character_transformer.py
```

The script writes `training_history.csv`, validation/test/OOD prediction CSVs,
and `model_metrics.json` under
`experiments/normalization/real_ocr/character_transformer_10k/`. The selected
checkpoint and training sample row IDs are under
`models/module1_character_transformer_10k/` and the experiment directory,
respectively. These artifacts are local research outputs and are excluded
from Git.

## Interpretation and next research iteration

The pilot used only 10k of more than 200k prepared training pairs, a short
10-epoch schedule, code points rather than akshara/byte-pair units, and a
simple greedy decoder. These limits may explain some quality loss, but they do
not change the measured result: this run is unusable. A future iteration
should first diagnose output collapse on training examples, inspect token and
length handling, and compare simpler edit-based or alignment-aware models
before spending time on a large GPU run. The current environment has CPU-only
PyTorch; selecting laptop dGPU mode does not itself enable CUDA.
