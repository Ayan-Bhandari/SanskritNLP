# Module 3: Sanskrit word segmentation

## Objective

Recover Sanskrit word boundaries from joined or partially joined text. The
Hackathon DCS benchmark uses Roman transliteration and contains both ordinary
boundary insertion and cases where the input/output character sequences differ
because of sandhi or annotation conventions.

## Current status

The local `Hackathon_dcs.csv` file was audited read-only. It contains 110,295
rows: 90,000 train, 10,332 dev, and 9,963 test. The audit found no missing or
empty fields, 35 inputs with multiple distinct output annotations, and 115
inputs occurring across multiple splits. In 88,434 rows, input and output
remain different after whitespace is removed. A boundary-only tagger therefore
cannot represent every target in this benchmark.

The reproducible audit script is
[`scripts/audit_hackathon_dataset.py`](scripts/audit_hackathon_dataset.py).
Run it from the repository root:

```powershell
python modules/module_03_word_segmentation/scripts/audit_hackathon_dataset.py `
  --input "C:\path\to\Hackathon_dcs.csv"
```

It writes a detailed Markdown and JSON report under the ignored local
`experiments/module_03_segmentation/` directory. The source CSV, generated
reports, cleaned datasets, and model files are not tracked in Git.

## Baselines and model

Run the first baselines with:

```powershell
python modules\module_03_word_segmentation\scripts\evaluate_baselines.py `
  --input "C:\path\to\Hackathon_dcs.csv"
```

This compares unchanged input with a train-set most-frequent mapping. The
current untouched-input test baseline is 0.539 macro word F1 and 6.43% exact
sentence match; this is a starting point, not a trained model result. Aggregate
metrics are written locally to `experiments/module_03_segmentation/`.

The full generative baseline is a character-level Transformer
encoder-decoder. It predicts the segmented output as a sequence, so it can
represent both boundary insertions and character changes. Train and evaluate
with:

```powershell
python modules\module_03_word_segmentation\scripts\train_character_transformer.py `
  --input "C:\path\to\Hackathon_dcs.csv"
```

Use `--resume` to continue from the best saved checkpoint after an interrupted
run. GPU is optional, but recommended. The project environment needs a
CUDA-enabled PyTorch build for training to use the NVIDIA GPU; a discrete GPU
being present does not make a CPU-only PyTorch install use it automatically.
The script detects the available device and records it in the report.

For a dev-only error report and up to 200 difficult examples for manual review:

```powershell
python modules\module_03_word_segmentation\scripts\analyze_predictions.py `
  --input "C:\path\to\Hackathon_dcs.csv" --split dev
```

The dev analysis identified a long-input failure mode. Current inference uses
the full-sentence model for inputs up to 55 characters and separately decodes
existing whitespace-delimited chunks for longer inputs. It caps generated
output at source length plus 32 characters and returns the unchanged input for
a chunk/sentence whose non-whitespace character edit ratio exceeds 30%. These
limits were chosen on dev; the original source text is always preserved.

## Limitations and follow-on work

- The current system is a compact baseline; reproducing full TransLIST would
  require its lattice candidates, linguistic resources, and PRCP inference.
- Train a chunk-aware model (rather than relying only on inference-time chunk
  routing) and evaluate it on dev before touching the frozen test set.
- Review ambiguous annotations and error cases before choosing a canonical
  output policy or claiming production accuracy.
- Keep the standard test result and the leakage-filtered result separate in
  future comparisons.
- Compare future models using the same splits, preprocessing, and metrics.

## Completed run

The model was trained on the RTX 4050 using CUDA-enabled PyTorch on all
**77,930 unique train input/target pairs** for 12 epochs, selecting epoch 12 by
dev token loss. No rows were excluded for length. Exact duplicate
input/target pairs were removed from the derived training and evaluation views;
the source CSV remains unchanged. Alternative outputs were retained as
references. The guarded decoder below uses the same checkpoint and adds the
dev-selected long-input route and identity fallback; it does not retrain.

| Split / metric | Unchanged input baseline | Transformer with guarded decoding |
|---|---:|---:|
| Dev macro word F1 | 0.473 | 0.780 |
| Dev exact sentence match | 7.67% | 37.09% |
| Test macro word F1 | 0.539 | 0.843 |
| Test exact sentence match | 6.43% | 43.07% |

Only 6 test inputs overlap the training inputs. On the remaining **9,752**
test inputs, macro word F1 was **0.843** and exact sentence match was
**43.07%**, nearly unchanged from the full test score. Metrics score each
unique input once and accept any distinct gold output available for that input.

The dev error report shows macro word F1 of 0.715 for 61–90 character inputs
and 0.748 for inputs longer than 90; exact sentence match is 10.8% and 3.1%
for those groups. The long-input route improves those groups substantially,
but long-context segmentation remains a major weakness.

The model is a compact character Transformer baseline, not a reproduction of
the full TransLIST architecture. It demonstrates strong improvement over the
unchanged-input baseline, but the 43.1% exact sentence match leaves meaningful
errors, especially in sandhi and long/ambiguous inputs. Module 3’s baseline and
inference path are implemented, but this is not production-ready accuracy.
Keep results labeled as this project’s baseline and do not claim direct
comparability with TransLIST.

The trained checkpoint is stored locally at
`models/module_03_word_segmentation/best_model.pt`. To segment a sentence:

```powershell
python modules\module_03_word_segmentation\scripts\segment_text.py `
  --text "corau śvapākacaṇḍālau vipreṇābhihatau yadi"
```

Or pass a UTF-8 file with one sentence per line using `--input-file`. The model
accepts Roman-transliterated inputs up to 256 Unicode code points and reports
an error for characters outside its training vocabulary. Aggregate metrics,
training history, and predictions remain local under
`experiments/module_03_segmentation/` and are excluded from Git.

## Research references

- [TransLIST paper](https://aclanthology.org/2022.findings-emnlp.513/)
- [TransLIST data directory](https://github.com/Jivnesh/SanskritShala/tree/master/Neural_Modules/TransLIST/LREC-Data)
