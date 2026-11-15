# Sanskrit NLP Toolkit

An early-stage toolkit for Sanskrit corpus inspection and text normalization,
with a separate research track for post-OCR correction. The current completed
baseline makes Unicode and whitespace representation consistent while
preserving the original text and verse line breaks.

## Project status

- **Module 1 normalization:** completed for the pinned Sansar
  `default/train` split: 1,829,785 records across 183 local Parquet shards.
  Each output row retains `raw_text` and adds `normalized_text`.
- **Coverage:** this is one pinned training split, not every upstream Sansar
  split or version.
- **OCR correction:** the benchmark audit and model feasibility pilot are
  complete. The pilot model performed far worse than the identity baseline
  and is not suitable for correction. See
  [the research report](modules/module_01_normalization/docs/research/05_character_transformer_pilot.md).
- **Data/model files:** large datasets, generated experiment outputs, and
  checkpoints stay local and are excluded from Git. Teammates can recreate
  them using the scripts and instructions below.

## Quick start (Windows PowerShell)

Use Python 3.12, which is the version used for the current local environment.

```powershell
Set-Location 'C:\path\to\SANSKRIT-NLP-TOOLKIT'
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, either follow your organization's execution
policy or skip activation and invoke the environment directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For GPU training, select the PyTorch build matching the installed NVIDIA GPU,
driver, and CUDA option using the official
[PyTorch installation selector](https://pytorch.org/get-started/locally/).
Changing Lenovo's Hybrid/dGPU mode alone does not install CUDA-enabled
PyTorch. Verify GPU availability in the same environment before training.

More detail: [environment setup](docs/SETUP.md).

## Normalization

The documented deterministic rules are implemented in
[`src/normalization/text_normalizer.py`](src/normalization/text_normalizer.py):
NFC Unicode normalization, consistent line endings, horizontal whitespace
collapse, and line-edge trimming. Punctuation and line breaks are preserved.
The rules do not fix OCR character substitutions or validate Sanskrit.

Run the small Module 1 examples from the repository root:

```powershell
python modules\module_01_normalization\scripts\normalization_examples.py
```

To reproduce the full pinned Sansar training-split normalization, first install
the dependencies, then run:

```powershell
python modules\module_01_normalization\scripts\normalize_sansar_full_corpus.py
```

This downloads/processes a large corpus and writes local output under
`datasets/processed/sansar_normalized/`. It may take substantial time and disk
space. See [dataset guidance](docs/DATASETS.md) before running data pipelines.

## Repository map

```text
modules/                 Numbered, self-contained learning/research modules
  module_01_normalization/
    src/                 Module 1 normalization implementation
    scripts/             Module 1 data and evaluation pipelines
    notebooks/           Exploratory corpus analysis
    docs/research/        Module 1 protocols, audits, and results
docs/SETUP.md            Shared environment setup and GPU notes
docs/DATASETS.md         Shared data sources and local storage policy
datasets/                Local-only raw and processed data (ignored by Git)
experiments/             Local generated metrics/predictions (ignored by Git)
models/                  Local checkpoints (ignored by Git)
```

## Data and attribution

The project uses the [MuseMesh Sansar Sanskrit Corpus](https://huggingface.co/datasets/MuseMesh/sansar-sanskrit-corpus)
and an OCR correction benchmark described by Maheshwari et al. ([paper](https://aclanthology.org/2022.findings-emnlp.466/),
[authors' repository](https://github.com/ayushbits/pe-ocr-sanskrit)). Follow
the source dataset terms and attribution. The OCR mirror's MIT declaration
does not independently establish rights for every underlying source book or
scan. Do not commit or redistribute corpus files through this repository.

## For contributors

See [the module index](modules/README.md) for the structure and planned
organization of future work. See [CONTRIBUTING.md](CONTRIBUTING.md) for the workflow and data-handling
rules. The project does not currently include an automated test suite or a
repository-level open-source license; agree on licensing before public
redistribution.
