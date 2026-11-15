# Environment setup

## Windows PowerShell

From the repository root, create and activate a virtual environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If activation is unavailable or blocked, invoke the interpreter by its full
path instead:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Activation must be run from the repository root, where `.venv` exists. If the
folder is absent, create the environment first. PowerShell's activation
command is `.\.venv\Scripts\Activate.ps1` (without a trailing slash).

## GPU training

The prior OCR pilot used a CPU-only PyTorch build. A Lenovo dGPU setting does
not change which PyTorch build is installed. For a GPU run:

1. Use Lenovo **dGPU** mode while training if you want to force discrete GPU
   use; keep the laptop connected to power. A system restart may be required.
2. Install the PyTorch build appropriate for the machine using the official
   [PyTorch installation selector](https://pytorch.org/get-started/locally/).
3. In the same virtual environment, verify `torch.cuda.is_available()` before
   starting a training job.

The exact NVIDIA model and CUDA driver compatibility should be checked on the
machine before selecting the PyTorch build. GPU training has not been completed
for this project.
