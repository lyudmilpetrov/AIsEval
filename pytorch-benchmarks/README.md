# PyTorch Benchmarks

Native Python benchmark project for PyTorch. It runs synthetic, reproducible MLP,
CNN, LSTM, and Transformer workloads and records training and inference metrics.

## Install

From the `pytorch-benchmarks/` directory, create and activate a virtual environment,
then install this project in editable mode so the `pytorch-bench` console command is
created on your PATH.

### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

### Windows Git Bash

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

Install the PyTorch wheel that matches your hardware first if you need a specific
CUDA build: <https://pytorch.org/get-started/locally/>.

## Run

```bash
pytorch-bench --models mlp,cnn,lstm,transformer --device auto --epochs 3 --train-batches 20 --output ../results/pytorch.json
```

If `pytorch-bench` prints `command not found`, the package has not been installed
into the currently active environment or that environment is not on your PATH.
Run these checks from `pytorch-benchmarks/`:

```bash
source .venv/bin/activate        # macOS/Linux
# or: source .venv/Scripts/activate      # Windows Git Bash
# or: .\.venv\Scripts\Activate.ps1   # Windows PowerShell
python -m pip install -e .
python -m pip show pytorch-benchmarks
python -c "import shutil; print(shutil.which('pytorch-bench'))"
```

As an install-free fallback from the project directory, run the package directory
with Python directly:

```bash
python src/pytorch_benchmarks --models mlp,cnn,lstm,transformer --device auto --epochs 3 --train-batches 20 --output ../results/pytorch.json
```

The JSON report includes per-model metrics for:

- training time per epoch and total training time
- CPU/GPU utilization samples and peak memory
- gradient computation time and data loading overhead
- inference latency for a single sample
- throughput for batch sizes 1, 8, 32, and 128
- warm-up versus steady-state inference timing
