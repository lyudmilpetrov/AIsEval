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

To exit the Python environment, run `deactivate` in the terminal.

### Windows Git Bash

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

### Windows PowerShell

PowerShell does not have a `source` command. Activate the environment with the
PowerShell activation script, or call the virtual environment's Python executable
directly.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

If PowerShell blocks `Activate.ps1` because of execution policy, you can skip
activation and run the commands through the virtual environment's Python
executable instead:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .
```

Install the PyTorch wheel that matches your hardware first if you need a specific
CUDA build: <https://pytorch.org/get-started/locally/>.


## Web API

The project also exposes a FastAPI web API with controller-style routers. Start
the API from the `pytorch-benchmarks/` directory after installing the package:

```bash
pytorch-bench-api
```

You can also run it directly with Uvicorn:

```bash
python -m uvicorn pytorch_benchmarks.api:app --host 0.0.0.0 --port 8000
```

The editable install pulls in FastAPI, Uvicorn, and `python-multipart`. If a
manual or partial environment setup raises `The python-multipart library must be
installed to use form parsing`, run this command inside the same active virtual
environment that starts Uvicorn:

```bash
python -m pip install -e .
# or, as a targeted repair:
python -m pip install python-multipart
```

The first controller endpoint is a regression health check:

```bash
curl http://localhost:8000/api/Regression/Test
# ping
```

## Run

```bash
pytorch-bench --models mlp,cnn,lstm,transformer --device auto --epochs 3 --train-batches 20 --output ../results/pytorch.json
```

PowerShell users can also run the generated executable from the virtual
environment explicitly, which avoids relying on PATH updates:

```powershell
.\.venv\Scripts\pytorch-bench.exe --models mlp,cnn,lstm,transformer --device auto --epochs 3 --train-batches 20 --output ..\results\pytorch.json
```

If `pytorch-bench` prints `command not found` or PowerShell reports that
`pytorch-bench` is not recognized, the package has not been installed into the
currently active environment or that environment is not on your PATH. The pip
message `Defaulting to user installation` also means the virtual environment was
not active when pip ran. Run these checks from `pytorch-benchmarks/`:

```bash
source .venv/bin/activate        # macOS/Linux
# or: source .venv/Scripts/activate      # Windows Git Bash
# or: .\.venv\Scripts\Activate.ps1   # Windows PowerShell
python -m pip install -e .
python -m pip show pytorch-benchmarks
python -c "import shutil; print(shutil.which('pytorch-bench'))"
```

On Windows PowerShell, use these equivalent checks:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m pip show pytorch-benchmarks
python -c "import shutil; print(shutil.which('pytorch-bench'))"
.\.venv\Scripts\pytorch-bench.exe --help
```

As an install-free fallback from the project directory, run the package directory
with Python directly:

```bash
python src/pytorch_benchmarks --models mlp,cnn,lstm,transformer --device auto --epochs 3 --train-batches 20 --output ../results/pytorch.json
```

If the package was installed but the generated script directory is not on PATH,
module execution also works without the `pytorch-bench` executable name:

```powershell
python -m pytorch_benchmarks --models mlp,cnn,lstm,transformer --device auto --epochs 3 --train-batches 20 --output ..\results\pytorch.json
```

The JSON report includes per-model metrics for:

- training time per epoch and total training time
- CPU/GPU utilization samples and peak memory
- gradient computation time and data loading overhead
- inference latency for a single sample
- throughput for batch sizes 1, 8, 32, and 128
- warm-up versus steady-state inference timing

### CSV Regression Prediction Endpoint

`POST /api/Regression/SimpleRegression?UseGPU=false` trains a native PyTorch
linear regression model from uploaded CSV data and returns JSON predictions.
`POST /api/Regression/Predict?UseGPU=false` is also supported as a compatibility
alias. Send a multipart request with:

- `features`: `features.csv`, where each row contains numeric feature columns
  followed by the numeric target value in the last column.
- `tests`: `tests.csv`, where each row contains only the numeric feature columns
  to predict.

Both files may include a single header row. The `timings` object is reported in
milliseconds (`timing_unit: "milliseconds"`), and the response also includes a
`Server-Timing: app;dur=...` header for comparing server-side duration with
Postman's client-side round-trip time. The endpoint currently runs on CPU only
to match the AiDotNet regression endpoint; `UseGPU=true` is echoed as
`gpuRequested: true`, but `gpuUsed` remains `false`.

```bash
curl -X POST "http://localhost:8000/api/Regression/SimpleRegression?UseGPU=false" \
  -F "features=@features.csv" \
  -F "tests=@tests.csv"
```

#### Postman setup

If Postman returns `No multipart/form-data body was received`, the request did
not reach FastAPI with a generated multipart `Content-Type` header. Use this
checklist:

1. Open **Headers** and delete or disable any manually added `Content-Type`
   header, including `application/json` or `multipart/form-data`. Postman must
   generate `multipart/form-data; boundary=...` automatically from the Body tab.
2. Open **Body**, select **form-data**, and keep both rows checked.
3. Add two rows with exact keys `features` and `tests`. Change each row type
   from **Text** to **File**, then choose `features.csv` and `tests.csv`.
4. If Postman shows a yellow warning icon on either file, reselect the file from
   disk or add its folder under **Settings > Working Directory** so the desktop
   agent can read it.
5. Send the request to
   `http://localhost:8000/api/Regression/SimpleRegression?UseGPU=false`.
