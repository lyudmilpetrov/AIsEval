# AIsEval: PyTorch vs AiDotNet Benchmark Projects

This repository contains two separate benchmark projects designed to compare
PyTorch and AiDotNet under the same high-level workload families and metric
schema.

## Projects

| Project | Runtime | Purpose |
| --- | --- | --- |
| `pytorch-benchmarks/` | Native Python | Runs real PyTorch MLP, CNN, LSTM, and Transformer benchmarks. |
| `aidotnet-benchmarks/` | C# / .NET | Evaluates AiDotNet package identity and provides C# benchmark plumbing with a swappable model backend. |

## Benchmark Coverage

### Training

- Training time per epoch
- Total training time
- CPU/GPU utilization samples
- Memory usage / footprint
- Gradient computation time
- Data-loading overhead

### Inference

- Single-sample latency via batch size 1
- Throughput for batch sizes 1, 8, 32, and 128
- Warm-up versus steady-state timing
- Memory footprint

### Model Families

- MLP: baseline sanity check
- CNN: vision workload
- RNN/LSTM: sequential workload
- Transformer: modern attention-style workload

## Quick Start

### PyTorch

```bash
cd pytorch-benchmarks
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# or: source .venv/Scripts/activate      # Windows Git Bash
# or: .\.venv\Scripts\Activate.ps1  # Windows PowerShell
python -m pip install --upgrade pip
python -m pip install -e .
pytorch-bench --models mlp,cnn,lstm,transformer --output ../results/pytorch.json
```

If `pytorch-bench` is not found, confirm that the virtual environment is active
and rerun `python -m pip install -e .` from `pytorch-benchmarks/`. You can also
run `python src/pytorch_benchmarks --models mlp,cnn,lstm,transformer --output ../results/pytorch.json`
from the same directory as an install-free fallback.

### AiDotNet

```bash
cd aidotnet-benchmarks
dotnet restore
dotnet run -c Release -- --models mlp,cnn,lstm,transformer --output ../results/aidotnet.json
```

## Result Files

Both projects emit indented JSON reports under `results/` by default. Keep raw
reports per hardware profile, for example:

```text
results/
  pytorch-rtx4090-cuda.json
  aidotnet-rtx4090-cuda.json
  pytorch-cpu.json
  aidotnet-cpu.json
```

## Fairness Notes

- Use identical hardware, power settings, process isolation, and batch sizes.
- Run release/optimized builds only (`dotnet run -c Release`, no Python debug tooling).
- Discard the first run if you want to eliminate package JIT/import cache effects.
- The C# project currently includes a managed reference backend for validating
  benchmark infrastructure. Add a production AiDotNet implementation behind
  `IBenchmarkModel` before publishing framework-to-framework claims.
