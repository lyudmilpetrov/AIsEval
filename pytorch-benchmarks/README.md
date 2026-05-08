# PyTorch Benchmarks

Native Python benchmark project for PyTorch. It runs synthetic, reproducible MLP,
CNN, LSTM, and Transformer workloads and records training and inference metrics.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Install the PyTorch wheel that matches your hardware first if you need a specific
CUDA build: <https://pytorch.org/get-started/locally/>.

## Run

```bash
pytorch-bench --models mlp,cnn,lstm,transformer --device auto --epochs 3 --train-batches 20 --output ../results/pytorch.json
```

The JSON report includes per-model metrics for:

- training time per epoch and total training time
- CPU/GPU utilization samples and peak memory
- gradient computation time and data loading overhead
- inference latency for a single sample
- throughput for batch sizes 1, 8, 32, and 128
- warm-up versus steady-state inference timing
