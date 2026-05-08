#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/results"

(
  cd "$ROOT/pytorch-benchmarks"
  python -m pytorch_benchmarks --models mlp,cnn,lstm,transformer --output "$ROOT/results/pytorch.json"
)

(
  cd "$ROOT/aidotnet-benchmarks"
  dotnet run -c Release -- --models mlp,cnn,lstm,transformer --output "$ROOT/results/aidotnet.json"
)
