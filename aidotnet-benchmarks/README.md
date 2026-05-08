# AiDotNet Benchmarks

C# benchmark project for evaluating AiDotNet-side workloads. The project pins the
AiDotNet NuGet package and provides a reproducible benchmark runner with the same
model families and metric names as the PyTorch project.

> Note: the runner uses `AiDotNetTensorBackend`, which builds benchmark inputs
> and weights with `AiDotNet.Tensors.LinearAlgebra.Tensor<float>` and executes
> forward passes through AiDotNet tensor operations so the reported numbers cover
> AiDotNet runtime work rather than a local reference implementation.

## Run

```bash
dotnet restore
dotnet run -c Release -- --models mlp,cnn,lstm,transformer --epochs 3 --train-batches 20 --output ../results/aidotnet.json
```

The runner records:

- training time per epoch and total training time
- CPU/GPU utilization samples and managed memory peak
- gradient phase time and data loading overhead
- inference latency for a single sample
- throughput for batch sizes 1, 8, 32, and 128
- warm-up versus steady-state inference timing
- AiDotNet assembly identity and a neural-network type probe
