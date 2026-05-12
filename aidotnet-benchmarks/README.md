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

## CSV Regression API

The web API exposes `POST /api/Regression/SimpleRegression?UseGPU=false` for the
AiDotNet quick-start style regression workflow. The legacy
`POST /api/Regression/Predict?UseGPU=false` route is also supported. Upload a
`multipart/form-data` request with:

- `features`: upload `features.csv`, where each row contains numeric feature
  columns followed by the numeric target value in the last column.
- `tests`: upload `tests.csv`, where each row contains only the numeric feature
  columns to predict.

For quick Postman tests, the same `features` and `tests` form-data keys may be
sent as `Text` values containing pasted CSV content instead of file uploads.

Both CSV files may include a single header row. The endpoint currently runs on
CPU only; `UseGPU=true` is echoed as `gpuRequested: true`, but `gpuUsed`
remains `false`. Small training uploads with fewer than seven data rows use a
least-squares prediction path to avoid AiDotNet's internal validation split
creating an empty validation matrix.

```bash
curl -X POST "http://localhost:5000/api/Regression/SimpleRegression?UseGPU=false" \
  -F "features=@features.csv" \
  -F "tests=@tests.csv"
```

If Postman still returns `No multipart/form-data body was received`, check the
request headers and remove any manually configured `Content-Type` header so
Postman can generate the required `multipart/form-data; boundary=...` value. A
yellow warning icon next to a selected file means Postman cannot read that file;
reselect `features.csv` and `tests.csv`, or switch those rows from `File` to
`Text` and paste the CSV contents.

The response contains JSON metadata plus one prediction per row in `tests.csv`.
The `timings` object is reported in milliseconds (`timing_unit:
"milliseconds"`), and the response also includes a `Server-Timing:
app;dur=...` header for comparing server-side duration with Postman's
client-side round-trip time.
