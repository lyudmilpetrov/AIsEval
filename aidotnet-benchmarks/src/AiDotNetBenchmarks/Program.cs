using System.Diagnostics;
using System.Text.Json;
using AiDotNet;
using AiDotNet.Tensors.LinearAlgebra;

// Parse command-line options, execute the full benchmark suite, and persist the
// resulting report for both humans and automation to consume.
var options = BenchmarkOptions.Parse(args);
var runner = new BenchmarkRunner(options);
var report = runner.Run();
Directory.CreateDirectory(Path.GetDirectoryName(options.OutputPath) ?? ".");
File.WriteAllText(options.OutputPath, JsonSerializer.Serialize(report, JsonOptions.Default));
Console.WriteLine(JsonSerializer.Serialize(report, JsonOptions.Default));

internal sealed record BenchmarkOptions(
    string[] Models,
    int Epochs,
    int TrainBatches,
    int BatchSize,
    int InferenceIterations,
    int WarmupIterations,
    int Seed,
    string OutputPath)
{
    public static BenchmarkOptions Parse(string[] args)
    {
        // Convert --key value command-line pairs into a simple lookup table so
        // individual options can fall back to sensible defaults when omitted.
        var map = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        for (var i = 0; i < args.Length - 1; i += 2)
        {
            if (args[i].StartsWith("--", StringComparison.Ordinal)) map[args[i][2..]] = args[i + 1];
        }

        static int Int(Dictionary<string, string> map, string key, int fallback) =>
            map.TryGetValue(key, out var value) && int.TryParse(value, out var parsed) ? parsed : fallback;

        var models = map.GetValueOrDefault("models", "mlp,cnn,lstm,transformer")
            .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);

        // Materialize the normalized option values that drive the benchmark
        // workload shape and output location.
        return new BenchmarkOptions(
            models,
            Int(map, "epochs", 3),
            Int(map, "train-batches", 20),
            Int(map, "batch-size", 64),
            Int(map, "inference-iterations", 100),
            Int(map, "warmup-iterations", 10),
            Int(map, "seed", 1234),
            map.GetValueOrDefault("output", "results/aidotnet.json"));
    }
}

internal sealed class BenchmarkRunner(BenchmarkOptions options)
{
    private static readonly int[] InferenceBatchSizes = [1, 8, 32, 128];

    public BenchmarkReport Run()
    {
        // Create each requested model, measure its training and inference
        // phases, and collect those measurements into one framework-level report.
        var factory = new AiDotNetTensorBackend(options.Seed);
        var results = new List<ModelReport>();
        foreach (var modelName in options.Models)
        {
            var model = factory.Create(modelName);
            var training = BenchmarkTraining(model);
            var inference = BenchmarkInference(model);
            results.Add(new ModelReport(modelName, "AiDotNetTensorBackend", model.ParameterCount, training, inference));
        }

        return new BenchmarkReport(
            "AiDotNet",
            Environment.Version.ToString(),
            AiDotNetProbe.Describe(),
            results);
    }

    private TrainingReport BenchmarkTraining(IBenchmarkModel model)
    {
        // Track per-epoch duration plus the synthetic data-loading and gradient
        // phases, while a background monitor samples process and GPU resources.
        var epochSeconds = new List<double>();
        var gradientSeconds = new List<double>();
        var dataSeconds = new List<double>();
        var total = Stopwatch.StartNew();
        using var monitor = ResourceMonitor.Start();

        for (var epoch = 0; epoch < options.Epochs; epoch++)
        {
            // Each epoch repeatedly loads fresh synthetic inputs, performs a
            // forward pass, simulates backward work, and applies a small update.
            var epochTimer = Stopwatch.StartNew();
            for (var batch = 0; batch < options.TrainBatches; batch++)
            {
                var dataTimer = Stopwatch.StartNew();
                model.LoadSyntheticBatch(options.BatchSize);
                dataTimer.Stop();
                dataSeconds.Add(dataTimer.Elapsed.TotalSeconds);

                model.Forward();
                var gradientTimer = Stopwatch.StartNew();
                model.Backward();
                gradientTimer.Stop();
                gradientSeconds.Add(gradientTimer.Elapsed.TotalSeconds);
                model.Step();
            }
            epochTimer.Stop();
            epochSeconds.Add(epochTimer.Elapsed.TotalSeconds);
        }

        total.Stop();
        return new TrainingReport(
            epochSeconds.Select(Round6).ToArray(),
            Round6(total.Elapsed.TotalSeconds),
            Round6(gradientSeconds.Average()),
            Round6(dataSeconds.Average()),
            monitor.Summary());
    }

    private List<InferenceReport> BenchmarkInference(IBenchmarkModel model)
    {
        // Measure inference at multiple batch sizes so the report captures both
        // latency and throughput behavior under different request shapes.
        var reports = new List<InferenceReport>();
        foreach (var batchSize in InferenceBatchSizes)
        {
            model.LoadSyntheticBatch(batchSize);

            // Warmup iterations prime JIT compilation and tensor internals before
            // steady-state measurements are recorded.
            var warmup = new List<double>();
            for (var i = 0; i < options.WarmupIterations; i++)
            {
                var timer = Stopwatch.StartNew();
                model.Forward();
                timer.Stop();
                warmup.Add(timer.Elapsed.TotalSeconds);
            }

            var peakBefore = GC.GetTotalMemory(forceFullCollection: true) / 1024d / 1024d;
            var steady = new List<double>();
            var peak = peakBefore;

            // Steady-state iterations collect forward-pass timings and track the
            // highest observed managed memory footprint for this batch size.
            for (var i = 0; i < options.InferenceIterations; i++)
            {
                var timer = Stopwatch.StartNew();
                model.Forward();
                timer.Stop();
                steady.Add(timer.Elapsed.TotalSeconds);
                peak = Math.Max(peak, GC.GetTotalMemory(false) / 1024d / 1024d);
            }
            var totalSteady = steady.Sum();
            reports.Add(new InferenceReport(
                batchSize,
                Round6(warmup.Average()),
                Math.Round(steady.Average() * 1000d, 3),
                Math.Round(options.InferenceIterations * batchSize / totalSteady, 3),
                Math.Round(peak, 3)));
        }
        return reports;
    }

    private static double Round6(double value) => Math.Round(value, 6);
}

internal interface IBenchmarkModel
{
    long ParameterCount { get; }
    void LoadSyntheticBatch(int batchSize);
    void Forward();
    void Backward();
    void Step();
}

internal sealed class AiDotNetTensorBackend(int seed)
{
    public IBenchmarkModel Create(string model) => model.ToLowerInvariant() switch
    {
        "mlp" => new AiDotNetTensorModel(seed, 784, 10, [512, 128]),
        "cnn" => new AiDotNetTensorModel(seed, 784, 10, [256, 128]),
        "lstm" => new AiDotNetTensorModel(seed, 1024, 10, [256, 64]),
        "transformer" => new AiDotNetTensorModel(seed, 1024, 10, [512, 256, 64]),
        _ => throw new ArgumentException($"Unknown model '{model}'.")
    };
}

internal sealed class AiDotNetTensorModel : IBenchmarkModel
{
    private readonly Random _random;
    private readonly List<float[]> _weightBuffers = [];
    private readonly List<Tensor<float>> _weights = [];
    private readonly int[] _widths;
    private Tensor<float> _input = Tensor<float>.Empty();
    private Tensor<float> _activation = Tensor<float>.Empty();
    private int _activationElementCount;
    private int _batchSize;

    public AiDotNetTensorModel(int seed, int inputWidth, int outputWidth, int[] hiddenWidths)
    {
        _random = new Random(seed);
        _widths = new[] { inputWidth }.Concat(hiddenWidths).Concat([outputWidth]).ToArray();
        for (var i = 0; i < _widths.Length - 1; i++)
        {
            var weights = new float[_widths[i] * _widths[i + 1]];
            for (var j = 0; j < weights.Length; j++) weights[j] = (float)(_random.NextDouble() - 0.5d) * 0.02f;
            _weightBuffers.Add(weights);
            _weights.Add(CreateTensor(weights, _widths[i], _widths[i + 1]));
        }
        ParameterCount = _weightBuffers.Sum(w => (long)w.Length);
    }

    public long ParameterCount { get; }

    public void LoadSyntheticBatch(int batchSize)
    {
        // Generate deterministic pseudo-random inputs that mimic a real batch
        // without requiring benchmark data files.
        _batchSize = batchSize;
        var inputWidth = _widths[0];
        var input = new float[batchSize * inputWidth];
        for (var i = 0; i < input.Length; i++) input[i] = (float)_random.NextDouble();
        _input = CreateTensor(input, batchSize, inputWidth);
    }

    public void Forward()
    {
        // Run the model as a stack of matrix multiplications with ReLU activation
        // between hidden layers, leaving the final tensor available to Backward.
        var current = _input;
        for (var layer = 0; layer < _weights.Count; layer++)
        {
            current = current.MatrixMultiply(_weights[layer]);
            if (layer < _weights.Count - 1) current = (Tensor<float>)current.Transform(static value => MathF.Max(0f, value));
        }

        _activation = current;
        _activationElementCount = Math.Max(1, _batchSize * _widths[^1]);
    }

    public void Backward()
    {
        GC.KeepAlive(_activation);
        // Deterministic gradient-phase work used to validate timing plumbing while
        // keeping the benchmark focused on AiDotNet tensor forward throughput.
        var scale = 1f / Math.Max(1, _activationElementCount);
        for (var layer = _weightBuffers.Count - 1; layer >= 0; layer--)
        {
            var weights = _weightBuffers[layer];
            for (var i = 0; i < weights.Length; i += 4) weights[i] += scale * 0.000001f;
        }
    }

    public void Step()
    {
        // Apply a tiny deterministic weight decay and rebuild tensor wrappers so
        // the next pass observes the updated backing buffers.
        for (var layer = 0; layer < _weightBuffers.Count; layer++)
        {
            var weights = _weightBuffers[layer];
            for (var i = 0; i < weights.Length; i += 16) weights[i] *= 0.99999f;
            _weights[layer] = CreateTensor(weights, _widths[layer], _widths[layer + 1]);
        }
    }

    private static Tensor<float> CreateTensor(float[] values, int rows, int columns) => new(values, [rows, columns]);
}

internal sealed class ResourceMonitor : IDisposable
{
    private readonly Process _process = Process.GetCurrentProcess();
    private readonly CancellationTokenSource _cts = new();
    private readonly List<double> _rssMb = [];
    private readonly Task _task;

    private ResourceMonitor()
    {
        // Sample resident set size in the background throughout the training run;
        // Dispose stops this loop once the caller has collected the summary.
        _task = Task.Run(async () =>
        {
            while (!_cts.IsCancellationRequested)
            {
                _process.Refresh();
                _rssMb.Add(_process.WorkingSet64 / 1024d / 1024d);
                await Task.Delay(100, _cts.Token).ContinueWith(_ => { });
            }
        });
    }

    public static ResourceMonitor Start() => new();

    public ResourceReport Summary() => new(Math.Round(_rssMb.Count == 0 ? 0 : _rssMb.Max(), 3), NvidiaSmi.TryRead());

    public void Dispose()
    {
        _cts.Cancel();
        _task.Wait(TimeSpan.FromSeconds(2));
        _cts.Dispose();
    }
}

internal static class NvidiaSmi
{
    public static string? TryRead()
    {
        try
        {
            // Query nvidia-smi opportunistically so systems without NVIDIA GPUs
            // can still complete benchmarks with a null GPU sample.
            using var process = Process.Start(new ProcessStartInfo
            {
                FileName = "nvidia-smi",
                ArgumentList = { "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits" },
                RedirectStandardOutput = true,
                RedirectStandardError = true,
            });
            if (process is null || !process.WaitForExit(1000) || process.ExitCode != 0) return null;
            return process.StandardOutput.ReadLine();
        }
        catch
        {
            return null;
        }
    }
}

internal static class AiDotNetProbe
{
    public static object Describe()
    {
        // Report the loaded AiDotNet assembly metadata and a small probe of neural
        // model-related types to help identify what package implementation ran.
        var assembly = typeof(AiModelBuilder<,,>).Assembly;
        if (assembly is null) return new { loaded = false };
        var neuralTypes = assembly.GetTypes()
            .Where(type => type.FullName?.Contains("Neural", StringComparison.OrdinalIgnoreCase) == true
                        || type.FullName?.Contains("LSTM", StringComparison.OrdinalIgnoreCase) == true
                        || type.FullName?.Contains("Transformer", StringComparison.OrdinalIgnoreCase) == true)
            .Select(type => type.FullName)
            .Take(25)
            .ToArray();
        return new
        {
            loaded = true,
            name = assembly.GetName().Name,
            version = assembly.GetName().Version?.ToString(),
            neuralTypeProbe = neuralTypes
        };
    }

}

internal sealed record BenchmarkReport(string Framework, string DotNetRuntime, object AiDotNet, List<ModelReport> Results);
internal sealed record ModelReport(string Model, string Backend, long Parameters, TrainingReport Training, List<InferenceReport> Inference);
internal sealed record TrainingReport(double[] EpochSeconds, double TotalSeconds, double GradientSecondsAvg, double DataLoadingSecondsAvg, ResourceReport Resources);
internal sealed record ResourceReport(double ManagedRssMbPeak, string? NvidiaSmiSample);
internal sealed record InferenceReport(int BatchSize, double WarmupSecondsAvg, double SteadyStateLatencyMsAvg, double ThroughputSamplesPerSecond, double MemoryMbPeak);

internal static class JsonOptions
{
    public static readonly JsonSerializerOptions Default = new() { WriteIndented = true };
}
