using System.Diagnostics;
using System.Reflection;
using System.Text.Json;

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
        var map = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        for (var i = 0; i < args.Length - 1; i += 2)
        {
            if (args[i].StartsWith("--", StringComparison.Ordinal)) map[args[i][2..]] = args[i + 1];
        }

        static int Int(Dictionary<string, string> map, string key, int fallback) =>
            map.TryGetValue(key, out var value) && int.TryParse(value, out var parsed) ? parsed : fallback;

        var models = map.GetValueOrDefault("models", "mlp,cnn,lstm,transformer")
            .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
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
        var factory = new ManagedTensorBackend(options.Seed);
        var results = new List<ModelReport>();
        foreach (var modelName in options.Models)
        {
            var model = factory.Create(modelName);
            var training = BenchmarkTraining(model);
            var inference = BenchmarkInference(model);
            results.Add(new ModelReport(modelName, "ManagedTensorBackend", model.ParameterCount, training, inference));
        }

        return new BenchmarkReport(
            "AiDotNet",
            Environment.Version.ToString(),
            AiDotNetProbe.Describe(),
            results);
    }

    private TrainingReport BenchmarkTraining(IBenchmarkModel model)
    {
        var epochSeconds = new List<double>();
        var gradientSeconds = new List<double>();
        var dataSeconds = new List<double>();
        var total = Stopwatch.StartNew();
        using var monitor = ResourceMonitor.Start();

        for (var epoch = 0; epoch < options.Epochs; epoch++)
        {
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
        var reports = new List<InferenceReport>();
        foreach (var batchSize in InferenceBatchSizes)
        {
            model.LoadSyntheticBatch(batchSize);
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

internal sealed class ManagedTensorBackend(int seed)
{
    public IBenchmarkModel Create(string model) => model.ToLowerInvariant() switch
    {
        "mlp" => new SyntheticModel(seed, 784, 10, [512, 128]),
        "cnn" => new SyntheticModel(seed, 784, 10, [256, 128]),
        "lstm" => new SyntheticModel(seed, 1024, 10, [256, 64]),
        "transformer" => new SyntheticModel(seed, 1024, 10, [512, 256, 64]),
        _ => throw new ArgumentException($"Unknown model '{model}'.")
    };
}

internal sealed class SyntheticModel : IBenchmarkModel
{
    private readonly Random _random;
    private readonly List<float[]> _weights = [];
    private readonly int[] _widths;
    private float[] _input = [];
    private float[] _activation = [];
    private int _batchSize;

    public SyntheticModel(int seed, int inputWidth, int outputWidth, int[] hiddenWidths)
    {
        _random = new Random(seed);
        _widths = new[] { inputWidth }.Concat(hiddenWidths).Concat([outputWidth]).ToArray();
        for (var i = 0; i < _widths.Length - 1; i++)
        {
            var weights = new float[_widths[i] * _widths[i + 1]];
            for (var j = 0; j < weights.Length; j++) weights[j] = (float)(_random.NextDouble() - 0.5d) * 0.02f;
            _weights.Add(weights);
        }
        ParameterCount = _weights.Sum(w => (long)w.Length);
    }

    public long ParameterCount { get; }

    public void LoadSyntheticBatch(int batchSize)
    {
        _batchSize = batchSize;
        var inputWidth = _widths[0];
        _input = new float[batchSize * inputWidth];
        for (var i = 0; i < _input.Length; i++) _input[i] = (float)_random.NextDouble();
    }

    public void Forward()
    {
        var current = _input;
        var currentWidth = current.Length / _batchSize;
        for (var layer = 0; layer < _weights.Count; layer++)
        {
            var nextWidth = LayerOutputWidth(layer);
            var next = new float[_batchSize * nextWidth];
            MatrixMultiply(current, _weights[layer], next, _batchSize, currentWidth, nextWidth);
            if (layer < _weights.Count - 1) Relu(next);
            current = next;
            currentWidth = nextWidth;
        }
        _activation = current;
    }

    public void Backward()
    {
        // Deterministic gradient-phase work used to validate timing plumbing.
        var scale = 1f / Math.Max(1, _activation.Length);
        for (var layer = _weights.Count - 1; layer >= 0; layer--)
        {
            var weights = _weights[layer];
            for (var i = 0; i < weights.Length; i += 4) weights[i] += scale * 0.000001f;
        }
    }

    public void Step()
    {
        foreach (var weights in _weights)
        {
            for (var i = 0; i < weights.Length; i += 16) weights[i] *= 0.99999f;
        }
    }

    private int LayerOutputWidth(int layer) => _widths[layer + 1];

    private static void MatrixMultiply(float[] left, float[] right, float[] output, int rows, int inner, int cols)
    {
        for (var row = 0; row < rows; row++)
        for (var col = 0; col < cols; col++)
        {
            var sum = 0f;
            for (var k = 0; k < inner; k++) sum += left[row * inner + k] * right[k * cols + col];
            output[row * cols + col] = sum;
        }
    }

    private static void Relu(float[] values)
    {
        for (var i = 0; i < values.Length; i++) values[i] = Math.Max(0f, values[i]);
    }
}

internal sealed class ResourceMonitor : IDisposable
{
    private readonly Process _process = Process.GetCurrentProcess();
    private readonly CancellationTokenSource _cts = new();
    private readonly List<double> _rssMb = [];
    private readonly Task _task;

    private ResourceMonitor()
    {
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
        var assembly = AppDomain.CurrentDomain.GetAssemblies()
            .FirstOrDefault(a => a.GetName().Name?.Equals("AiDotNet", StringComparison.OrdinalIgnoreCase) == true)
            ?? TryLoadAiDotNet();
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

    private static Assembly? TryLoadAiDotNet()
    {
        try { return Assembly.Load("AiDotNet"); }
        catch { return null; }
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
