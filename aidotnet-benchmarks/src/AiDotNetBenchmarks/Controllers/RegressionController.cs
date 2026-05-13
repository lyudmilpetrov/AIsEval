using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;
using Microsoft.AspNetCore.Mvc;
using System.Diagnostics;
using System.Globalization;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json.Serialization;

namespace AiDotNetBenchmarks.Controllers;

[ApiController]
[Route("api/[controller]")]
public sealed class RegressionController : ControllerBase
{
    private const int MinimumRowsForAiDotNetValidationSplit = 7;

    [HttpGet("Test")]
    public ActionResult<string> Test() => "ping";
     
    [HttpPost("SimpleRegression")]
    [RequestSizeLimit(long.MaxValue)]
    public async Task<ActionResult<CsvRegressionResponse>> SimpleRegression([FromQuery] bool UseGPU = false)
    {
        var measurement = RequestPerformanceMeasurement.Start();
        var preprocessTimer = Stopwatch.StartNew();

        if (!Request.HasFormContentType)
        {
            return BadRequest(new
            {
                error = "No multipart/form-data body was received. In Postman, remove any manually configured Content-Type header, keep Body set to form-data, reselect both CSV files if a yellow warning icon is shown, and send fields named features and tests.",
                receivedContentType = Request.ContentType ?? "<missing>"
            });
        }

        var form = await Request.ReadFormAsync(HttpContext.RequestAborted);
        var featuresInput = FindCsvInput(form, "features", "features.csv");
        var testsInput = FindCsvInput(form, "tests", "tests.csv");

        if (featuresInput is null || testsInput is null)
        {
            return BadRequest(new
            {
                error = "Upload multipart/form-data files named features.csv and tests.csv, or paste CSV text into form fields named features and tests. In Postman, yellow warning icons beside selected files mean the files must be reselected before sending.",
                receivedFiles = form.Files.Select(file => new { field = file.Name, fileName = file.FileName }).ToArray(),
                receivedFields = form.Keys.ToArray()
            });
        }

        List<double[]> trainingRows;
        List<double[]> testRows;
        try
        {
            trainingRows = await CsvRegressionReader.ReadRowsAsync(featuresInput, HttpContext.RequestAborted);
            testRows = await CsvRegressionReader.ReadRowsAsync(testsInput, HttpContext.RequestAborted);
        }
        catch (FormatException exception)
        {
            return BadRequest(new { error = exception.Message });
        }

        if (trainingRows.Count == 0)
        {
            return BadRequest(new { error = "features.csv must contain at least one training row." });
        }

        if (testRows.Count == 0)
        {
            return BadRequest(new { error = "tests.csv must contain at least one test row." });
        }

        if (trainingRows[0].Length < 2)
        {
            return BadRequest(new { error = "features.csv must contain one or more feature columns followed by a target column." });
        }

        var featureCount = trainingRows[0].Length - 1;
        if (trainingRows.Any(row => row.Length != featureCount + 1))
        {
            return BadRequest(new { error = "Every features.csv row must have the same number of columns." });
        }

        if (testRows.Any(row => row.Length != featureCount))
        {
            return BadRequest(new { error = $"Every tests.csv row must contain exactly {featureCount} feature columns." });
        }

        var features = ToFeatureArray(trainingRows, featureCount);
        var labels = ToLabelArray(trainingRows, featureCount);
        var testData = ToMatrix(testRows, featureCount);
        preprocessTimer.Stop();

        var regressionModelName = GetRegressionModelName(featureCount);
        var inferenceTimer = Stopwatch.StartNew();
        var predictedValues = await PredictWithAiDotNetAsync(features, labels, testData, featureCount, HttpContext.RequestAborted);
        inferenceTimer.Stop();

        var postprocessTimer = Stopwatch.StartNew();
        var predictions = Enumerable.Range(0, testRows.Count)
            .Select(index => new CsvPrediction(index, predictedValues[index]))
            .ToArray();
        postprocessTimer.Stop();

        var totalMs = measurement.ElapsedMilliseconds;
        Response.Headers["Server-Timing"] = $"app;dur={totalMs.ToString(CultureInfo.InvariantCulture)}";
        var gpu = UseGPU ? RegressionGpuProbe.Read() : RegressionGpuProbe.Empty();

        return Ok(new CsvRegressionResponse(
            predictions,
            new TimingMetrics(
                "milliseconds",
                totalMs,
                RoundMilliseconds(preprocessTimer.Elapsed.TotalMilliseconds),
                RoundMilliseconds(inferenceTimer.Elapsed.TotalMilliseconds),
                RoundMilliseconds(postprocessTimer.Elapsed.TotalMilliseconds)),
            measurement.ToCpuMetrics(totalMs),
            gpu,
            new SystemMetrics(
                RuntimeInformation.OSDescription,
                RuntimeInformation.FrameworkDescription,
                GetAiDotNetVersion(),
                "cpu",
                false),
            new ModelMetrics(
                testRows.Count,
                featureCount + 1,
                EstimateLinearRegressionFlops(trainingRows.Count, testRows.Count, featureCount)),
            "AiDotNet",
            regressionModelName,
            UseGPU,
            false,
            trainingRows.Count,
            testRows.Count,
            featureCount,
            predictions));
    }

    private static double RoundMilliseconds(double value) => Math.Round(value, 3);

    private static long EstimateLinearRegressionFlops(int trainingRows, int testRows, int featureCount)
    {
        var coefficientCount = featureCount + 1L;
        var normalEquationFlops = trainingRows * coefficientCount * coefficientCount * 2L;
        var solveFlops = 2L * coefficientCount * coefficientCount * coefficientCount / 3L;
        var inferenceFlops = testRows * coefficientCount * 2L;
        return normalEquationFlops + solveFlops + inferenceFlops;
    }

    private static string GetAiDotNetVersion() =>
        typeof(AiModelBuilder<,,>).Assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion
        ?? typeof(AiModelBuilder<,,>).Assembly.GetName().Version?.ToString()
        ?? "unknown";

    private static string GetRegressionModelName(int featureCount) =>
        featureCount == 1 ? "SimpleRegression" : "MultipleRegression";

    private static async Task<Vector<double>> PredictWithAiDotNetAsync(
        double[,] features,
        double[] labels,
        Matrix<double> testData,
        int featureCount,
        CancellationToken cancellationToken)
    {
        // AiDotNet currently performs an internal 70/15/15 train/validation/test
        // split. Datasets with fewer than seven rows produce a zero-row
        // validation matrix, so small CSV uploads use the least-squares fallback
        // below instead of entering the builder path. Select the AiDotNet
        // regression implementation to match the uploaded CSV shape so a
        // multi-column features.csv uses MultipleRegression instead of failing
        // SimpleRegression's single-feature validation.
        var loader = DataLoaders.FromArrays(features, labels);

        if (featureCount == 1)
        {
            var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
                .ConfigureDataLoader(loader)
                .ConfigureModel(new SimpleRegression<double>())
                .BuildAsync();

            return result.Predict(testData);
        }

        var multipleRegressionResult = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(new MultipleRegression<double>())
            .BuildAsync();

        return multipleRegressionResult.Predict(testData);
    }
       
    private static double[,] ToFeatureArray(IReadOnlyList<double[]> rows, int featureCount)
    {
        var features = new double[rows.Count, featureCount];
        for (var rowIndex = 0; rowIndex < rows.Count; rowIndex++)
        {
            for (var featureIndex = 0; featureIndex < featureCount; featureIndex++)
            {
                features[rowIndex, featureIndex] = rows[rowIndex][featureIndex];
            }
        }

        return features;
    }

    private static double[] ToLabelArray(IReadOnlyList<double[]> rows, int featureCount)
    {
        var labels = new double[rows.Count];
        for (var rowIndex = 0; rowIndex < rows.Count; rowIndex++)
        {
            labels[rowIndex] = rows[rowIndex][featureCount];
        }

        return labels;
    }

    private static Matrix<double> ToMatrix(IReadOnlyList<double[]> rows, int featureCount)
    {
        var matrix = new Matrix<double>(rows.Count, featureCount);
        for (var rowIndex = 0; rowIndex < rows.Count; rowIndex++)
        {
            for (var featureIndex = 0; featureIndex < featureCount; featureIndex++)
            {
                matrix[rowIndex, featureIndex] = rows[rowIndex][featureIndex];
            }
        }

        return matrix;
    }

    private static CsvRegressionInput? FindCsvInput(IFormCollection form, string fieldName, string fileName)
    {
        var file = form.Files.FirstOrDefault(file => string.Equals(file.Name, fieldName, StringComparison.OrdinalIgnoreCase))
            ?? form.Files.FirstOrDefault(file => string.Equals(file.Name, fileName, StringComparison.OrdinalIgnoreCase))
            ?? form.Files.FirstOrDefault(file => string.Equals(file.FileName, fileName, StringComparison.OrdinalIgnoreCase));

        if (file is not null)
        {
            return CsvRegressionInput.FromFile(file);
        }

        if (TryFindCsvText(form, fieldName, fileName, out var csvText))
        {
            return CsvRegressionInput.FromText(fileName, csvText);
        }

        return null;
    }

    private static bool TryFindCsvText(IFormCollection form, string fieldName, string fileName, out string csvText)
    {
        foreach (var key in new[] { fieldName, fileName })
        {
            if (form.TryGetValue(key, out var values))
            {
                csvText = values.FirstOrDefault(value => !string.IsNullOrWhiteSpace(value)) ?? string.Empty;
                return !string.IsNullOrWhiteSpace(csvText);
            }
        }

        csvText = string.Empty;
        return false;
    }
}

internal sealed record CsvRegressionInput(string FileName, IFormFile? File, string? Content)
{
    public static CsvRegressionInput FromFile(IFormFile file) =>
        new(file.FileName, file, null);

    public static CsvRegressionInput FromText(string fileName, string content) =>
        new(fileName, null, content);
}

public sealed record CsvRegressionResponse(
    [property: JsonPropertyName("prediction")] IReadOnlyList<CsvPrediction> Prediction,
    [property: JsonPropertyName("timings")] TimingMetrics Timings,
    [property: JsonPropertyName("cpu")] CpuMetrics Cpu,
    [property: JsonPropertyName("gpu")] GpuMetrics Gpu,
    [property: JsonPropertyName("system")] SystemMetrics System,
    [property: JsonPropertyName("model_metrics")] ModelMetrics ModelMetrics,
    string Framework,
    string Model,
    bool GpuRequested,
    bool GpuUsed,
    int TrainingRows,
    int TestRows,
    int FeatureCount,
    IReadOnlyList<CsvPrediction> Predictions);

public sealed record TimingMetrics(
    [property: JsonPropertyName("timing_unit")] string TimingUnit,
    [property: JsonPropertyName("total_ms")] double TotalMs,
    [property: JsonPropertyName("preprocess_ms")] double PreprocessMs,
    [property: JsonPropertyName("inference_ms")] double InferenceMs,
    [property: JsonPropertyName("postprocess_ms")] double PostprocessMs);

public sealed record CpuMetrics(
    [property: JsonPropertyName("usage_percent")] double? UsagePercent,
    [property: JsonPropertyName("memory_mb")] double MemoryMb,
    [property: JsonPropertyName("memory_before_mb")] double MemoryBeforeMb,
    [property: JsonPropertyName("memory_after_mb")] double MemoryAfterMb,
    [property: JsonPropertyName("threads")] int Threads,
    [property: JsonPropertyName("cpu_cycles")] long? CpuCycles,
    [property: JsonPropertyName("cpu_time_ms")] double CpuTimeMs,
    [property: JsonPropertyName("gc_collections")] IReadOnlyDictionary<string, int> GcCollections,
    [property: JsonPropertyName("gc_allocated_bytes")] long GcAllocatedBytes);

public sealed record GpuMetrics(
    [property: JsonPropertyName("name")] string? Name,
    [property: JsonPropertyName("utilization_percent")] double? UtilizationPercent,
    [property: JsonPropertyName("memory_allocated_mb")] double? MemoryAllocatedMb,
    [property: JsonPropertyName("memory_reserved_mb")] double? MemoryReservedMb,
    [property: JsonPropertyName("memory_peak_allocated_mb")] double? MemoryPeakAllocatedMb,
    [property: JsonPropertyName("memory_total_mb")] double? MemoryTotalMb,
    [property: JsonPropertyName("temperature_c")] double? TemperatureC,
    [property: JsonPropertyName("kernel_execution_ms")] double? KernelExecutionMs,
    [property: JsonPropertyName("tensor_core_utilization_percent")] double? TensorCoreUtilizationPercent,
    [property: JsonPropertyName("device_info")] string? DeviceInfo);

public sealed record SystemMetrics(
    [property: JsonPropertyName("os")] string Os,
    [property: JsonPropertyName("framework_version")] string FrameworkVersion,
    [property: JsonPropertyName("library_version")] string LibraryVersion,
    [property: JsonPropertyName("device")] string Device,
    [property: JsonPropertyName("mixed_precision")] bool MixedPrecision);

public sealed record ModelMetrics(
    [property: JsonPropertyName("batch_size")] int BatchSize,
    [property: JsonPropertyName("parameter_count")] long ParameterCount,
    [property: JsonPropertyName("flops_estimate")] long? FlopsEstimate);

public sealed record CsvPrediction(int RowIndex, double Prediction);

internal sealed class RequestPerformanceMeasurement
{
    private readonly long _startedAt;
    private readonly Process _process = Process.GetCurrentProcess();
    private readonly TimeSpan _startCpuTime;
    private readonly long _startMemoryBytes;
    private readonly long _startAllocatedBytes;
    private readonly int[] _startGcCollections;

    private RequestPerformanceMeasurement()
    {
        _process.Refresh();
        _startMemoryBytes = _process.WorkingSet64;
        _startAllocatedBytes = GC.GetTotalAllocatedBytes(precise: false);
        _startGcCollections = Enumerable.Range(0, GC.MaxGeneration + 1)
            .Select(GC.CollectionCount)
            .ToArray();
        _startCpuTime = _process.TotalProcessorTime;
        _startedAt = Stopwatch.GetTimestamp();
    }

    public static RequestPerformanceMeasurement Start() => new();

    public double ElapsedMilliseconds =>
        RoundElapsedMilliseconds(Stopwatch.GetTimestamp() - _startedAt);

    private static double RoundElapsedMilliseconds(long elapsedTicks) =>
        Math.Round(elapsedTicks * 1000d / Stopwatch.Frequency, 3);

    public CpuMetrics ToCpuMetrics(double elapsedMilliseconds)
    {
        _process.Refresh();
        var cpuTime = _process.TotalProcessorTime - _startCpuTime;
        var elapsedSeconds = Math.Max(elapsedMilliseconds / 1000d, 1e-9);
        var usage = cpuTime.TotalSeconds / elapsedSeconds / Math.Max(Environment.ProcessorCount, 1) * 100d;
        var gcCollections = Enumerable.Range(0, GC.MaxGeneration + 1)
            .ToDictionary(
                generation => $"gen{generation}",
                generation => GC.CollectionCount(generation) - _startGcCollections[generation]);

        return new CpuMetrics(
            Math.Round(Math.Max(0d, usage), 3),
            Math.Round(_process.WorkingSet64 / 1024d / 1024d, 3),
            Math.Round(_startMemoryBytes / 1024d / 1024d, 3),
            Math.Round(_process.WorkingSet64 / 1024d / 1024d, 3),
            _process.Threads.Count,
            null,
            Math.Round(cpuTime.TotalMilliseconds, 3),
            gcCollections,
            GC.GetTotalAllocatedBytes(precise: false) - _startAllocatedBytes);
    }
}

internal static class RegressionGpuProbe
{
    public static GpuMetrics Read()
    {
        try
        {
            using var process = Process.Start(new ProcessStartInfo
            {
                FileName = "nvidia-smi",
                RedirectStandardOutput = true,
                RedirectStandardError = true,
            }.WithArguments(
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits"));

            if (process is null || !process.WaitForExit(1000) || process.ExitCode != 0)
            {
                return Empty();
            }

            var line = process.StandardOutput.ReadLine();
            if (string.IsNullOrWhiteSpace(line)) return Empty();

            var parts = line.Split(',', StringSplitOptions.TrimEntries);
            if (parts.Length < 5) return Empty();

            return new GpuMetrics(
                parts[0],
                ParseNullableDouble(parts[1]),
                ParseNullableDouble(parts[2]),
                null,
                null,
                ParseNullableDouble(parts[3]),
                ParseNullableDouble(parts[4]),
                null,
                null,
                "NVIDIA CUDA device sampled by nvidia-smi; AiDotNet regression inference currently executes on CPU.");
        }
        catch
        {
            return Empty();
        }
    }

    public static GpuMetrics Empty() => new(null, null, null, null, null, null, null, null, null, null);

    private static double? ParseNullableDouble(string value) =>
        double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out var parsed) ? parsed : null;
}

internal static class ProcessStartInfoExtensions
{
    public static ProcessStartInfo WithArguments(this ProcessStartInfo startInfo, params string[] arguments)
    {
        foreach (var argument in arguments)
        {
            startInfo.ArgumentList.Add(argument);
        }

        return startInfo;
    }
}

internal static class CsvRegressionReader
{
    public static async Task<List<double[]>> ReadRowsAsync(CsvRegressionInput input, CancellationToken cancellationToken)
    {
        if (input.File is not null)
        {
            await using var stream = input.File.OpenReadStream();
            using var reader = new StreamReader(stream, Encoding.UTF8, detectEncodingFromByteOrderMarks: true);
            return await ReadRowsAsync(reader, input.FileName, cancellationToken);
        }

        using var stringReader = new StringReader(input.Content ?? string.Empty);
        return await ReadRowsAsync(stringReader, input.FileName, cancellationToken);
    }

    private static async Task<List<double[]>> ReadRowsAsync(TextReader reader, string fileName, CancellationToken cancellationToken)
    {
        var rows = new List<double[]>();
        var firstDataLine = true;

        while (await reader.ReadLineAsync(cancellationToken) is { } line)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (string.IsNullOrWhiteSpace(line)) continue;

            var fields = SplitCsvLine(line);
            if (fields.Count == 0 || fields.All(string.IsNullOrWhiteSpace)) continue;

            if (TryParseRow(fields, out var values))
            {
                rows.Add(values);
                firstDataLine = false;
                continue;
            }

            if (firstDataLine)
            {
                firstDataLine = false;
                continue;
            }

            throw new FormatException($"CSV file '{fileName}' contains a non-numeric data row: {line}");
        }

        return rows;
    }

    private static bool TryParseRow(IReadOnlyList<string> fields, out double[] values)
    {
        values = new double[fields.Count];
        for (var index = 0; index < fields.Count; index++)
        {
            if (!double.TryParse(
                fields[index],
                NumberStyles.Float | NumberStyles.AllowThousands,
                CultureInfo.InvariantCulture,
                out values[index]))
            {
                return false;
            }
        }

        return true;
    }

    private static List<string> SplitCsvLine(string line)
    {
        var fields = new List<string>();
        var field = new StringBuilder();
        var inQuotes = false;

        for (var index = 0; index < line.Length; index++)
        {
            var current = line[index];
            if (current == '"')
            {
                if (inQuotes && index + 1 < line.Length && line[index + 1] == '"')
                {
                    field.Append('"');
                    index++;
                }
                else
                {
                    inQuotes = !inQuotes;
                }
            }
            else if (current == ',' && !inQuotes)
            {
                fields.Add(field.ToString().Trim());
                field.Clear();
            }
            else
            {
                field.Append(current);
            }
        }

        fields.Add(field.ToString().Trim());
        return fields;
    }
}
