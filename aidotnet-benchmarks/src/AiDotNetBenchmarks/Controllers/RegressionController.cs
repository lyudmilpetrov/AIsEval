using AiDotNet;
using AiDotNet.Regression;
using Microsoft.AspNetCore.Mvc;
using System.Globalization;
using System.Text;

namespace AiDotNetBenchmarks.Controllers;

[ApiController]
[Route("api/[controller]")]
public sealed class RegressionController : ControllerBase
{
    [HttpGet("Test")]
    public ActionResult<string> Test() => "ping";

    [HttpPost("Predict")]
    [RequestSizeLimit(long.MaxValue)]
    public async Task<ActionResult<CsvRegressionResponse>> Predict([FromQuery] bool UseGPU = false)
    {
        var form = await Request.ReadFormAsync(HttpContext.RequestAborted);
        var featuresFile = FindCsvFile(form.Files, "features", "features.csv");
        var testsFile = FindCsvFile(form.Files, "tests", "tests.csv");

        if (featuresFile is null || testsFile is null)
        {
            return BadRequest(new
            {
                error = "Upload multipart/form-data files named features.csv and tests.csv, or use form fields named features and tests."
            });
        }

        List<double[]> trainingRows;
        List<double[]> testRows;
        try
        {
            trainingRows = await CsvRegressionReader.ReadRowsAsync(featuresFile, HttpContext.RequestAborted);
            testRows = await CsvRegressionReader.ReadRowsAsync(testsFile, HttpContext.RequestAborted);
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

        var features = trainingRows
            .Select(row => row.Take(featureCount).ToArray())
            .ToArray();
        var targets = trainingRows
            .Select(row => row[featureCount])
            .ToArray();
        var tests = testRows.ToArray();

        var modelBuilder = new AiModelBuilder<double, double[], double>()
            .ConfigureModel(new GradientBoostingRegression<double>(nEstimators: 100))
            .ConfigurePreprocessing();

        if (UseGPU)
        {
            modelBuilder = modelBuilder.ConfigureGpuAcceleration(new GpuAccelerationConfig
            {
                Enabled = true,
                DeviceId = 0
            });
        }

        var result = await modelBuilder.BuildAsync(features, targets);
        var predictions = tests
            .Select((row, index) => new CsvPrediction(index, result.Predict(row)))
            .ToArray();

        return Ok(new CsvRegressionResponse(
            "AiDotNet",
            "GradientBoostingRegression",
            UseGPU,
            UseGPU,
            trainingRows.Count,
            tests.Length,
            featureCount,
            predictions));
    }

    private static IFormFile? FindCsvFile(IFormFileCollection files, string fieldName, string fileName)
    {
        return files.FirstOrDefault(file => string.Equals(file.Name, fieldName, StringComparison.OrdinalIgnoreCase))
            ?? files.FirstOrDefault(file => string.Equals(file.Name, fileName, StringComparison.OrdinalIgnoreCase))
            ?? files.FirstOrDefault(file => string.Equals(file.FileName, fileName, StringComparison.OrdinalIgnoreCase));
    }
}

public sealed record CsvRegressionResponse(
    string Framework,
    string Model,
    bool GpuRequested,
    bool GpuUsed,
    int TrainingRows,
    int TestRows,
    int FeatureCount,
    IReadOnlyList<CsvPrediction> Predictions);

public sealed record CsvPrediction(int RowIndex, double Prediction);

internal static class CsvRegressionReader
{
    public static async Task<List<double[]>> ReadRowsAsync(IFormFile file, CancellationToken cancellationToken)
    {
        await using var stream = file.OpenReadStream();
        using var reader = new StreamReader(stream, Encoding.UTF8, detectEncodingFromByteOrderMarks: true);
        var rows = new List<double[]>();
        var firstDataLine = true;

        while (!reader.EndOfStream)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var line = await reader.ReadLineAsync(cancellationToken);
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

            throw new FormatException($"CSV file '{file.FileName}' contains a non-numeric data row: {line}");
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
