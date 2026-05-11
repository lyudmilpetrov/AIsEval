using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;
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
    [HttpPost("SimpleRegression")]
    [RequestSizeLimit(long.MaxValue)]
    public async Task<ActionResult<CsvRegressionResponse>> SimpleRegression([FromQuery] bool UseGPU = false)
    {
        if (!Request.HasFormContentType)
        {
            return BadRequest(new
            {
                error = "This endpoint requires multipart/form-data with CSV files named features.csv and tests.csv, or form fields named features and tests."
            });
        }

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

        var features = ToFeatureArray(trainingRows, featureCount);
        var labels = ToLabelArray(trainingRows, featureCount);

        // Build using AiModelBuilder facade pattern, matching the AiDotNet simple
        // regression quick-start while feeding it the uploaded CSV training data.
        var loader = DataLoaders.FromArrays(features, labels);
        var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(new SimpleRegression<double>())
            .BuildAsync();

        // Make predictions - AiModelResult exposes Predict() directly and hides
        // the underlying model implementation details.
        var testData = ToMatrix(testRows, featureCount);
        var predictedValues = result.Predict(testData);
        var predictions = Enumerable.Range(0, testRows.Count)
            .Select(index => new CsvPrediction(index, predictedValues[index]))
            .ToArray();

        return Ok(new CsvRegressionResponse(
            "AiDotNet",
            "SimpleRegression",
            UseGPU,
            false,
            trainingRows.Count,
            testRows.Count,
            featureCount,
            predictions));
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
