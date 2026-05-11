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
    private const int MinimumRowsForAiDotNetValidationSplit = 7;

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

        var predictedValues = trainingRows.Count >= MinimumRowsForAiDotNetValidationSplit
            ? await PredictWithAiDotNetAsync(trainingRows, testRows, featureCount, HttpContext.RequestAborted)
            : PredictWithLeastSquares(trainingRows, testRows, featureCount);
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


    private static async Task<Vector<double>> PredictWithAiDotNetAsync(
        IReadOnlyList<double[]> trainingRows,
        IReadOnlyList<double[]> testRows,
        int featureCount,
        CancellationToken cancellationToken)
    {
        var features = ToFeatureArray(trainingRows, featureCount);
        var labels = ToLabelArray(trainingRows, featureCount);

        // AiDotNet currently performs an internal 70/15/15 train/validation/test
        // split. Datasets with fewer than seven rows produce a zero-row
        // validation matrix, so small CSV uploads use the least-squares fallback
        // below instead of entering the builder path.
        var loader = DataLoaders.FromArrays(features, labels);
        var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(new SimpleRegression<double>())
            .BuildAsync(cancellationToken);

        var testData = ToMatrix(testRows, featureCount);
        return result.Predict(testData);
    }

    private static Vector<double> PredictWithLeastSquares(
        IReadOnlyList<double[]> trainingRows,
        IReadOnlyList<double[]> testRows,
        int featureCount)
    {
        var coefficients = FitLeastSquares(trainingRows, featureCount);
        var predictions = new Vector<double>(testRows.Count);

        for (var rowIndex = 0; rowIndex < testRows.Count; rowIndex++)
        {
            var prediction = coefficients[0];
            for (var featureIndex = 0; featureIndex < featureCount; featureIndex++)
            {
                prediction += coefficients[featureIndex + 1] * testRows[rowIndex][featureIndex];
            }

            predictions[rowIndex] = prediction;
        }

        return predictions;
    }

    private static double[] FitLeastSquares(IReadOnlyList<double[]> rows, int featureCount)
    {
        var coefficientCount = featureCount + 1;
        var normalMatrix = new double[coefficientCount, coefficientCount];
        var normalVector = new double[coefficientCount];

        foreach (var row in rows)
        {
            var target = row[featureCount];
            for (var i = 0; i < coefficientCount; i++)
            {
                var left = i == 0 ? 1d : row[i - 1];
                normalVector[i] += left * target;

                for (var j = 0; j < coefficientCount; j++)
                {
                    var right = j == 0 ? 1d : row[j - 1];
                    normalMatrix[i, j] += left * right;
                }
            }
        }

        return SolveLinearSystem(normalMatrix, normalVector);
    }

    private static double[] SolveLinearSystem(double[,] matrix, double[] vector)
    {
        const double Ridge = 1e-8;
        const double PivotTolerance = 1e-12;

        var size = vector.Length;
        var augmented = new double[size, size + 1];

        for (var row = 0; row < size; row++)
        {
            for (var column = 0; column < size; column++)
            {
                augmented[row, column] = matrix[row, column] + (row == column ? Ridge : 0d);
            }

            augmented[row, size] = vector[row];
        }

        for (var pivot = 0; pivot < size; pivot++)
        {
            var pivotRow = pivot;
            var pivotMagnitude = Math.Abs(augmented[pivot, pivot]);
            for (var row = pivot + 1; row < size; row++)
            {
                var candidate = Math.Abs(augmented[row, pivot]);
                if (candidate > pivotMagnitude)
                {
                    pivotMagnitude = candidate;
                    pivotRow = row;
                }
            }

            if (pivotRow != pivot)
            {
                for (var column = pivot; column <= size; column++)
                {
                    (augmented[pivot, column], augmented[pivotRow, column]) =
                        (augmented[pivotRow, column], augmented[pivot, column]);
                }
            }

            if (Math.Abs(augmented[pivot, pivot]) < PivotTolerance)
            {
                augmented[pivot, pivot] = PivotTolerance;
            }

            var pivotValue = augmented[pivot, pivot];
            for (var row = pivot + 1; row < size; row++)
            {
                var factor = augmented[row, pivot] / pivotValue;
                for (var column = pivot; column <= size; column++)
                {
                    augmented[row, column] -= factor * augmented[pivot, column];
                }
            }
        }

        var solution = new double[size];
        for (var row = size - 1; row >= 0; row--)
        {
            var sum = augmented[row, size];
            for (var column = row + 1; column < size; column++)
            {
                sum -= augmented[row, column] * solution[column];
            }

            solution[row] = sum / augmented[row, row];
        }

        return solution;
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
