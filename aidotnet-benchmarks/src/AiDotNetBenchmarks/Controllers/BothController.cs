using Microsoft.AspNetCore.Mvc;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace AiDotNetBenchmarks.Controllers;

[ApiController]
[Route("api/[controller]")]
public sealed class BothController : ControllerBase
{
    private static readonly Uri AiDotNetSimpleRegressionEndpoint = new("https://localhost:50724/api/Regression/SimpleRegression");
    private static readonly Uri PyTorchSimpleRegressionEndpoint = new("http://localhost:8000/api/Regression/SimpleRegression");
    private static readonly JsonSerializerOptions JsonOptions = new(JsonSerializerDefaults.Web);

    [HttpPost("SimpleRegression")]
    [RequestSizeLimit(long.MaxValue)]
    public async Task<ActionResult<BothSimpleRegressionResponse>> SimpleRegression([FromQuery] bool UseGPU = true)
    {
        if (!Request.HasFormContentType)
        {
            return BadRequest(new
            {
                error = "No multipart/form-data body was received. Send fields named features and tests so the request can be forwarded to both regression controllers.",
                receivedContentType = Request.ContentType ?? "<missing>"
            });
        }

        var form = await Request.ReadFormAsync(HttpContext.RequestAborted);
        var payload = await RegressionForwardingPayload.FromFormAsync(form, HttpContext.RequestAborted);
        var aiDotNetEndpoint = WithUseGpu(AiDotNetSimpleRegressionEndpoint, UseGPU);
        var pyTorchEndpoint = WithUseGpu(PyTorchSimpleRegressionEndpoint, UseGPU);

        using var httpClient = CreateLocalhostHttpClient();
        var aiDotNetTask = CallRegressionEndpointAsync(httpClient, aiDotNetEndpoint, payload, "AiDotNet", HttpContext.RequestAborted);
        var pyTorchTask = CallRegressionEndpointAsync(httpClient, pyTorchEndpoint, payload, "PyTorch", HttpContext.RequestAborted);

        await Task.WhenAll(aiDotNetTask, pyTorchTask);

        var results = new[] { aiDotNetTask.Result, pyTorchTask.Result };
        var response = new BothSimpleRegressionResponse(UseGPU, results);

        return results.All(result => result.IsSuccessStatusCode)
            ? Ok(response)
            : StatusCode(StatusCodes.Status502BadGateway, response);
    }

    private static HttpClient CreateLocalhostHttpClient()
    {
        var handler = new HttpClientHandler
        {
            ServerCertificateCustomValidationCallback = (message, certificate, chain, errors) =>
                message.RequestUri?.IsLoopback is true
        };

        return new HttpClient(handler)
        {
            Timeout = TimeSpan.FromMinutes(10)
        };
    }

    private static Uri WithUseGpu(Uri endpoint, bool useGpu)
    {
        var separator = string.IsNullOrEmpty(endpoint.Query) ? "?" : "&";
        return new Uri($"{endpoint}{separator}UseGPU={useGpu.ToString().ToLowerInvariant()}");
    }

    private static async Task<RegressionEndpointResult> CallRegressionEndpointAsync(
        HttpClient httpClient,
        Uri endpoint,
        RegressionForwardingPayload payload,
        string framework,
        CancellationToken cancellationToken)
    {
        try
        {
            using var content = payload.ToMultipartContent();
            using var response = await httpClient.PostAsync(endpoint, content, cancellationToken);
            var responseText = await response.Content.ReadAsStringAsync(cancellationToken);

            return new RegressionEndpointResult(
                framework,
                endpoint.ToString(),
                (int)response.StatusCode,
                response.IsSuccessStatusCode,
                TryParseJson(responseText),
                response.IsSuccessStatusCode ? null : responseText);
        }
        catch (Exception exception) when (exception is HttpRequestException or TaskCanceledException)
        {
            return new RegressionEndpointResult(
                framework,
                endpoint.ToString(),
                StatusCodes.Status503ServiceUnavailable,
                false,
                null,
                exception.Message);
        }
    }

    private static JsonElement? TryParseJson(string responseText)
    {
        if (string.IsNullOrWhiteSpace(responseText))
        {
            return null;
        }

        try
        {
            using var document = JsonDocument.Parse(responseText);
            return document.RootElement.Clone();
        }
        catch (JsonException)
        {
            return JsonSerializer.SerializeToElement(new { raw = responseText }, JsonOptions);
        }
    }
}

internal sealed record RegressionForwardingPayload(IReadOnlyList<RegressionForwardingPart> Parts)
{
    public static async Task<RegressionForwardingPayload> FromFormAsync(IFormCollection form, CancellationToken cancellationToken)
    {
        var parts = new List<RegressionForwardingPart>();

        foreach (var file in form.Files)
        {
            await using var stream = file.OpenReadStream();
            using var memory = new MemoryStream();
            await stream.CopyToAsync(memory, cancellationToken);
            parts.Add(RegressionForwardingPart.File(file.Name, file.FileName, file.ContentType, memory.ToArray()));
        }

        foreach (var field in form)
        {
            foreach (var value in field.Value)
            {
                parts.Add(RegressionForwardingPart.Field(field.Key, value ?? string.Empty));
            }
        }

        return new RegressionForwardingPayload(parts);
    }

    public MultipartFormDataContent ToMultipartContent()
    {
        var content = new MultipartFormDataContent();

        foreach (var part in Parts)
        {
            if (part.Content is not null)
            {
                var fileContent = new ByteArrayContent(part.Content);
                if (!string.IsNullOrWhiteSpace(part.ContentType))
                {
                    fileContent.Headers.ContentType = new System.Net.Http.Headers.MediaTypeHeaderValue(part.ContentType);
                }

                content.Add(fileContent, part.Name, part.FileName ?? part.Name);
            }
            else
            {
                content.Add(new StringContent(part.Value ?? string.Empty), part.Name);
            }
        }

        return content;
    }
}

internal sealed record RegressionForwardingPart(string Name, string? FileName, string? ContentType, byte[]? Content, string? Value)
{
    public static RegressionForwardingPart File(string name, string fileName, string? contentType, byte[] content) =>
        new(name, fileName, contentType, content, null);

    public static RegressionForwardingPart Field(string name, string value) =>
        new(name, null, null, null, value);
}

public sealed record BothSimpleRegressionResponse(
    [property: JsonPropertyName("gpuRequested")] bool GpuRequested,
    [property: JsonPropertyName("results")] IReadOnlyList<RegressionEndpointResult> Results);

public sealed record RegressionEndpointResult(
    [property: JsonPropertyName("framework")] string Framework,
    [property: JsonPropertyName("url")] string Url,
    [property: JsonPropertyName("statusCode")] int StatusCode,
    [property: JsonPropertyName("isSuccessStatusCode")] bool IsSuccessStatusCode,
    [property: JsonPropertyName("body")] JsonElement? Body,
    [property: JsonPropertyName("error")] string? Error);
