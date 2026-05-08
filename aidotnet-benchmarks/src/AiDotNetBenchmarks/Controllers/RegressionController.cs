using Microsoft.AspNetCore.Mvc;

namespace AiDotNetBenchmarks.Controllers;

[ApiController]
[Route("api/[controller]")]
public sealed class RegressionController : ControllerBase
{
    [HttpGet("Test")]
    public ActionResult<string> Test() => "ping";
}
