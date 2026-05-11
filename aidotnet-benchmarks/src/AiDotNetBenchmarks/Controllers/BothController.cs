using Microsoft.AspNetCore.Mvc;

namespace AiDotNetBenchmarks.src.AiDotNetBenchmarks.Controllers
{
    public class BothController : Controller
    {
        public IActionResult Index()
        {
            return View();
        }
    }
}
