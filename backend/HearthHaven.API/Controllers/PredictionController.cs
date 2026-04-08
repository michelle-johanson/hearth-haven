// backend/HearthHaven.API/Controllers/PredictionController.cs
//
// Calls the Python ML inference server and returns prediction results
// to the React frontend. The Python server runs separately on port 8000.
//
// Local dev:  start the Python server first with `uvicorn server:app --port 8000`
// Production: set ML_SERVICE_URL in Azure App Service environment variables

using Microsoft.AspNetCore.Mvc;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace HearthHaven.API.Controllers;

[Route("api/[controller]")]
[ApiController]
public class PredictionController : ControllerBase
{
    private readonly HttpClient _http;
    private readonly string _mlServiceUrl;
    private readonly ILogger<PredictionController> _logger;

    public PredictionController(IHttpClientFactory factory,
                                IConfiguration config,
                                ILogger<PredictionController> logger)
    {
        _http = factory.CreateClient();
        _mlServiceUrl = config["ML_SERVICE_URL"] ?? "http://localhost:8000";
        _logger = logger;
    }

    /// <summary>
    /// Returns a Reintegration Readiness Score for the given resident.
    /// Score is 0–100; higher = more likely ready for reintegration.
    /// </summary>
    [HttpGet("reintegration/{residentId}")]
    public async Task<IActionResult> GetReintegrationScore(int residentId)
    {
        try
        {
            var url = $"{_mlServiceUrl}/predict/reintegration/{residentId}";
            _logger.LogInformation("Calling ML service: {Url}", url);

            var response = await _http.GetAsync(url);

            if (response.StatusCode == System.Net.HttpStatusCode.NotFound)
                return NotFound(new { error = $"Resident {residentId} not found" });

            if (!response.IsSuccessStatusCode)
            {
                var errorBody = await response.Content.ReadAsStringAsync();
                _logger.LogError("ML service error {Status}: {Body}",
                    response.StatusCode, errorBody);
                return StatusCode(502, new { error = "ML service returned an error" });
            }

            var json = await response.Content.ReadAsStringAsync();
            var result = JsonSerializer.Deserialize<ReintegrationPrediction>(json,
                new JsonSerializerOptions { PropertyNameCaseInsensitive = true });

            return Ok(result);
        }
        catch (HttpRequestException ex)
        {
            _logger.LogError(ex, "Could not reach ML service at {Url}", _mlServiceUrl);
            return StatusCode(503, new
            {
                error = "ML inference service is unavailable",
                detail = "Ensure the Python server is running on port 8000"
            });
        }
    }

    /// <summary>
    /// Health check — confirms the .NET backend can reach the Python ML server.
    /// </summary>
    [HttpGet("health")]
    public async Task<IActionResult> GetMlHealth()
    {
        try
        {
            var response = await _http.GetAsync($"{_mlServiceUrl}/health");
            var body = await response.Content.ReadAsStringAsync();
            return Ok(new { mlService = JsonSerializer.Deserialize<object>(body) });
        }
        catch (HttpRequestException)
        {
            return StatusCode(503, new { error = "ML service unreachable" });
        }
    }
}

// ── Response shape (matches Python server output) ─────────────────────────────

public class ReintegrationPrediction
{
    [JsonPropertyName("resident_id")]
    public int ResidentId { get; set; }

    [JsonPropertyName("readiness_score")]
    public double ReadinessScore { get; set; }

    [JsonPropertyName("probability")]
    public double Probability { get; set; }

    [JsonPropertyName("recommendation")]
    public string Recommendation { get; set; } = "";

    [JsonPropertyName("model_version")]
    public string ModelVersion { get; set; } = "";

    [JsonPropertyName("predicted_at")]
    public string PredictedAt { get; set; } = "";
}