using Xunit;

namespace MinimalVoiceAgent.Tests;

/// <summary>
/// Tests for the startup preflight (<see cref="Algos.ValidateModelAccessAsync"/> and its
/// <see cref="Algos.ParseModelIds"/> helper) that verifies API key + model availability.
/// Covers the network-free guard branches and the model-list parsing; the live HTTP path
/// is exercised against the real endpoint at runtime.
/// </summary>
public class ModelAccessValidationTests
{
    private static LanguageModelConfig Config(string apiKeyEnv = "GROK_API_KEY", string model = "grok-4.20-0309-non-reasoning", string endpoint = "https://api.x.ai/v1")
        => new(ApiKeyEnvironmentVariable: apiKeyEnv, Model: model, EndPoint: endpoint);

    [Fact]
    public async Task Validate_FailsWhenApiKeyEnvVarUnset()
    {
        // An env var name that is guaranteed not to be set in the test process.
        var cfg = Config(apiKeyEnv: "GROK_API_KEY_UNSET_" + Guid.NewGuid().ToString("N"));

        var result = await Algos.ValidateModelAccessAsync(cfg);

        Assert.False(result.Ok);
        Assert.Contains("not set", result.Message);
    }

    [Fact]
    public async Task Validate_FailsWhenApiKeyEnvVarNameMissing()
    {
        var result = await Algos.ValidateModelAccessAsync(Config(apiKeyEnv: ""));

        Assert.False(result.Ok);
        Assert.Contains("API key environment variable", result.Message);
    }

    [Fact]
    public async Task Validate_FailsWhenModelMissing()
    {
        var result = await Algos.ValidateModelAccessAsync(Config(model: ""));

        Assert.False(result.Ok);
        Assert.Contains("model", result.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void ParseModelIds_ExtractsIdsFromOpenAiCompatibleBody()
    {
        var body = """
        { "data": [ { "id": "grok-4.20-0309-non-reasoning" }, { "id": "grok-4.3" } ], "object": "list" }
        """;

        var ids = Algos.ParseModelIds(body);

        Assert.Equal(new[] { "grok-4.20-0309-non-reasoning", "grok-4.3" }, ids);
    }

    [Theory]
    [InlineData("not json")]
    [InlineData("{}")]
    [InlineData("""{ "data": "unexpected" }""")]
    public void ParseModelIds_ReturnsEmptyForUnparseableOrUnexpectedBody(string body)
    {
        Assert.Empty(Algos.ParseModelIds(body));
    }
}
