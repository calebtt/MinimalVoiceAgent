using Xunit;

namespace MinimalVoiceAgent.Tests;

/// <summary>
/// Tests for <see cref="Algos.LoadLanguageModelConfigAsync"/>, which parses the agent
/// profile JSON and applies the de-duplication / validation rules for prompt sections,
/// rules, and tool instructions.
/// </summary>
public class LanguageModelConfigTests : IDisposable
{
    private readonly string _tempDir;

    public LanguageModelConfigTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), "mva-tests-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_tempDir);
    }

    public void Dispose()
    {
        try { Directory.Delete(_tempDir, recursive: true); } catch { /* best effort */ }
    }

    private async Task<string> WriteProfileAsync(string json)
    {
        var path = Path.Combine(_tempDir, "profile.json");
        await File.WriteAllTextAsync(path, json);
        return path;
    }

    [Fact]
    public async Task LoadAsync_MissingFile_Throws()
    {
        await Assert.ThrowsAsync<FileNotFoundException>(
            () => Algos.LoadLanguageModelConfigAsync(Path.Combine(_tempDir, "nope.json")));
    }

    [Fact]
    public async Task LoadAsync_MissingLanguageModelSection_Throws()
    {
        var path = await WriteProfileAsync("""{ "SomethingElse": {} }""");

        await Assert.ThrowsAsync<InvalidOperationException>(
            () => Algos.LoadLanguageModelConfigAsync(path));
    }

    [Fact]
    public async Task LoadAsync_ParsesCoreFields()
    {
        var path = await WriteProfileAsync("""
        {
          "LanguageModel": {
            "Model": "grok-beta",
            "EndPoint": "https://api.x.ai/v1",
            "MaxTokens": 512,
            "Temperature": 0.2
          }
        }
        """);

        var config = await Algos.LoadLanguageModelConfigAsync(path);

        Assert.Equal("grok-beta", config.Model);
        Assert.Equal("https://api.x.ai/v1", config.EndPoint);
        Assert.Equal(512, config.MaxTokens);
        Assert.Equal(0.2f, config.Temperature);
    }

    [Fact]
    public async Task LoadAsync_DedupesPromptSectionsByType_KeepingLast()
    {
        // Two Base sections (Type 0); the loader keeps the last and drops empty ones.
        var path = await WriteProfileAsync("""
        {
          "LanguageModel": {
            "Model": "m",
            "EndPoint": "https://e",
            "PromptSections": [
              { "type": 0, "content": "first base" },
              { "type": 0, "content": "second base" },
              { "type": 1, "content": "   " }
            ]
          }
        }
        """);

        var config = await Algos.LoadLanguageModelConfigAsync(path);

        var section = Assert.Single(config.PromptSections!);
        Assert.Equal(SectionType.Base, section.Type);
        Assert.Equal("second base", section.Content);
    }

    [Fact]
    public async Task LoadAsync_DedupesRulesById_KeepingFirst_AndSkipsEmpty()
    {
        var path = await WriteProfileAsync("""
        {
          "LanguageModel": {
            "Model": "m",
            "EndPoint": "https://e",
            "Rules": [
              { "id": "r1", "description": "keep me", "appliesTo": [] },
              { "id": "r1", "description": "duplicate id", "appliesTo": [] },
              { "id": "r2", "description": "  ", "appliesTo": [] }
            ]
          }
        }
        """);

        var config = await Algos.LoadLanguageModelConfigAsync(path);

        var rule = Assert.Single(config.Rules!);
        Assert.Equal("r1", rule.Id);
        Assert.Equal("keep me", rule.Description);
    }

    [Fact]
    public async Task LoadAsync_SkipsToolInstructionsWithEmptyGuidance()
    {
        var path = await WriteProfileAsync("""
        {
          "LanguageModel": {
            "Model": "m",
            "EndPoint": "https://e",
            "ToolInstructions": [
              { "toolName": "good", "guidance": "do the thing", "priority": 1 },
              { "toolName": "bad", "guidance": "", "priority": 0 }
            ]
          }
        }
        """);

        var config = await Algos.LoadLanguageModelConfigAsync(path);

        var instruction = Assert.Single(config.ToolInstructions!);
        Assert.Equal("good", instruction.ToolName);
    }
}
