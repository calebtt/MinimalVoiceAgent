using System.Collections.Immutable;
using Microsoft.SemanticKernel;
using Xunit;

namespace MinimalVoiceAgent.Tests;

/// <summary>
/// Tests that <see cref="LlmChat.BuildSystemPrompt"/> actually incorporates the profile data
/// (instructions, rules, and per-tool guidance) rather than only the base instructions.
/// </summary>
public class SystemPromptTests
{
    [Fact]
    public void BuildSystemPrompt_IncludesInstructionsRulesAndLoadedToolGuidance()
    {
        var config = new LanguageModelConfig(
            InstructionsText: "You are Alina, a local voice assistant.",
            Rules: new List<Rule> { new("dimming", "DIMMING: dim the screen when asked.") },
            ToolInstructions: new List<ToolInstruction>
            {
                new("lower_volume", "Confirm: 'Volume lowered.'", Priority: 1),
                new("skip_youtube_ad", "Guidance for a tool that is not loaded.", Priority: 2),
            });

        var loadedTools = new[] { new KernelFunctionMetadata("lower_volume") }.ToImmutableList();

        var prompt = LlmChat.BuildSystemPrompt(config, loadedTools);

        Assert.StartsWith("You are Alina, a local voice assistant.", prompt);
        Assert.Contains("Guidelines:", prompt);
        Assert.Contains("DIMMING: dim the screen when asked.", prompt);
        Assert.Contains("Tool usage notes:", prompt);
        Assert.Contains("lower_volume: Confirm: 'Volume lowered.'", prompt);
        // Guidance for a tool that is not loaded must be filtered out.
        Assert.DoesNotContain("skip_youtube_ad", prompt);
    }

    [Fact]
    public void BuildSystemPrompt_OrdersToolGuidanceByPriority()
    {
        var config = new LanguageModelConfig(
            InstructionsText: "Base.",
            ToolInstructions: new List<ToolInstruction>
            {
                new("b_tool", "second", Priority: 5),
                new("a_tool", "first", Priority: 1),
            });

        var loaded = new[] { new KernelFunctionMetadata("a_tool"), new KernelFunctionMetadata("b_tool") }.ToImmutableList();

        var prompt = LlmChat.BuildSystemPrompt(config, loaded);

        Assert.True(prompt.IndexOf("a_tool: first", StringComparison.Ordinal)
                  < prompt.IndexOf("b_tool: second", StringComparison.Ordinal),
            "Lower-priority tool guidance should appear first.");
    }

    [Fact]
    public void BuildSystemPrompt_WithMinimalConfig_ReturnsInstructionsOnly()
    {
        var config = new LanguageModelConfig(InstructionsText: "Just the persona.");

        var prompt = LlmChat.BuildSystemPrompt(config, ImmutableList<KernelFunctionMetadata>.Empty);

        Assert.Equal("Just the persona.", prompt);
    }
}
