using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.OpenAI;
using Serilog;
using System.Collections.Immutable;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace MinimalVoiceAgent;

/// <summary>
/// Records for JSON tool schema (OpenAPI-style for SK compatibility). Immutable by design.
/// </summary>
public record ToolSchema
{
    [JsonPropertyName("name")]
    public string Name { get; init; }

    [JsonPropertyName("description")]
    public string Description { get; init; }

    [JsonPropertyName("parameters")]
    public ParametersSchema Parameters { get; init; }

    public ToolSchema(string name, string description, ParametersSchema parameters)
    {
        Name = name;
        Description = description;
        Parameters = parameters;
    }

    public record ParametersSchema(
        [property: JsonPropertyName("type")] string Type, // e.g., "object"
        [property: JsonPropertyName("properties")] IReadOnlyDictionary<string, ParamSchema> Properties,
        [property: JsonPropertyName("required")] string[]? Required = null);

    public record ParamSchema(
        [property: JsonPropertyName("type")] string Type,
        [property: JsonPropertyName("description")] string Description,
        [property: JsonPropertyName("default")] object? Default = null,
        [property: JsonPropertyName("enum")] string[]? Enum = null);
}

/// <summary>
/// Immutable record for per-tool guidance. Supports priority for prompt ordering.
/// </summary>
public record ToolInstruction(
    [property: JsonPropertyName("toolName")] string ToolName,
    [property: JsonPropertyName("guidance")] string Guidance,
    [property: JsonPropertyName("priority")] int Priority = 0);

/// <summary>
/// Enum for prompt section ordering. Extensible for future types.
/// </summary>
public enum SectionType
{
    Base = 0,
    Addendum = 1,
    Rules = 2,
    Tools = 3  // Placeholder for tool-specific insertion
}

/// <summary>
/// Immutable record for composable prompt sections.
/// </summary>
public record PromptSection(
    [property: JsonPropertyName("type")] SectionType Type,
    [property: JsonPropertyName("content")] string Content,
    [property: JsonPropertyName("condition")] string? Condition = null);  // e.g., "tools_loaded"

/// <summary>
/// Immutable record for rules (extracted from ToolGuidance).
/// </summary>
public record Rule(
    [property: JsonPropertyName("id")] string Id,
    [property: JsonPropertyName("description")] string Description,
    [property: JsonPropertyName("appliesTo")] string[] AppliesTo = null!);  // Default to global if empty

/// <summary>
/// Immutable record for LLM configuration. Loaded from JSON; supports modular prompt/rules.
/// Legacy fields (InstructionsAddendum, ToolGuidance) deserialized but ignored.
/// </summary>
public record LanguageModelConfig(
    [property: JsonPropertyName("ApiKeyEnvironmentVariable")] string ApiKeyEnvironmentVariable = "",
    [property: JsonPropertyName("Model")] string Model = "",
    [property: JsonPropertyName("EndPoint")] string EndPoint = "",
    [property: JsonPropertyName("MaxTokens")] int MaxTokens = 0,
    [property: JsonPropertyName("WelcomeMessage")] string WelcomeMessage = "",
    [property: JsonPropertyName("WelcomeFilePath")] string WelcomeFilePath = "recordings/welcome_message.wav",
    [property: JsonPropertyName("InstructionsText")] string InstructionsText = "",
    [property: JsonPropertyName("Temperature")] float? Temperature = 0.1f,
    [property: JsonPropertyName("Tools")] List<ToolSchema> Tools = default!,
    [property: JsonPropertyName("ToolInstructions")] List<ToolInstruction>? ToolInstructions = default,
    [property: JsonPropertyName("PromptSections")] List<PromptSection>? PromptSections = default,
    [property: JsonPropertyName("Rules")] List<Rule>? Rules = default);


/// <summary>
/// A place to put free functions.
/// </summary>
public static partial class Algos
{
    public static Kernel BuildKernel(LanguageModelConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);

        var keyName = config.ApiKeyEnvironmentVariable;
        var lmApiKey = Environment.GetEnvironmentVariable(keyName) ?? String.Empty;
        var builder = Kernel.CreateBuilder();
        builder.AddOpenAIChatCompletion(
            modelId: config.Model ?? throw new ArgumentException("Model is required", nameof(config.Model)),
            apiKey: lmApiKey,
            endpoint: new Uri(config.EndPoint ?? throw new ArgumentException("EndPoint is required", nameof(config.EndPoint))));
        return builder.Build();
    }

    public static async Task<LanguageModelConfig> LoadLanguageModelConfigAsync(string profilePath)
    {
        if (!File.Exists(profilePath))
        {
            throw new FileNotFoundException($"Profile file not found: {profilePath}");
        }

        var json = await File.ReadAllTextAsync(profilePath);
        using var document = JsonDocument.Parse(json);

        if (!document.RootElement.TryGetProperty("LanguageModel", out var lmElement))
        {
            throw new InvalidOperationException("LanguageModel section is missing in the profile JSON.");
        }

        var options = new JsonSerializerOptions { PropertyNameCaseInsensitive = true };
        var languageModel = JsonSerializer.Deserialize<LanguageModelConfig>(lmElement.GetRawText(), options)
            ?? throw new InvalidOperationException("Failed to deserialize LanguageModel configuration.");

        // Validate PromptSections: Skip empties, dedupe by Type (keep last)
        var sections = languageModel.PromptSections ?? new List<PromptSection>();
        var invalids = sections.Where(s => string.IsNullOrWhiteSpace(s.Content)).ToList();
        if (invalids.Any())
        {
            Log.Warning("Skipping {InvalidCount} empty PromptSections.", invalids.Count);
            sections = sections.Except(invalids).ToList();
        }
        sections = sections.GroupBy(s => s.Type).Select(g => g.Last()).ToList();
        languageModel = languageModel with { PromptSections = sections };

        // Validate Rules: Skip invalids, dedupe by Id (keep first)
        var rules = languageModel.Rules ?? new List<Rule>();
        var ruleInvalids = rules.Where(r => string.IsNullOrWhiteSpace(r.Description)).ToList();
        if (ruleInvalids.Any())
        {
            Log.Warning("Skipping {InvalidCount} invalid Rules.", ruleInvalids.Count);
            rules = rules.Except(ruleInvalids).ToList();
        }
        rules = rules.GroupBy(r => r.Id).Select(g => g.First()).ToList();
        languageModel = languageModel with { Rules = rules };

        // Validate ToolInstructions: Skip empties
        if (languageModel.ToolInstructions?.Any() == true)
        {
            var invalid = languageModel.ToolInstructions.Where(ti => string.IsNullOrWhiteSpace(ti.Guidance)).ToList();
            if (invalid.Any())
            {
                Log.Warning("Skipping {InvalidCount} invalid ToolInstructions.", invalid.Count);
                languageModel = languageModel with { ToolInstructions = languageModel.ToolInstructions.Except(invalid).ToList() };
            }
        }

        return languageModel;
    }

}

/// <summary>
/// Upgraded LlmChat class using Semantic Kernel for enhanced tool calling.
/// Integrates profile configs for dynamic prompts from JSON.
/// Tools loaded dynamically via kernel factory; assumes pre-loaded kernel.
/// TODO: Add rate-limiting for API usage, it should not spam queries and rack up costs.
/// </summary>
public class LlmChat
{
    private readonly Kernel _kernel;
    private readonly IChatCompletionService _chatService;
    private readonly ChatHistory _chatHistory;
    private readonly string _systemPrompt;
    private readonly float _temperature;
    private readonly int _maxTokens;
    private readonly ImmutableList<KernelFunctionMetadata> _functionsMetadata;

    public LlmChat(
        LanguageModelConfig config,
        ComputerToolFunctions toolFunctions,
        Kernel kernel) // Required pre-loaded kernel
    {
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(toolFunctions);
        ArgumentNullException.ThrowIfNull(kernel);

        _kernel = kernel;
        _kernel.ImportPluginFromObject(toolFunctions, pluginName: "SemanticTools");

        _temperature = config.Temperature ?? 0.7f;
        _maxTokens = config.MaxTokens > 0 ? config.MaxTokens : 1024; // Robust default

        _chatService = _kernel.GetRequiredService<IChatCompletionService>();

        // Validate tools/plugins loaded (robust: Use GetFunctionsMetadata for count + metadata)
        _functionsMetadata = _kernel.Plugins.GetFunctionsMetadata().ToImmutableList();
        var pluginCount = _kernel.Plugins.Count;
        var functionCount = _functionsMetadata.Count;
        Log.Information("Kernel loaded with {PluginCount} plugins and {FunctionCount} functions.", pluginCount, functionCount);

        // Log metadata for debug
        if (functionCount > 0)
        {
            var sampleFunc = _functionsMetadata[0];
            var paramCount = sampleFunc.Parameters.Count;
            Log.Debug("Sample function '{Name}': {ParamCount} params (e.g., {FirstParam})",
                sampleFunc.Name, paramCount, sampleFunc.Parameters.FirstOrDefault()?.Name ?? "none");
        }
        else
        {
            Log.Warning("No tools/functions registered—check plugin loader/DLL. Falling back to prompt-only chat.");
        }

        // Quick 422 risk check (modern: Flag non-string req'd params)
        var riskyParams = _functionsMetadata
            .SelectMany(f => f.Parameters.Where(p => p.IsRequired && p.ParameterType != typeof(string)))
            .ToList();
        if (riskyParams.Any())
        {
            Log.Warning("Potential 422 risks: {RiskyCount} non-string required params across tools (xAI strict). Consider stringifying params.", riskyParams.Count);
        }

        // Build dynamic system prompt from profile (now uses loaded metadata)
        _systemPrompt = BuildSystemPrompt(config, _functionsMetadata);

        // Chat history for multi-turn context
        _chatHistory = new ChatHistory();
    }

    /// <summary>
    /// Processes a user message using function calling for intelligent tool invocation.
    /// With AutoInvoke, a single call handles tool execution and returns the final response.
    /// Disables tool calling if no functions loaded (robust against 422 errors).
    /// </summary>
    public async Task<string> ProcessMessageAsync(string userMessage, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNullOrWhiteSpace(userMessage);

        if (_chatHistory.Count == 0)
        {
            _chatHistory.AddSystemMessage(_systemPrompt);
        }

        _chatHistory.AddUserMessage(userMessage);

        var executionSettings = new OpenAIPromptExecutionSettings
        {
            ToolCallBehavior = _functionsMetadata.Count > 0 ? ToolCallBehavior.AutoInvokeKernelFunctions : null,
            Temperature = _temperature,
            MaxTokens = _maxTokens
        };

        try
        {
            var response = await _chatService.GetChatMessageContentAsync(
                _chatHistory,
                executionSettings,
                _kernel,
                cancellationToken);

            _chatHistory.Add(response);

            return response?.Content ?? "No response generated.";
        }
        catch (Exception ex)
        {
            Log.Error(ex, "LLM processing failed for message: {Message}", userMessage);
            return $"Error in processing: {ex.Message}. Falling back to basic chat.";
        }
    }

    /// <summary>
    /// Builds the full system prompt from profile config (Instructions + Addendum + interpolated ToolGuidance).
    /// Enhanced: Dynamically includes descriptions/params only for loaded tools (avoids mismatch if none loaded).
    /// </summary>
    private static string BuildSystemPrompt(LanguageModelConfig config, ImmutableList<KernelFunctionMetadata> functionsMetadata)
    {
        var promptBuilder = new StringBuilder(config.InstructionsText ?? string.Empty);

        // Enhance with loaded tool descriptions/params (immutable; detailed for better LLM guidance)
        if (functionsMetadata.Count > 0)
        {
            var toolDescriptions = string.Join("\n\n", functionsMetadata.Select(f =>
                $"Tool '{f.PluginName ?? "default"}-{f.Name}': {f.Description}\nParameters:\n" +
                string.Join("\n", f.Parameters.Select(p =>
                    $"- {p.Name} ({p.ParameterType?.Name ?? "string"}): {p.Description ?? "No description"} (required: {p.IsRequired}, default: {p.DefaultValue ?? "none"})"))));
        }

        return promptBuilder.ToString();
    }

    public void ClearHistory()
    {
        _chatHistory.Clear();
        _chatHistory.AddSystemMessage(_systemPrompt);  // Ensure system prompt is always present after clear
    }

    public void AddAssistantMessage(string message)
    {
        ArgumentNullException.ThrowIfNullOrWhiteSpace(message);
        _chatHistory.AddAssistantMessage(message);
    }
}