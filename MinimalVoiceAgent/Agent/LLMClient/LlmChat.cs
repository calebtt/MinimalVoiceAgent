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

public class LanguageModelConfig
{
    [JsonPropertyName("ApiKeyEnvironmentVariable")] public string ApiKeyEnvironmentVariable { get; set; } = string.Empty;
    [JsonPropertyName("Model")] public string Model { get; set; } = string.Empty;
    [JsonPropertyName("EndPoint")] public string EndPoint { get; set; } = string.Empty;
    [JsonPropertyName("MaxTokens")] public int MaxTokens { get; set; }
    [JsonPropertyName("WelcomeMessage")] public string WelcomeMessage { get; set; } = string.Empty;
    [JsonPropertyName("WelcomeFilePath")] public string WelcomeFilePath { get; set; } = "recordings/welcome_message.wav";
    [JsonPropertyName("InstructionsText")] public string InstructionsText { get; set; } = string.Empty;
    [JsonPropertyName("InstructionsAddendum")] public string InstructionsAddendum { get; set; } = string.Empty;
    [JsonPropertyName("ToolGuidance")] public string ToolGuidance { get; set; } = string.Empty;
    [JsonPropertyName("Temperature")] public float? Temperature { get; set; } = 0.7f;
    [JsonPropertyName("Tools")] public List<ToolSchema> Tools { get; set; } = new();
}

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

        return languageModel;
    }
}

/// <summary>
/// Upgraded LlmChat class using Semantic Kernel for enhanced tool calling.
/// Integrates profile configs for dynamic prompts from JSON.
/// Tools loaded dynamically via kernel factory; assumes pre-loaded kernel.
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
        promptBuilder.Append(config.InstructionsAddendum ?? string.Empty);

        // Enhance with loaded tool descriptions/params (immutable; detailed for better LLM guidance)
        if (functionsMetadata.Count > 0)
        {
            var toolDescriptions = string.Join("\n\n", functionsMetadata.Select(f =>
                $"Tool '{f.PluginName ?? "default"}-{f.Name}': {f.Description}\nParameters:\n" +
                string.Join("\n", f.Parameters.Select(p =>
                    $"- {p.Name} ({p.ParameterType?.Name ?? "string"}): {p.Description ?? "No description"} (required: {p.IsRequired}, default: {p.DefaultValue ?? "none"})"))));
            config.ToolGuidance += $"\nAvailable tools:\n{toolDescriptions}\nUse tools when appropriate; respond naturally otherwise.";
        }

        promptBuilder.Append(config.ToolGuidance);
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