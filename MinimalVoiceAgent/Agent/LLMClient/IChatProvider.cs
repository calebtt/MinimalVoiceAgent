namespace MinimalVoiceAgent;

/// <summary>
/// Abstraction over a conversational language-model backend.
/// <para>
/// The voice agent depends only on this interface, not on any concrete provider
/// (Semantic Kernel, an OpenAI-compatible endpoint such as Grok, a local model, or a
/// fake used in tests). Swapping the backend is a matter of supplying a different
/// <see cref="IChatProvider"/> implementation to
/// <see cref="VoiceAgentCore.Builder.WithLlmChat(IChatProvider)"/>.
/// </para>
/// </summary>
public interface IChatProvider
{
    /// <summary>
    /// Sends a user message to the model and returns the assistant's reply.
    /// Implementations are expected to maintain their own multi-turn conversation
    /// history and to perform any tool/function calling internally.
    /// </summary>
    /// <param name="userMessage">The user's (typically transcribed) message. Must be non-empty.</param>
    /// <param name="cancellationToken">Token used to cancel an in-flight request.</param>
    /// <returns>The assistant's textual response.</returns>
    Task<string> ProcessMessageAsync(string userMessage, CancellationToken cancellationToken = default);

    /// <summary>
    /// Clears the conversation history, resetting the model back to its initial
    /// system prompt for the next turn.
    /// </summary>
    void ClearHistory();

    /// <summary>
    /// Appends an assistant message to the conversation history without issuing a
    /// model request. Useful for seeding context or recording out-of-band replies.
    /// </summary>
    /// <param name="message">The assistant message to record. Must be non-empty.</param>
    void AddAssistantMessage(string message);
}
