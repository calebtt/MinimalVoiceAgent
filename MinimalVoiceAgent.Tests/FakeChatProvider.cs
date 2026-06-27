namespace MinimalVoiceAgent.Tests;

/// <summary>
/// In-memory <see cref="IChatProvider"/> used to exercise the voice agent without a
/// live LLM endpoint. Records the messages it receives and returns a scripted reply,
/// which is exactly what makes the <see cref="IChatProvider"/> seam valuable for testing.
/// </summary>
public sealed class FakeChatProvider : IChatProvider
{
    private readonly string _cannedResponse;

    public FakeChatProvider(string cannedResponse = "ok")
        => _cannedResponse = cannedResponse;

    public List<string> ReceivedUserMessages { get; } = new();
    public List<string> AssistantMessages { get; } = new();
    public int ClearHistoryCalls { get; private set; }

    public Task<string> ProcessMessageAsync(string userMessage, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        ReceivedUserMessages.Add(userMessage);
        return Task.FromResult(_cannedResponse);
    }

    public void ClearHistory()
    {
        ClearHistoryCalls++;
        AssistantMessages.Clear();
    }

    public void AddAssistantMessage(string message) => AssistantMessages.Add(message);
}
