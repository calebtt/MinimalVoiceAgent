using Xunit;

namespace MinimalVoiceAgent.Tests;

/// <summary>
/// Tests covering the <see cref="IChatProvider"/> abstraction that decouples the
/// voice agent from any specific LLM backend.
/// </summary>
public class ChatProviderTests
{
    [Fact]
    public void LlmChat_ImplementsIChatProvider()
    {
        // The concrete Semantic Kernel client must satisfy the abstraction so it can be
        // swapped for an alternative backend (or a fake) without touching VoiceAgentCore.
        Assert.True(typeof(IChatProvider).IsAssignableFrom(typeof(LlmChat)));
    }

    [Fact]
    public void VoiceAgentCoreBuilder_AcceptsAnyChatProvider()
    {
        // WithLlmChat must be typed against the interface; passing a fake here would not
        // compile if the builder still required the concrete LlmChat.
        IChatProvider provider = new FakeChatProvider();
        var builder = VoiceAgentCore.CreateBuilder().WithLlmChat(provider);

        Assert.NotNull(builder);
    }

    [Fact]
    public async Task ProcessMessageAsync_ReturnsResponseAndRecordsUserMessage()
    {
        var provider = new FakeChatProvider(cannedResponse: "hello there");

        var response = await provider.ProcessMessageAsync("hi");

        Assert.Equal("hello there", response);
        Assert.Equal(new[] { "hi" }, provider.ReceivedUserMessages);
    }

    [Fact]
    public async Task ProcessMessageAsync_HonorsCancellation()
    {
        var provider = new FakeChatProvider();
        using var cts = new CancellationTokenSource();
        await cts.CancelAsync();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            () => provider.ProcessMessageAsync("hi", cts.Token));
    }
}
