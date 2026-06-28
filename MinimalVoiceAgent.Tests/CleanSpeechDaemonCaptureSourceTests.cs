using System.Text;
using Xunit;

namespace MinimalVoiceAgent.Tests;

/// <summary>
/// Tests for the clean-speech-daemon socket protocol parsing in
/// <see cref="CleanSpeechDaemonCaptureSource"/> (JSON metadata header + raw PCM framing).
/// The live socket connection and resampling are exercised at runtime against the daemon.
/// </summary>
public class CleanSpeechDaemonCaptureSourceTests
{
    [Fact]
    public void ParseHeader_ReadsRateAndChannels()
    {
        var (rate, channels) = CleanSpeechDaemonCaptureSource.ParseHeader(
            """{"format":"s16le","sample_rate":48000,"channels":1,"note":"x"}""");

        Assert.Equal(48000, rate);
        Assert.Equal(1, channels);
    }

    [Fact]
    public void ParseHeader_DefaultsWhenFieldsMissing()
    {
        var (rate, channels) = CleanSpeechDaemonCaptureSource.ParseHeader("""{"format":"s16le"}""");

        Assert.Equal(48000, rate);
        Assert.Equal(1, channels);
    }

    [Fact]
    public void ParseHeader_RejectsNonS16leFormat()
    {
        Assert.Throws<NotSupportedException>(() =>
            CleanSpeechDaemonCaptureSource.ParseHeader("""{"format":"f32le","sample_rate":48000,"channels":1}"""));
    }

    [Fact]
    public void ParseHeader_RejectsNonMono()
    {
        Assert.Throws<NotSupportedException>(() =>
            CleanSpeechDaemonCaptureSource.ParseHeader("""{"format":"s16le","sample_rate":48000,"channels":2}"""));
    }

    [Fact]
    public async Task ReadHeaderLineAsync_ReturnsLineAndLeavesPcmForReading()
    {
        const string header = """{"format":"s16le","sample_rate":48000,"channels":1}""";
        var pcm = new byte[] { 1, 2, 3, 4 };
        using var stream = new MemoryStream(Encoding.UTF8.GetBytes(header + "\n").Concat(pcm).ToArray());

        var line = await CleanSpeechDaemonCaptureSource.ReadHeaderLineAsync(stream, CancellationToken.None);

        Assert.Equal(header, line);
        // The newline is consumed; the PCM bytes remain for the read loop.
        var remaining = new byte[pcm.Length];
        int read = await stream.ReadAsync(remaining);
        Assert.Equal(pcm.Length, read);
        Assert.Equal(pcm, remaining);
    }

    [Fact]
    public async Task ReadHeaderLineAsync_ThrowsIfClosedBeforeNewline()
    {
        using var stream = new MemoryStream(Encoding.UTF8.GetBytes("""{"format":"s16le"}"""));

        await Assert.ThrowsAsync<EndOfStreamException>(
            () => CleanSpeechDaemonCaptureSource.ReadHeaderLineAsync(stream, CancellationToken.None));
    }

    [Fact]
    public async Task ReadHeaderLineAsync_ThrowsWhenHeaderExceedsMaxLengthWithoutNewline()
    {
        var oversized = new byte[CleanSpeechDaemonCaptureSource.MaxHeaderBytes];
        Array.Fill(oversized, (byte)'x');
        using var stream = new MemoryStream(oversized);

        await Assert.ThrowsAsync<InvalidDataException>(
            () => CleanSpeechDaemonCaptureSource.ReadHeaderLineAsync(stream, CancellationToken.None));
    }

    [Fact]
    public void Constructor_RejectsBlankSocketPath()
    {
        Assert.Throws<ArgumentException>(() => new CleanSpeechDaemonCaptureSource("  "));
    }

    [Fact]
    public void IsPlatformSupported_IsTrueOnLinux()
    {
        if (!OperatingSystem.IsLinux())
            return;

        Assert.True(CleanSpeechDaemonCaptureSource.IsPlatformSupported);
    }
}
