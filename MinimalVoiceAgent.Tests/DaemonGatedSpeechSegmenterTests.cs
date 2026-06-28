using MinimalSileroVAD.Core;
using Xunit;

namespace MinimalVoiceAgent.Tests;

public class DaemonGatedSpeechSegmenterTests
{
    private static byte[] SilenceFrame() => new byte[640];
    private static byte[] SpeechFrame(float amplitude = 0.25f)
    {
        var frame = new byte[640];
        short sample = (short)(amplitude * short.MaxValue);
        for (int i = 0; i < frame.Length; i += 2)
            BitConverter.TryWriteBytes(frame.AsSpan(i, 2), sample);
        return frame;
    }

    [Fact]
    public void PushFrame_Silence_DoesNotEmitSegment()
    {
        using var segmenter = new DaemonGatedSpeechSegmenter();
        SpeechSegment? completed = null;
        segmenter.SpeechCompleted += (_, s) => completed = s;

        for (int i = 0; i < 50; i++)
            segmenter.PushFrame(SilenceFrame(), 20);

        Assert.Null(completed);
        Assert.False(segmenter.IsSpeechInProgress);
    }

    [Fact]
    public void PushFrame_SpeechThenSilence_EmitsOneSegment()
    {
        using var segmenter = new DaemonGatedSpeechSegmenter();
        int started = 0;
        SpeechSegment? completed = null;
        segmenter.SpeechStarted += (_, _) => started++;
        segmenter.SpeechCompleted += (_, s) => completed = s;

        for (int i = 0; i < 5; i++)
            segmenter.PushFrame(SpeechFrame(), 20);
        for (int i = 0; i < 20; i++)
            segmenter.PushFrame(SilenceFrame(), 20);

        Assert.Equal(1, started);
        Assert.NotNull(completed);
        Assert.True(completed!.Pcm.Length > 0);
    }

    [Fact]
    public void ComputeRms_IsZeroForSilence()
    {
        Assert.Equal(0f, DaemonGatedSpeechSegmenter.ComputeRms(SilenceFrame()));
    }
}