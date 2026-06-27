using Xunit;

namespace MinimalVoiceAgent.Tests;

/// <summary>
/// Tests for the PCM16 &lt;-&gt; float32 conversion helpers in <see cref="Algos"/>.
/// These sit on the hot audio path (mic capture and TTS playback), so correctness
/// and round-trip fidelity matter.
/// </summary>
public class AudioConversionTests
{
    [Fact]
    public void ConvertPcmToFloat_RejectsOddLengthBuffer()
    {
        // 16-bit PCM must be 2-byte aligned; a 3-byte buffer is malformed.
        var malformed = new byte[] { 0x00, 0x01, 0x02 };

        var ex = Assert.Throws<ArgumentException>(() => Algos.ConvertPcmToFloat(malformed));
        Assert.Equal("pcmBytes", ex.ParamName);
    }

    [Fact]
    public void ConvertPcmToFloat_ProducesOneSamplePerTwoBytes()
    {
        var pcm = new byte[8]; // 4 samples

        var floats = Algos.ConvertPcmToFloat(pcm);

        Assert.Equal(4, floats.Length);
        Assert.All(floats, f => Assert.Equal(0f, f));
    }

    [Theory]
    [InlineData(new short[] { 0, 1, -1, 1000, -1000, 16384, -16384 })]
    [InlineData(new short[] { short.MinValue, short.MaxValue })]
    public void PcmFloatRoundTrip_PreservesSamplesWithinQuantizationError(short[] samples)
    {
        // Arrange: encode the known samples as little-endian PCM16.
        var pcm = new byte[samples.Length * 2];
        for (int i = 0; i < samples.Length; i++)
            BitConverter.TryWriteBytes(pcm.AsSpan(i * 2, 2), samples[i]);

        // Act: PCM -> float -> PCM.
        float[] floats = Algos.ConvertPcmToFloat(pcm);
        byte[] roundTripped = Algos.ConvertFloatToPcm(floats);
        var recovered = new short[samples.Length];
        for (int i = 0; i < samples.Length; i++)
            recovered[i] = BitConverter.ToInt16(roundTripped, i * 2);

        // Assert: the normalize (÷32767) / denormalize path is lossy by at most one quantum.
        for (int i = 0; i < samples.Length; i++)
            Assert.True(Math.Abs(samples[i] - recovered[i]) <= 1,
                $"Sample {i}: expected ~{samples[i]} but got {recovered[i]}");
    }

    [Fact]
    public void ConvertFloatToPcm_ClampsValuesOutsideUnitRange()
    {
        // Values beyond [-1, 1] must saturate rather than wrap around.
        var floats = new[] { 2.0f, -2.0f };

        byte[] pcm = Algos.ConvertFloatToPcm(floats);

        Assert.Equal(short.MaxValue, BitConverter.ToInt16(pcm, 0));
        Assert.Equal(-short.MaxValue, BitConverter.ToInt16(pcm, 2));
    }
}
