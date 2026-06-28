namespace MinimalVoiceAgent;

/// <summary>
/// A source of microphone audio for the voice agent. Implementations produce 16 kHz mono
/// 16-bit PCM in ~20 ms (320-sample) chunks — the format the VAD/STT pipeline expects — and
/// deliver each chunk to the supplied handler until cancelled.
/// </summary>
/// <remarks>
/// This is the seam that lets the agent capture either from the local microphone (via SoundFlow)
/// or from the external clean-speech-daemon, which has already removed system playback and noise.
/// </remarks>
public interface IMicrophoneCaptureSource : IAsyncDisposable
{
    /// <summary>
    /// Begins capturing and invokes <paramref name="onChunk"/> for each 16 kHz mono PCM16 frame
    /// until <paramref name="ct"/> is cancelled or the source is disposed.
    /// </summary>
    /// <param name="onChunk">Receives one ~20 ms PCM16 chunk at a time.</param>
    /// <param name="ct">Cancels capture.</param>
    Task StartAsync(Action<byte[]> onChunk, CancellationToken ct);
}
