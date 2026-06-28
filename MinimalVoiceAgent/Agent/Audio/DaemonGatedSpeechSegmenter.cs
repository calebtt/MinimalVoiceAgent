using MinimalSileroVAD.Core;

namespace MinimalVoiceAgent;

/// <summary>
/// Fallback utterance segmenter for daemon capture when <c>enable_vad = true</c> in the daemon
/// profile (socket stream already gated to silence between speech; Silero on top would
/// double-gate). Prefer <see cref="VadSpeechSegmenter"/> when the bundled profile keeps daemon
/// VAD off — the shipped <c>clean-speech-agent.toml</c> does, so the agent uses Silero there.
/// <para>
/// Fixed RMS thresholding can stick open on room noise or residual echo; this type is kept for
/// the daemon-VAD-on case only.
/// </para>
/// </summary>
public sealed class DaemonGatedSpeechSegmenter : IVadSpeechSegmenter
{
    private const int PreRollFrames = 8;   // 160 ms at 20 ms/frame
    private const int PostRollFrames = 15; // 300 ms trailing silence ends an utterance
    private const float EnergyThreshold = 0.002f;

    private readonly List<byte[]> _preRoll = new(PreRollFrames);
    private readonly List<byte[]> _segmentFrames = new();
    private int _silenceFrames;
    private bool _inSpeech;
    private long _framesSeen;
    private long _segmentStartFrame;
    private float _peakEnergy;

    public event EventHandler? SpeechStarted;
    public event EventHandler<SpeechSegment>? SpeechCompleted;

    public bool IsSpeechInProgress => _inSpeech;
    public float LastProbability { get; private set; }

    public void PushFrame(ReadOnlySpan<byte> monoPcm, int frameLengthMs)
    {
        ArgumentOutOfRangeException.ThrowIfNotEqual(frameLengthMs, 20, nameof(frameLengthMs));

        float energy = ComputeRms(monoPcm);
        LastProbability = Math.Min(1f, energy / 0.05f);
        bool speech = energy >= EnergyThreshold;

        if (!_inSpeech)
        {
            _preRoll.Add(monoPcm.ToArray());
            if (_preRoll.Count > PreRollFrames)
                _preRoll.RemoveAt(0);

            if (!speech)
                return;

            _inSpeech = true;
            _segmentStartFrame = _framesSeen - _preRoll.Count;
            _segmentFrames.Clear();
            _segmentFrames.AddRange(_preRoll);
            _preRoll.Clear();
            _silenceFrames = 0;
            _peakEnergy = energy;
            SpeechStarted?.Invoke(this, EventArgs.Empty);
        }
        else
        {
            _segmentFrames.Add(monoPcm.ToArray());
            _peakEnergy = Math.Max(_peakEnergy, energy);

            if (speech)
            {
                _silenceFrames = 0;
            }
            else if (++_silenceFrames >= PostRollFrames)
            {
                CompleteSegment();
            }
        }

        _framesSeen++;
    }

    public void Reset()
    {
        _preRoll.Clear();
        _segmentFrames.Clear();
        _silenceFrames = 0;
        _inSpeech = false;
        _framesSeen = 0;
        _segmentStartFrame = 0;
        _peakEnergy = 0;
        LastProbability = 0;
    }

    public void Dispose() => Reset();

    private void CompleteSegment()
    {
        var pcm = ConcatFrames(_segmentFrames);
        var segment = new SpeechSegment
        {
            StartTime = TimeSpan.FromSeconds(_segmentStartFrame * 0.02),
            Duration = TimeSpan.FromSeconds(_segmentFrames.Count * 0.02),
            Probability = _peakEnergy,
            Pcm = pcm,
        };

        _inSpeech = false;
        _segmentFrames.Clear();
        _silenceFrames = 0;
        SpeechCompleted?.Invoke(this, segment);
    }

    internal static float ComputeRms(ReadOnlySpan<byte> pcm16)
    {
        if (pcm16.Length < 2)
            return 0f;

        double sum = 0;
        int count = pcm16.Length / 2;
        for (int i = 0; i < count; i++)
        {
            short sample = BitConverter.ToInt16(pcm16.Slice(i * 2, 2));
            float normalized = sample / 32768f;
            sum += normalized * normalized;
        }

        return (float)Math.Sqrt(sum / count);
    }

    private static byte[] ConcatFrames(IReadOnlyList<byte[]> frames)
    {
        int length = frames.Sum(f => f.Length);
        var pcm = new byte[length];
        int offset = 0;
        foreach (var frame in frames)
        {
            frame.CopyTo(pcm.AsSpan(offset));
            offset += frame.Length;
        }
        return pcm;
    }
}