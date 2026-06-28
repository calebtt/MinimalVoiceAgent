using NAudio.Wave;
using NAudio.Wave.SampleProviders;
using Serilog;
using System.Net.Sockets;
using System.Text;
using System.Text.Json;

namespace MinimalVoiceAgent;

/// <summary>
/// Microphone capture source backed by the external <c>clean-speech-daemon</c>.
/// <para>
/// The daemon captures the webcam mic plus the system playback monitor, performs acoustic echo
/// cancellation (removing all system audio, not just the agent's own TTS), noise suppression, and
/// speech gating, then publishes the cleaned stream over a Unix domain socket
/// (default <c>/tmp/clean-speech-daemon.sock</c>): one JSON metadata line
/// (<c>{"format":"s16le","sample_rate":...,"channels":...}</c>) followed by raw little-endian
/// 16-bit mono PCM.
/// </para>
/// <para>
/// This source reads that stream, resamples it to 16 kHz, and emits ~20 ms (320-sample) PCM16
/// chunks into the VAD/STT pipeline. Because the daemon has already done echo cancellation, the
/// agent's own WebRTC APM is not used on this path.
/// </para>
/// </summary>
public sealed class CleanSpeechDaemonCaptureSource : IMicrophoneCaptureSource
{
    private const int TargetSampleRate = 16000;
    private const int FrameSamples = 320; // 20 ms at 16 kHz

    private readonly string _socketPath;
    private Socket? _socket;
    private Task? _readLoop;
    private CancellationTokenSource? _linkedCts;

    public CleanSpeechDaemonCaptureSource(string socketPath)
        => _socketPath = socketPath ?? throw new ArgumentNullException(nameof(socketPath));

    public async Task StartAsync(Action<byte[]> onChunk, CancellationToken ct)
    {
        ArgumentNullException.ThrowIfNull(onChunk);

        if (!File.Exists(_socketPath))
            throw new FileNotFoundException(
                $"clean-speech-daemon socket not found at '{_socketPath}'. Is the daemon running ('clean-speech-daemon run')?");

        _socket = new Socket(AddressFamily.Unix, SocketType.Stream, ProtocolType.Unspecified);
        await _socket.ConnectAsync(new UnixDomainSocketEndPoint(_socketPath), ct);
        var stream = new NetworkStream(_socket, ownsSocket: false);

        var header = await ReadHeaderLineAsync(stream, ct);
        var (sourceRate, channels) = ParseHeader(header);
        Log.Information("clean-speech-daemon connected: {Rate} Hz {Ch} ch s16le -> resampling to {Target} Hz",
            sourceRate, channels, TargetSampleRate);

        _linkedCts = CancellationTokenSource.CreateLinkedTokenSource(ct);
        _readLoop = Task.Run(() => ReadLoopAsync(stream, sourceRate, onChunk, _linkedCts.Token), CancellationToken.None);
    }

    /// <summary>Reads bytes up to the first newline and returns them as the (UTF-8) header line.</summary>
    internal static async Task<string> ReadHeaderLineAsync(Stream stream, CancellationToken ct)
    {
        var bytes = new List<byte>(256);
        var one = new byte[1];
        while (bytes.Count < 8192)
        {
            int n = await stream.ReadAsync(one.AsMemory(0, 1), ct);
            if (n == 0) throw new EndOfStreamException("Socket closed before the metadata newline.");
            if (one[0] == (byte)'\n') break;
            bytes.Add(one[0]);
        }
        return Encoding.UTF8.GetString(bytes.ToArray());
    }

    /// <summary>Parses the daemon's JSON metadata line, validating the PCM format.</summary>
    internal static (int sampleRate, int channels) ParseHeader(string headerJson)
    {
        using var doc = JsonDocument.Parse(headerJson);
        var root = doc.RootElement;

        var format = root.TryGetProperty("format", out var f) ? f.GetString() : "s16le";
        if (!string.Equals(format, "s16le", StringComparison.OrdinalIgnoreCase))
            throw new NotSupportedException($"Unsupported daemon PCM format '{format}'; expected 's16le'.");

        int sampleRate = root.TryGetProperty("sample_rate", out var r) ? r.GetInt32() : 48000;
        int channels = root.TryGetProperty("channels", out var c) ? c.GetInt32() : 1;
        if (channels != 1)
            throw new NotSupportedException($"Unsupported daemon channel count {channels}; expected mono.");
        if (sampleRate <= 0)
            throw new NotSupportedException($"Invalid daemon sample rate {sampleRate}.");

        return (sampleRate, channels);
    }

    private async Task ReadLoopAsync(NetworkStream stream, int sourceRate, Action<byte[]> onChunk, CancellationToken ct)
    {
        // Stream the incoming mono PCM16 through a cross-platform resampler to 16 kHz, then
        // re-frame into exact 320-sample (20 ms) chunks for the VAD/STT pipeline.
        var buffered = new BufferedWaveProvider(new WaveFormat(sourceRate, 16, 1))
        {
            ReadFully = false,
            DiscardOnBufferOverflow = true,
            BufferDuration = TimeSpan.FromSeconds(10),
        };
        ISampleProvider samples = buffered.ToSampleProvider();
        if (sourceRate != TargetSampleRate)
            samples = new WdlResamplingSampleProvider(samples, TargetSampleRate);

        var readBuf = new byte[8192];
        var frame = new float[FrameSamples];
        int framePos = 0;

        try
        {
            while (!ct.IsCancellationRequested)
            {
                int n = await stream.ReadAsync(readBuf.AsMemory(), ct);
                if (n == 0)
                {
                    Log.Warning("clean-speech-daemon socket closed by the daemon.");
                    break;
                }
                buffered.AddSamples(readBuf, 0, n);

                // Drain all currently-available resampled audio into exact 320-sample frames.
                int got;
                while ((got = samples.Read(frame, framePos, FrameSamples - framePos)) > 0)
                {
                    framePos += got;
                    if (framePos == FrameSamples)
                    {
                        onChunk(Algos.ConvertFloatToPcm(frame));
                        framePos = 0;
                    }
                }
            }
        }
        catch (OperationCanceledException) { /* normal shutdown */ }
        catch (Exception ex)
        {
            Log.Error(ex, "clean-speech-daemon read loop failed.");
        }
    }

    public async ValueTask DisposeAsync()
    {
        try { if (_linkedCts is { IsCancellationRequested: false }) await _linkedCts.CancelAsync(); } catch { /* best effort */ }
        if (_readLoop is not null)
        {
            try { await _readLoop; } catch { /* swallow shutdown races */ }
        }
        _socket?.Dispose();
        _linkedCts?.Dispose();
    }
}
