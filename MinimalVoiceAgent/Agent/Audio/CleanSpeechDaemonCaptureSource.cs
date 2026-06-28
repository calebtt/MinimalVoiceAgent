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
/// agent's own WebRTC APM is not used on this path. Disconnects are retried automatically with
/// exponential backoff until cancelled.
/// </para>
/// </summary>
public sealed class CleanSpeechDaemonCaptureSource : IMicrophoneCaptureSource
{
    internal const int MaxHeaderBytes = 8192;
    internal static readonly TimeSpan ConnectTimeout = TimeSpan.FromSeconds(5);
    internal static readonly TimeSpan HeaderReadTimeout = TimeSpan.FromSeconds(5);
    private static readonly TimeSpan ReconnectDelayInitial = TimeSpan.FromSeconds(1);
    private static readonly TimeSpan ReconnectDelayMax = TimeSpan.FromSeconds(10);
    private static readonly TimeSpan BufferDuration = TimeSpan.FromSeconds(30);

    private const int TargetSampleRate = 16000;
    private const int FrameSamples = 320; // 20 ms at 16 kHz
    private const double BufferHighWatermarkRatio = 0.8;

    private readonly string _socketPath;
    private Action<byte[]>? _onChunk;
    private Task? _sessionLoop;
    private CancellationTokenSource? _linkedCts;
    private Socket? _socket;
    private NetworkStream? _networkStream;
    private int _bufferOverflowLogCooldown;

    public CleanSpeechDaemonCaptureSource(string socketPath)
    {
        if (string.IsNullOrWhiteSpace(socketPath))
            throw new ArgumentException("Socket path is required.", nameof(socketPath));
        _socketPath = socketPath;
    }

    /// <summary>Unix domain sockets are only available on Linux and macOS.</summary>
    public static bool IsPlatformSupported => OperatingSystem.IsLinux() || OperatingSystem.IsMacOS();

    public async Task StartAsync(Action<byte[]> onChunk, CancellationToken ct)
    {
        ArgumentNullException.ThrowIfNull(onChunk);
        if (!IsPlatformSupported)
            throw new PlatformNotSupportedException("clean-speech-daemon capture requires Linux or macOS (Unix domain sockets).");

        _onChunk = onChunk;
        _linkedCts = CancellationTokenSource.CreateLinkedTokenSource(ct);

        // Fail fast at startup so Program can fall back to the local microphone.
        var firstSession = await OpenSessionAsync(_linkedCts.Token);
        _socket = firstSession.Socket;
        _networkStream = firstSession.Stream;
        Log.Information(
            "clean-speech-daemon connected: {Rate} Hz {Ch} ch s16le -> resampling to {Target} Hz",
            firstSession.SampleRate, firstSession.Channels, TargetSampleRate);

        _sessionLoop = Task.Run(
            () => RunSessionLoopAsync(firstSession.SampleRate, _linkedCts.Token),
            CancellationToken.None);
    }

    private async Task RunSessionLoopAsync(int initialSampleRate, CancellationToken ct)
    {
        var reconnectDelay = ReconnectDelayInitial;

        await ReadLoopAsync(_networkStream!, initialSampleRate, _onChunk!, ct);

        while (!ct.IsCancellationRequested)
        {
            Log.Warning("clean-speech-daemon socket closed by the daemon; reconnecting.");
            CloseTransport();

            try
            {
                await Task.Delay(reconnectDelay, ct);
            }
            catch (OperationCanceledException) when (ct.IsCancellationRequested)
            {
                break;
            }

            try
            {
                var session = await OpenSessionAsync(ct);
                _socket = session.Socket;
                _networkStream = session.Stream;
                Log.Information(
                    "clean-speech-daemon reconnected: {Rate} Hz -> resampling to {Target} Hz",
                    session.SampleRate, TargetSampleRate);
                reconnectDelay = ReconnectDelayInitial;

                await ReadLoopAsync(session.Stream, session.SampleRate, _onChunk!, ct);
            }
            catch (OperationCanceledException) when (ct.IsCancellationRequested)
            {
                break;
            }
            catch (Exception ex)
            {
                Log.Warning(ex, "clean-speech-daemon reconnect failed; retrying in {DelayMs} ms.",
                    reconnectDelay.TotalMilliseconds);
                reconnectDelay = TimeSpan.FromMilliseconds(
                    Math.Min(reconnectDelay.TotalMilliseconds * 2, ReconnectDelayMax.TotalMilliseconds));
            }
        }
    }

    private static async Task<DaemonSession> OpenSessionAsync(string socketPath, CancellationToken ct)
    {
        if (!File.Exists(socketPath))
            throw new FileNotFoundException(
                $"clean-speech-daemon socket not found at '{socketPath}'. Is the daemon running ('clean-speech-daemon run')?");

        var socket = new Socket(AddressFamily.Unix, SocketType.Stream, ProtocolType.Unspecified);
        try
        {
            using var connectTimeout = CancellationTokenSource.CreateLinkedTokenSource(ct);
            connectTimeout.CancelAfter(ConnectTimeout);
            await socket.ConnectAsync(new UnixDomainSocketEndPoint(socketPath), connectTimeout.Token);

            var stream = new NetworkStream(socket, ownsSocket: true);
            var header = await ReadHeaderLineAsync(stream, ct);
            var (sampleRate, channels) = ParseHeader(header);
            return new DaemonSession(socket, stream, sampleRate, channels);
        }
        catch
        {
            socket.Dispose();
            throw;
        }
    }

    private Task<DaemonSession> OpenSessionAsync(CancellationToken ct)
        => OpenSessionAsync(_socketPath, ct);

    /// <summary>Reads bytes up to the first newline and returns them as the (UTF-8) header line.</summary>
    internal static async Task<string> ReadHeaderLineAsync(Stream stream, CancellationToken ct)
    {
        using var headerTimeout = CancellationTokenSource.CreateLinkedTokenSource(ct);
        headerTimeout.CancelAfter(HeaderReadTimeout);

        var bytes = new List<byte>(256);
        var one = new byte[1];
        var sawNewline = false;

        while (bytes.Count < MaxHeaderBytes)
        {
            int n = await stream.ReadAsync(one.AsMemory(0, 1), headerTimeout.Token);
            if (n == 0)
                throw new EndOfStreamException("Socket closed before the metadata newline.");
            if (one[0] == (byte)'\n')
            {
                sawNewline = true;
                break;
            }
            bytes.Add(one[0]);
        }

        if (!sawNewline)
        {
            throw new InvalidDataException(
                $"Daemon metadata header exceeded {MaxHeaderBytes} bytes without a newline.");
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
        var buffered = new BufferedWaveProvider(new WaveFormat(sourceRate, 16, 1))
        {
            ReadFully = false,
            DiscardOnBufferOverflow = true,
            BufferDuration = BufferDuration,
        };
        var maxBufferedBytes = (int)(BufferDuration.TotalSeconds * sourceRate * sizeof(short));

        ISampleProvider samples = buffered.ToSampleProvider();
        if (sourceRate != TargetSampleRate)
            samples = new WdlResamplingSampleProvider(samples, TargetSampleRate);

        var readBuf = new byte[8192];
        var frame = new float[FrameSamples];
        int framePos = 0;

        while (!ct.IsCancellationRequested)
        {
            int n = await stream.ReadAsync(readBuf.AsMemory(), ct);
            if (n == 0)
                return;

            var beforeBuffered = buffered.BufferedBytes;
            buffered.AddSamples(readBuf, 0, n);
            MaybeLogBufferPressure(buffered.BufferedBytes, maxBufferedBytes, beforeBuffered);

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

    private void MaybeLogBufferPressure(int bufferedBytes, int maxBytes, int beforeBytes)
    {
        if (maxBytes <= 0 || bufferedBytes < maxBytes * BufferHighWatermarkRatio)
            return;
        if (bufferedBytes <= beforeBytes)
            return;

        if (_bufferOverflowLogCooldown > 0)
        {
            _bufferOverflowLogCooldown--;
            return;
        }

        _bufferOverflowLogCooldown = 50;
        Log.Warning(
            "clean-speech-daemon PCM buffer high ({Buffered}/{Max} bytes); overflow may drop audio",
            bufferedBytes, maxBytes);
    }

    private void CloseTransport()
    {
        try { _networkStream?.Dispose(); } catch { /* best effort */ }
        try { _socket?.Dispose(); } catch { /* best effort */ }
        _networkStream = null;
        _socket = null;
    }

    public async ValueTask DisposeAsync()
    {
        try
        {
            if (_linkedCts is { IsCancellationRequested: false })
                await _linkedCts.CancelAsync();
        }
        catch { /* best effort */ }

        if (_sessionLoop is not null)
        {
            try { await _sessionLoop; } catch { /* swallow shutdown races */ }
        }

        CloseTransport();
        _linkedCts?.Dispose();
    }

    private sealed record DaemonSession(Socket Socket, NetworkStream Stream, int SampleRate, int Channels);
}