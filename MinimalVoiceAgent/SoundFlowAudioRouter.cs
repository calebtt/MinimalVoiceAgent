using Serilog;
using SoundFlow.Abstracts.Devices;
using SoundFlow.Backends.MiniAudio;
using SoundFlow.Components;
using SoundFlow.Enums;
using SoundFlow.Extensions.WebRtc.Apm.Modifiers;
using SoundFlow.Interfaces;
using SoundFlow.Structs;
using System.Collections.Concurrent;

namespace MinimalVoiceAgent;

// Custom TTS Data Provider for streaming chunks (implements ISoundDataProvider from SoundFlow)
public class TtsQueueDataProvider : ISoundDataProvider
{
    private readonly ConcurrentQueue<byte[]> _chunks = new();
    private readonly AudioFormat _format;
    private int _position;
    private int _length = -1;  // Live stream
    private bool _isDisposed = false;
    public TtsQueueDataProvider(AudioFormat format)
    {
        _format = format;
    }

    public int Position => _position;
    public int Length => _length;
    public bool CanSeek => false;
    public SampleFormat SampleFormat => _format.Format;
    public int SampleRate => _format.SampleRate;
    public bool IsDisposed => _isDisposed;

    public event EventHandler<PositionChangedEventArgs>? PositionChanged;
    public event EventHandler<EventArgs>? EndOfStreamReached;

    public int ReadBytes(Span<float> buffer)
    {
        int totalRead = 0;
        while (totalRead < buffer.Length && !_chunks.IsEmpty)
        {
            if (_chunks.TryDequeue(out byte[]? chunk))
            {
                if (chunk == null) continue;

                float[] floats = Algos.ConvertPcmToFloat(chunk.AsSpan());
                int toCopy = Math.Min(buffer.Length - totalRead, floats.Length);
                floats.AsSpan(0, toCopy).CopyTo(buffer.Slice(totalRead));
                totalRead += toCopy;
                _position += toCopy;
            }
        }

        if (totalRead == 0 && _chunks.IsEmpty)
        {
            EndOfStreamReached?.Invoke(this, EventArgs.Empty);
        }

        return totalRead;
    }

    public void Seek(int offset)
    {
        // Not seekable
    }

    public void EnqueueChunk(byte[] chunk)
    {
        _chunks.Enqueue(chunk);
        PositionChanged?.Invoke(this, new(_position));
    }

    public void Dispose()
    {
        _isDisposed = true;
    }
}

public class SoundFlowAudioRouter : IAsyncDisposable
{
    private MiniAudioEngine? _soundEngine;
    private FullDuplexDevice? _duplexDevice;
    private TtsQueueDataProvider? _ttsQueueProvider;
    private SoundPlayer? _ttsPlayer;
    private Recorder? _micRecorder;
    private WebRtcApmModifier? _apmModifier;
    private int _periodFrames = 0;

    private readonly AudioPacer _audioPacer;
    private readonly Action<byte[]> _micChunkHandler;
    private readonly CancellationTokenSource _cts;

    private bool _isDisposed;

    private AudioFormat _workingFormat;

    private SoundFlowAudioRouter(AudioPacer audioPacer, Action<byte[]> micChunkHandler, CancellationTokenSource cts)
    {
        _audioPacer = audioPacer ?? throw new ArgumentNullException(nameof(audioPacer));
        _micChunkHandler = micChunkHandler ?? throw new ArgumentNullException(nameof(micChunkHandler));
        _cts = cts ?? throw new ArgumentNullException(nameof(cts));
    }

    public async Task InitializeAsync()
    {
        Log.Information("Initializing SoundFlow audio system...");

        _soundEngine = new MiniAudioEngine();
        _soundEngine.UpdateDevicesInfo();

        var captureInfo = _soundEngine.CaptureDevices.FirstOrDefault(d => d.IsDefault);
        if (captureInfo == default)
            captureInfo = _soundEngine.CaptureDevices.First();

        var renderInfo = _soundEngine.PlaybackDevices.FirstOrDefault(d => d.IsDefault);
        if (renderInfo == default)
            renderInfo = _soundEngine.PlaybackDevices.First();

        Log.Information("Using capture: {Cap} / render: {Ren}", captureInfo.Name, renderInfo.Name);

        // Find working format
        _workingFormat = ProbeWorkingFormat(_soundEngine, captureInfo, renderInfo);

        // Initialize full duplex device
        try
        {
            _duplexDevice = _soundEngine.InitializeFullDuplexDevice(renderInfo, captureInfo, _workingFormat);
            Log.Debug("FullDuplexDevice initialized successfully with format: {0} Hz {1} ch {2}", _workingFormat.SampleRate, _workingFormat.Channels, _workingFormat.Format);
        }
        catch (Exception ex)
        {
            Log.Error(ex, "Failed to initialize FullDuplexDevice.");
            throw;
        }

        // WebRTC APM setup
        _apmModifier = new WebRtcApmModifier(
            device: _duplexDevice,
            aecEnabled: true,
            nsEnabled: true,
            agc1Enabled: true,
            agc2Enabled: true,
            nsLevel: SoundFlow.Extensions.WebRtc.Apm.NoiseSuppressionLevel.High
        );

        // Microphone recorder
        _micRecorder = new Recorder(
            captureDevice: _duplexDevice.CaptureDevice,
            callback: (samples, channels) =>
            {
                if (_cts.IsCancellationRequested) return;

                try
                {
                    if (_periodFrames == 0 && samples.Length > 0)
                    {
                        _periodFrames = samples.Length;  // Intuit once: e.g., 960
                        Log.Information("Intuited period size: {0} frames ({1}ms at {2}Hz)",
                            _periodFrames, (_periodFrames * 1000 / _workingFormat.SampleRate), _workingFormat.SampleRate);
                    }

                    const int targetFrames = 320;  // 20ms at 16kHz (expected by APM/VAD)
                    for (int i = 0; i < samples.Length; i += targetFrames)
                    {
                        int sliceLength = Math.Min(targetFrames, samples.Length - i);
                        var subSamples = samples.Slice(i, sliceLength);
                        byte[] micChunk = Algos.ConvertFloatToPcm(subSamples);
                        _micChunkHandler(micChunk);
                    }
                }
                catch (Exception ex)
                {
                    Log.Error(ex, "Error processing mic chunk.");
                }
            }
    );

        _micRecorder.AddModifier(_apmModifier);
        _micRecorder.StartRecording();

        // TTS playback
        _ttsQueueProvider = new TtsQueueDataProvider(_workingFormat);
        _ttsPlayer = new SoundPlayer(_soundEngine, _workingFormat, _ttsQueueProvider);
        _duplexDevice.PlaybackDevice.MasterMixer.AddComponent(_ttsPlayer);
        _ttsPlayer.Play();

        // Start devices
        _duplexDevice.Start();
        _audioPacer.Initialize(chunk => _ttsQueueProvider.EnqueueChunk(chunk));

        Log.Information("Audio initialized: {Rate}Hz {Ch}ch {Fmt}", _workingFormat.SampleRate, _workingFormat.Channels, _workingFormat.Format);
    }

    public void EnqueueTtsChunk(byte[] pcmChunk)
    {
        if (_cts.IsCancellationRequested || pcmChunk == null || pcmChunk.Length == 0) return;

        try
        {
            _audioPacer.EnqueueBufferForSendManual(pcmChunk);
        }
        catch (Exception ex)
        {
            Log.Error(ex, "Failed to enqueue TTS audio chunk for playback.");
        }
    }

    private static AudioFormat ProbeWorkingFormat(MiniAudioEngine engine,
                                                  DeviceInfo captureInfo,
                                                  DeviceInfo renderInfo)
    {
        var formatsToTry = new[]
        {
            new AudioFormat { SampleRate = 16000, Channels = 1, Format = SampleFormat.S16 },  // Prioritize for pipeline compatibility
            new AudioFormat { SampleRate = 16000, Channels = 2, Format = SampleFormat.S16 },  // Stereo at 16kHz
            new AudioFormat { SampleRate = 48000, Channels = 1, Format = SampleFormat.S16 },
            new AudioFormat { SampleRate = 44100, Channels = 1, Format = SampleFormat.S16 },
            new AudioFormat { SampleRate = 48000, Channels = 2, Format = SampleFormat.S16 },
            new AudioFormat { SampleRate = 44100, Channels = 2, Format = SampleFormat.S16 },
        };

        foreach (var fmt in formatsToTry)
        {
            try
            {
                using var test = engine.InitializeFullDuplexDevice(renderInfo, captureInfo, fmt);
                test.Start();
                Thread.Sleep(500);  // Extended test for stability (e.g., capture a few frames)
                test.Stop();
                Log.Information("Device accepted {0} Hz {1} ch {2}", fmt.SampleRate, fmt.Channels, fmt.Format);
                return fmt;
            }
            catch (Exception ex)
            {
                Log.Warning("Rejected {0} Hz {1} ch {2}: {3}", fmt.SampleRate, fmt.Channels, fmt.Format, ex.Message);
            }
        }

        throw new InvalidOperationException("No compatible capture format found.");
    }

    private async Task ShutdownAsync()
    {
        try
        {
            // Stop audio input first to prevent further callbacks
            _micRecorder?.StopRecording();
            _duplexDevice?.Stop();

            // Dispose remaining resources
            _micRecorder?.Dispose();
            _ttsPlayer?.Stop();
            _ttsPlayer?.Dispose();
            _ttsQueueProvider?.Dispose();
            _duplexDevice?.Dispose();
            _audioPacer?.Dispose();
            _apmModifier?.Dispose();
            _soundEngine?.Dispose();

            Log.Information("SoundFlowAudioRouter shutdown complete.");
        }
        catch (Exception ex)
        {
            Log.Error(ex, "Error during SoundFlowAudioRouter shutdown");
        }
    }

    protected virtual void Dispose(bool disposing)
    {
        if (_isDisposed)
            return;

        if (disposing)
        {
            // Sync disposal: Only managed resources; no async here to avoid blocks
            try
            {
                // Other sync cleanups if needed
            }
            catch (Exception ex)
            {
                Log.Error(ex, "Error during managed disposal in SoundFlowAudioRouter.");
            }
        }

        // Unmanaged cleanup here if needed

        _isDisposed = true;
    }

    protected virtual async ValueTask DisposeAsyncCore()
    {
        if (_isDisposed)
            return;

        if (_cts?.IsCancellationRequested != true)
        {
            try
            {
                await _cts.CancelAsync();
                await ShutdownAsync();
            }
            catch (OperationCanceledException)
            {
                Log.Debug("Shutdown canceled during async disposal.");
            }
            catch (Exception ex)
            {
                Log.Error(ex, "Error during async shutdown in SoundFlowAudioRouter disposal.");
            }
        }

        // Other managed cleanup if needed

        _isDisposed = true;
    }

    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    public async ValueTask DisposeAsync()
    {
        await DisposeAsyncCore().ConfigureAwait(false);
        GC.SuppressFinalize(this);
    }

    ~SoundFlowAudioRouter()
    {
        Dispose(false);
    }

    public static Builder CreateBuilder()
    {
        return new Builder();
    }

    public class Builder
    {
        private AudioPacer? _audioPacer;
        private Action<byte[]>? _micChunkHandler;
        private CancellationTokenSource? _cts;

        public Builder WithAudioPacer(AudioPacer audioPacer)
        {
            _audioPacer = audioPacer;
            return this;
        }

        public Builder WithMicChunkHandler(Action<byte[]> handler)
        {
            _micChunkHandler = handler;
            return this;
        }

        public Builder WithCancellationTokenSource(CancellationTokenSource cts)
        {
            _cts = cts;
            return this;
        }

        public SoundFlowAudioRouter Build()
        {
            if (_audioPacer == null) throw new InvalidOperationException("Audio pacer is required.");
            if (_micChunkHandler == null) throw new InvalidOperationException("Mic chunk handler is required.");
            if (_cts == null) throw new InvalidOperationException("Cancellation token source is required.");

            return new SoundFlowAudioRouter(_audioPacer, _micChunkHandler, _cts);
        }
    }
}