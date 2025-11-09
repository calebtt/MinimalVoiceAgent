using MinimalSileroVAD.Core;
using Serilog;
using Serilog.Events;
using Serilog.Extensions.Logging;
using SoundFlow.Abstracts.Devices;  // For AudioDevice, AudioCaptureDevice
using SoundFlow.Backends.MiniAudio;  // For MiniAudioEngine
using SoundFlow.Components;
using SoundFlow.Enums;  // For SampleFormat
using SoundFlow.Extensions.WebRtc.Apm.Modifiers;  // For WebRtcApmModifier
using SoundFlow.Interfaces;
using SoundFlow.Structs;  // For AudioFormat
using System.Collections.Concurrent;

namespace MinimalVoiceAgent;

public static partial class Algos
{
    public static float[] ConvertPcmToFloat(ReadOnlySpan<byte> pcmBytes)
    {
        if (pcmBytes.Length % 2 != 0) throw new ArgumentException("PCM must be 16-bit aligned.", nameof(pcmBytes));
        int numSamples = pcmBytes.Length / 2;
        var floatSamples = new float[numSamples];
        for (int i = 0; i < numSamples; i++)
        {
            short sample = BitConverter.ToInt16(pcmBytes.Slice(i * 2, 2));
            floatSamples[i] = Math.Clamp(sample / 32767f, -1f, 1f);  // Signed 16-bit normalize with headroom
        }
        return floatSamples;
    }

    public static byte[] ConvertFloatToPcm(ReadOnlySpan<float> floatSamples)
    {
        int numSamples = floatSamples.Length;
        var pcmBytes = new byte[numSamples * 2];
        Span<byte> byteSpan = pcmBytes.AsSpan();
        for (int i = 0; i < numSamples; i++)
        {
            short sample = (short)(Math.Clamp(floatSamples[i], -1f, 1f) * 32767f);
            if (!BitConverter.TryWriteBytes(byteSpan.Slice(i * 2, 2), sample))
                throw new InvalidOperationException($"Failed to encode sample {i}.");  // Rare: buffer overflow
        }
        return pcmBytes;
    }

    public static SoundFlow.Extensions.WebRtc.Apm.NoiseSuppressionLevel MapNoiseLevel(NoiseSuppressionLevel customLevel)
    {
        return customLevel switch
        {
            NoiseSuppressionLevel.Conservative => SoundFlow.Extensions.WebRtc.Apm.NoiseSuppressionLevel.Low,
            NoiseSuppressionLevel.Moderate => SoundFlow.Extensions.WebRtc.Apm.NoiseSuppressionLevel.Moderate,
            NoiseSuppressionLevel.Aggressive => SoundFlow.Extensions.WebRtc.Apm.NoiseSuppressionLevel.High,
            _ => SoundFlow.Extensions.WebRtc.Apm.NoiseSuppressionLevel.Moderate
        };
    }

    public static async Task PlayWelcomeMessageAsync(LanguageModelConfig lmConfig, TtsStreamer tts, CancellationTokenSource cts)
    {
        if (cts.IsCancellationRequested)
            return;

        var welcomeText = lmConfig.WelcomeMessage;
        if (string.IsNullOrWhiteSpace(welcomeText))
        {
            Log.Warning("No welcome message defined in settings; skipping playback.");
            return;
        }

        try
        {
            Log.Information("Generating and playing welcome message: '{WelcomeText}'", welcomeText);

            // Start streaming TTS for welcome (queues chunks via OnAudioReplyReady)
            await tts.StartStreamingAsync(welcomeText, ct: cts.Token);
        }
        catch (OperationCanceledException)
        {
            Log.Debug("Welcome message playback canceled.");
        }
        catch (Exception ex)
        {
            Log.Error(ex, "Failed to generate/play welcome message.");
        }
    }

    public static void AddConsoleLogger()
    {
        var serilogLogger = new LoggerConfiguration()
            .Enrich.FromLogContext()
            .MinimumLevel.Verbose()
            .MinimumLevel.Override("Microsoft", LogEventLevel.Warning)
            .MinimumLevel.Override("System", LogEventLevel.Warning)
            .WriteTo.Console(
                restrictedToMinimumLevel: LogEventLevel.Debug,
                outputTemplate: "[{Timestamp:HH:mm:ss} {Level:u3}] {Message:lj}{NewLine}{Exception}")
            .WriteTo.File(
                path: "minimal_voice_agent_log.txt",
                restrictedToMinimumLevel: LogEventLevel.Debug,
                rollingInterval: RollingInterval.Infinite,
                rollOnFileSizeLimit: true,
                fileSizeLimitBytes: 100 * 1024 * 1024, // 100 MB
                retainedFileCountLimit: 5)
            .CreateLogger();

        var factory = new SerilogLoggerFactory(serilogLogger);
        Log.Logger = serilogLogger;

        Log.Information("Serilog configured for minimal voice agent.");
    }
}

// Custom TTS Data Provider for streaming chunks (implements ISoundDataProvider)
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

public class Program
{
    private static VoiceAgentCore? _voiceAgentCore;
    private static CancellationTokenSource _cts = new();
    private static AudioProcessingConfig _audioConfig = AudioProcessingConfig.CreateDefault();
    private static AudioPacer? _audioPacer;

    // SoundFlow graph components
    private static MiniAudioEngine? _soundEngine;
    private static FullDuplexDevice? _duplexDevice;
    private static TtsQueueDataProvider? _ttsQueueProvider;
    private static SoundPlayer? _ttsPlayer;
    private static Recorder? _micRecorder;
    private static WebRtcApmModifier? _apmModifier;
    private static int _periodFrames = 0;  // Store intuited period size (samples.Length for mono)

    public static async Task Main(string[] args)
    {
        Algos.AddConsoleLogger();

        var lmConfig = await Algos.LoadLanguageModelConfigAsync("profiles/personal.json");
        var sttConfig = await Algos.LoadSttSettingsAsync("sttsettings.json");

        // Initialize TTS provider
        await TtsProviderStreaming.InitializeAsync();

        // Stub tool functions
        var computerToolFunctions = new ComputerToolFunctions();

        // Build Semantic Kernel
        var kernel = Algos.BuildKernel(lmConfig);

        // Initialize core components
        _audioPacer = new AudioPacer();
        var vad = new VadSpeechSegmenterSileroV5();
        var tts = new TtsStreamer();
        var stt = new SttProviderStreaming();
        await stt.InitializeAsync(sttConfig.SttModelUrl);
        var llm = new LlmChat(lmConfig, computerToolFunctions, kernel);

        _voiceAgentCore = new VoiceAgentCore(stt, llm, tts, _audioPacer, doUseInterruption: false);
        await _voiceAgentCore.InitializeAsync(vad);

        // Wire TTS to pacer (pacer enqueues to provider)
        _voiceAgentCore.OnAudioReplyReady += OnAudioReplyReady;

        // Initialize engine and devices
        await InitializeAudioAsync();

        // play welcome
        await Algos.PlayWelcomeMessageAsync(lmConfig, tts, _cts);

        Log.Information("Voice Agent started. Speak now!");
        Console.ReadKey();
        await ShutdownAsync();
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

    private static async Task<AudioFormat> InitializeAudioAsync()
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
        var workingFormat = ProbeWorkingFormat(_soundEngine, captureInfo, renderInfo);

        // Initialize full duplex device
        try
        {
            _duplexDevice = _soundEngine.InitializeFullDuplexDevice(renderInfo, captureInfo, workingFormat);
            Log.Debug("FullDuplexDevice initialized successfully with format: {0} Hz {1} ch {2}", workingFormat.SampleRate, workingFormat.Channels, workingFormat.Format);
        }
        catch (Exception ex)
        {
            Log.Error(ex, "Failed to initialize FullDuplexDevice.");
            throw;
        }

        // Configure audio config for pipeline
        //_audioConfig.ProcessingSampleRate = 16000;
        //_audioConfig.Channels = workingFormat.Channels;

        // WebRTC APM setup
        _apmModifier = new WebRtcApmModifier(
            device: _duplexDevice,
            aecEnabled: true,
            nsEnabled: true,
            agc1Enabled: false,
            agc2Enabled: true,
            nsLevel: SoundFlow.Extensions.WebRtc.Apm.NoiseSuppressionLevel.High
        );
                
        // Microphone recorder
        _micRecorder = new Recorder(
            captureDevice: _duplexDevice.CaptureDevice,
            callback: (Span<float> samples, Capability channels) =>
        {
            if (_cts.IsCancellationRequested) return;

            try
            {
                if (_periodFrames == 0 && samples.Length > 0)
                {
                    _periodFrames = samples.Length;  // Intuit once: e.g., 960
                    Log.Information("Intuited period size: {0} frames ({1}ms at {2}Hz)",
                        _periodFrames, (_periodFrames * 1000 / workingFormat.SampleRate), workingFormat.SampleRate);
                }

                const int targetFrames = 320;  // 20ms at 16kHz (expected by APM/VAD)
                for (int i = 0; i < samples.Length; i += targetFrames)
                {
                    int sliceLength = Math.Min(targetFrames, samples.Length - i);
                    var subSamples = samples.Slice(i, sliceLength);
                    byte[] micChunk = Algos.ConvertFloatToPcm(subSamples);
                    _voiceAgentCore?.ProcessIncomingAudioChunk(micChunk);
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
        _ttsQueueProvider = new TtsQueueDataProvider(workingFormat);
        _ttsPlayer = new SoundPlayer(_soundEngine, workingFormat, _ttsQueueProvider);
        _duplexDevice.PlaybackDevice.MasterMixer.AddComponent(_ttsPlayer);
        _ttsPlayer.Play();

        // Start devices
        _duplexDevice.Start();
        _audioPacer.Initialize(chunk => _ttsQueueProvider.EnqueueChunk(chunk));

        Log.Information("Audio initialized: {Rate}Hz {Ch}ch {Fmt}", workingFormat.SampleRate, workingFormat.Channels, workingFormat.Format);
        return workingFormat;
    }


    private static void OnAudioReplyReady(byte[] pcmChunk)
    {
        if (_cts.IsCancellationRequested || pcmChunk == null || pcmChunk.Length == 0) return;

        try
        {
            _audioPacer?.EnqueueBufferForSendManual(pcmChunk);
        }
        catch (Exception ex)
        {
            Log.Error(ex, "Failed to enqueue TTS audio chunk for playback.");
        }
    }

    private static async Task ShutdownAsync()
    {
        await _cts.CancelAsync();

        if (_voiceAgentCore != null)
        {
            await _voiceAgentCore.ShutdownAsync();
            _voiceAgentCore.Dispose();
        }

        _micRecorder?.StopRecording();
        _micRecorder?.Dispose();

        _ttsPlayer?.Stop();
        _ttsPlayer?.Dispose();
        _ttsQueueProvider?.Dispose();

        _duplexDevice?.Stop();
        _duplexDevice?.Dispose();

        _audioPacer?.Dispose();

        _apmModifier?.Dispose();

        _soundEngine?.Dispose();

        Log.Information("Minimal Voice Agent shutdown complete.");
    }
}