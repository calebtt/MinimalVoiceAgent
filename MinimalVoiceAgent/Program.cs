using NAudio.Wave;
using Serilog;
using Serilog.Events;
using Serilog.Extensions.Logging;
using MinimalSileroVAD.Core;

namespace MinimalVoiceAgent;

public class Program
{
    private static WaveInEvent? _waveIn;
    private static WaveOutEvent? _waveOut;
    private static BufferedWaveProvider? _bufferedWaveProvider;
    private static VoiceAgentCore? _voiceAgentCore;
    private static CancellationTokenSource _cts = new();

    public static async Task Main(string[] args)
    {
        AddConsoleLogger();

        var lmConfig = await Algos.LoadLanguageModelConfigAsync("profiles/personal.json");
        var sttConfig = await Algos.LoadSttSettingsAsync("sttsettings.json");

        // Initialize TTS provider (assumes global/static init as in original)
        await TtsProviderStreaming.InitializeAsync();

        // Stub tool functions since no SIP (hangup and transfer do nothing)
        var computerToolFunctions = new ComputerToolFunctions();

        // Build Semantic Kernel
        var kernel = Algos.BuildKernel(lmConfig);

        // Initialize core components
        var stt = new SttProviderStreaming(sttConfig.SttModelUrl);
        var llm = new LlmChat(lmConfig, computerToolFunctions, kernel);
        var tts = new TtsStreamer(); // Assumes TtsStreamer is defined; handles streaming TTS
        var audioPacer = new RtpAudioPacer(); // Assumes RtpAudioPacer is defined; for buffering/playing TTS

        _voiceAgentCore = new VoiceAgentCore(stt, llm, tts, audioPacer);
        await _voiceAgentCore.InitializeAsync(new VadSpeechSegmenterSileroV5());

        // Hook up audio reply event to play TTS chunks
        _voiceAgentCore.OnAudioReplyReady += OnAudioReplyReady;

        // Setup microphone and speaker
        SetupMicrophone();
        SetupSpeaker();

        // Play welcome message immediately after init
        await PlayWelcomeMessageAsync(lmConfig, tts);

        Log.Information("Minimal Voice Agent started. Speak into the microphone. Press any key to exit.");

        Console.ReadKey();

        await ShutdownAsync();
    }

    private static async Task PlayWelcomeMessageAsync(LanguageModelConfig lmConfig, TtsStreamer tts)
    {
        if (_cts.IsCancellationRequested) return;

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
            await tts.StartStreamingAsync(welcomeText, ct: _cts.Token);
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

    private static void SetupMicrophone()
    {
        _waveIn = new WaveInEvent
        {
            // Capture at 16kHz, 16-bit, mono (matches expected input for VAD/STT)
            WaveFormat = new WaveFormat(rate: 16000, bits: 16, channels: 1)
        };

        // Buffer size for ~20ms frames (matches VAD frame length)
        _waveIn.BufferMilliseconds = 20;
        _waveIn.DataAvailable += OnMicrophoneDataAvailable;
        _waveIn.StartRecording();

        Log.Information("Microphone capture started at {Format}.", _waveIn.WaveFormat);
    }

    private static void OnMicrophoneDataAvailable(object? sender, WaveInEventArgs e)
    {
        if (_cts.IsCancellationRequested) return;

        // Process the incoming audio chunk (16kHz PCM)
        // No resampling needed if captured at 16kHz
        // TODO: Integrate AEC (e.g., via WebRtcFilter) here to prevent self-detection of TTS output as input speech
        _voiceAgentCore?.ProcessIncomingAudioChunk(e.Buffer);
    }

    private static void SetupSpeaker()
    {
        // Assume TTS outputs PCMU (8kHz, 8-bit, mono ulaw)
        // Convert to PCM for playback
        var playbackFormat = new WaveFormat(rate: 8000, bits: 16, channels: 1); // Standard PCM for WaveOut

        _bufferedWaveProvider = new BufferedWaveProvider(playbackFormat)
        {
            DiscardOnBufferOverflow = true // Prevent backlog during interruptions
        };

        _waveOut = new WaveOutEvent();
        _waveOut.Init(_bufferedWaveProvider);
        _waveOut.Play();

        Log.Information("Speaker playback started at {Format}.", playbackFormat);
    }

    private static void OnAudioReplyReady(byte[] pcmuChunk)
    {
        if (_cts.IsCancellationRequested || _bufferedWaveProvider == null) return;

        try
        {
            // Convert PCMU (ulaw) to linear PCM 16-bit for playback
            var pcmChunk = AudioAlgos.ConvertPcmuToPcm16kHz(pcmuChunk); // Assumes method exists in AudioAlgos
            var pcm8KhzChunk = AudioAlgos.ResamplePcmWithNAudio(pcmChunk, inputSampleRate: 16000, outputSampleRate: 8000);

            _bufferedWaveProvider.AddSamples(pcm8KhzChunk, 0, pcm8KhzChunk.Length);
        }
        catch (Exception ex)
        {
            Log.Error(ex, "Failed to play TTS audio chunk.");
        }
    }

    private static async Task ShutdownAsync()
    {
        _cts.Cancel();

        _waveIn?.StopRecording();
        _waveIn?.Dispose();

        _waveOut?.Stop();
        _waveOut?.Dispose();

        if (_voiceAgentCore != null)
        {
            await _voiceAgentCore.ShutdownAsync();
            _voiceAgentCore.Dispose();
        }

        Log.Information("Minimal Voice Agent shutdown complete.");
    }

    private static void AddConsoleLogger()
    {
        var serilogLogger = new LoggerConfiguration()
            .Enrich.FromLogContext()
            .MinimumLevel.Verbose()
            .MinimumLevel.Override("Microsoft", LogEventLevel.Warning)
            .MinimumLevel.Override("System", LogEventLevel.Warning)
            .WriteTo.Console(
                restrictedToMinimumLevel: LogEventLevel.Information,
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