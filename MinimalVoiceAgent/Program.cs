using MinimalSileroVAD.Core;
using Serilog;
using Serilog.Events;
using MinimalTextClassifier.Core;
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
                restrictedToMinimumLevel: LogEventLevel.Information,
                outputTemplate: "[{Timestamp:HH:mm:ss} {Level:u3}] {Message:lj}{NewLine}{Exception}")
            .WriteTo.File(
                path: "minimal_voice_agent_log.txt",
                restrictedToMinimumLevel: LogEventLevel.Information,
                rollingInterval: RollingInterval.Infinite,
                rollOnFileSizeLimit: true,
                fileSizeLimitBytes: 100 * 1024 * 1024, // 100 MB
                retainedFileCountLimit: 5)
            .CreateLogger();

        Log.Logger = serilogLogger;
        Log.Information("Serilog configured for minimal voice agent.");
    }
}


public class Program
{
    private static VoiceAgentCore? _voiceAgentCore;
    private static CancellationTokenSource _cts = new();
    private static readonly AudioProcessingConfig _audioConfig = AudioProcessingConfig.CreateDefault();
    private static AudioPacer? _audioPacer;
    private static SoundFlowAudioRouter? _audioRouter;
    private static CleanSpeechDaemonCaptureSource? _cleanSpeechSource;
    private static CleanSpeechDaemonProcess? _cleanSpeechDaemon;

    public static async Task Main(string[] args)
    {
        Algos.AddConsoleLogger();

        var lmConfig = await Algos.LoadLanguageModelConfigAsync("profiles/personal.json");
        var sttConfig = await Algos.LoadSttSettingsAsync("sttsettings.json");

        // Preflight: verify the API key and model are usable before any heavy init
        // (model downloads, audio devices). Fails fast with a clear message for testing.
        var modelCheck = await Algos.ValidateModelAccessAsync(lmConfig);
        if (!modelCheck.Ok)
        {
            Log.Fatal("Startup check failed: {Message}", modelCheck.Message);
            await Console.Error.WriteLineAsync($"Startup check failed: {modelCheck.Message}");
            Environment.Exit(1);
        }
        Log.Information("Startup check passed: {Message}", modelCheck.Message);

        // Initialize TTS provider
        await TtsProviderStreaming.InitializeAsync();

        // Stub tool functions
        var computerToolFunctions = new ComputerToolFunctions();

        // Build Semantic Kernel
        var kernel = Algos.BuildKernel(lmConfig);

        // Initialize core components
        _audioPacer = new AudioPacer();
        var vad = new VadSpeechSegmenter();
        var tts = new TtsStreamer();
        var stt = new SttProviderStreaming();
        await stt.InitializeAsync(sttConfig.SttModelUrl);
        var llm = new LlmChat(lmConfig, computerToolFunctions, kernel);
        var wakeDetector = new MinimalTransformerClassifier("models/deberta_v3_small_fine_tuned_int8.onnx");

        _voiceAgentCore = VoiceAgentCore.CreateBuilder()
            .WithSttProvider(stt)
            .WithLlmChat(llm)
            .WithTtsStreamer(tts)
            .WithAudioPacer(_audioPacer)
            .WithInterruption(false)
            .WithWakeDetector(wakeDetector)
            .WithVadSegmenter(vad)
            .Build();

        // Microphone source selection. When enabled, consume cleaned audio from the external
        // clean-speech-daemon (which removes system playback and noise) instead of capturing the
        // local microphone. On that path the router runs no WebRTC APM, since the daemon already
        // performs echo cancellation. If the daemon is unavailable, fall back to the local mic.
        bool useInternalCapture = true;
        if (sttConfig.Capture.UseCleanSpeechDaemon)
        {
            if (!CleanSpeechDaemonCaptureSource.IsPlatformSupported)
            {
                Log.Warning(
                    "clean-speech-daemon capture requires Linux or macOS; falling back to the local microphone (with APM).");
            }
            else
            {
                // Optionally launch the bundled daemon ourselves before connecting.
                if (sttConfig.Capture.AutoStartDaemon)
                {
                    try
                    {
                        var daemon = new CleanSpeechDaemonProcess(sttConfig.Capture.DaemonDirectory, sttConfig.Capture.SocketPath);
                        bool started = await daemon.EnsureStartedAsync(
                            TimeSpan.FromSeconds(sttConfig.Capture.DaemonStartupTimeoutSeconds), _cts.Token);
                        if (started)
                            _cleanSpeechDaemon = daemon;          // we own it; stop it on shutdown
                        else
                            await daemon.DisposeAsync();           // already running; leave it alone
                    }
                    catch (Exception ex)
                    {
                        Log.Warning(ex, "Failed to auto-start clean-speech-daemon; will try to connect to an existing instance.");
                    }
                }

                var source = new CleanSpeechDaemonCaptureSource(sttConfig.Capture.SocketPath);
                try
                {
                    await source.StartAsync(chunk => _voiceAgentCore.ProcessIncomingAudioChunk(chunk), _cts.Token);
                    _cleanSpeechSource = source;
                    useInternalCapture = false;
                    Log.Information(
                        "Microphone source: clean-speech-daemon ({Path}). Internal capture and APM disabled.",
                        sttConfig.Capture.SocketPath);
                }
                catch (Exception ex)
                {
                    Log.Warning(ex,
                        "clean-speech-daemon capture unavailable; falling back to the local microphone (with APM).");
                    await source.DisposeAsync();
                    if (_cleanSpeechDaemon != null)
                    {
                        await _cleanSpeechDaemon.DisposeAsync();   // we started it but can't use it; stop it
                        _cleanSpeechDaemon = null;
                    }
                }
            }
        }

        _audioRouter = SoundFlowAudioRouter.CreateBuilder()
            .WithAudioPacer(_audioPacer)
            .WithMicChunkHandler(chunk => _voiceAgentCore.ProcessIncomingAudioChunk(chunk))
            .WithCancellationTokenSource(_cts)
            .WithInternalMicCapture(useInternalCapture)
            .Build();

        await _audioRouter.InitializeAsync();

        _voiceAgentCore.OnAudioReplyReady += pcmChunk => _audioRouter.EnqueueTtsChunk(pcmChunk);

        // play welcome
        await Algos.PlayWelcomeMessageAsync(lmConfig, tts, _cts);

        Log.Information("Voice Agent started. Speak now!");
        Console.ReadKey();
        await ShutdownAsync();
    }

    private static async Task ShutdownAsync()
    {
        await _cts.CancelAsync();

        // Stop the external capture source first so no more mic chunks arrive
        if (_cleanSpeechSource != null)
        {
            await _cleanSpeechSource.DisposeAsync();
        }

        // Stop the daemon if we started it
        if (_cleanSpeechDaemon != null)
        {
            await _cleanSpeechDaemon.DisposeAsync();
        }

        // Now safe to shut down core
        if (_voiceAgentCore != null)
        {
            await _voiceAgentCore.ShutdownAsync();
            await _voiceAgentCore.DisposeAsync();
        }

        if (_audioRouter != null)
        {
            await _audioRouter.DisposeAsync();
        }

        Log.Information("Minimal Voice Agent shutdown complete.");
    }
}