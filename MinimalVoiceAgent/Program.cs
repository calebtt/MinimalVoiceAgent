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

        _voiceAgentCore = VoiceAgentCore.CreateBuilder()
            .WithSttProvider(stt)
            .WithLlmChat(llm)
            .WithTtsStreamer(tts)
            .WithAudioPacer(_audioPacer)
            .WithInterruption(false)
            .WithWakeIdentifier("Alina")
            .WithVadSegmenter(vad)
            .Build();

        _audioRouter = SoundFlowAudioRouter.CreateBuilder()
            .WithAudioPacer(_audioPacer)
            .WithMicChunkHandler(chunk => _voiceAgentCore.ProcessIncomingAudioChunk(chunk))
            .WithCancellationTokenSource(_cts)
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

        // Now safe to shut down core (disposes VAD/SileroModel)
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