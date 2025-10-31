using AudioProcessingModuleCs;
using AudioProcessingModuleCs.Media.Dsp.WebRtc;
using MinimalSileroVAD.Core;
using NAudio.Wave;
using Serilog;
using Serilog.Events;
using Serilog.Extensions.Logging;
using System.Runtime.InteropServices;
using System;

namespace MinimalVoiceAgent;

public class Program
{
    private static WaveInEvent? _waveIn;
    private static WaveOutEvent? _waveOut;
    private static BufferedWaveProvider? _bufferedWaveProvider;
    private static VoiceAgentCore? _voiceAgentCore;
    private static CancellationTokenSource _cts = new();
    private static WebRtcFilter? _webRtcFilter;
    private static AudioProcessingConfig _audioConfig = AudioProcessingConfig.CreateDefault();
    private static AudioPacer? _audioPacer;  // New merged pacer

    public static async Task Main(string[] args)
    {
        AddConsoleLogger();

        var lmConfig = await Algos.LoadLanguageModelConfigAsync("profiles/personal.json");
        var sttConfig = await Algos.LoadSttSettingsAsync("sttsettings.json");

        // Initialize TTS provider
        await TtsProviderStreaming.InitializeAsync();

        // Stub tool functions
        var computerToolFunctions = new ComputerToolFunctions();

        // Build Semantic Kernel
        var kernel = Algos.BuildKernel(lmConfig);

        // Initialize core components
        var stt = new SttProviderStreaming();
        await stt.InitializeAsync(sttConfig.SttModelUrl);
        var llm = new LlmChat(lmConfig, computerToolFunctions, kernel);
        var tts = new TtsStreamer();
        _audioPacer = new AudioPacer();

        _voiceAgentCore = new VoiceAgentCore(stt, llm, tts, _audioPacer);
        await _voiceAgentCore.InitializeAsync(new VadSpeechSegmenterSileroV5());

        // Hook up audio reply event to play TTS chunks (now PCM)
        _voiceAgentCore.OnAudioReplyReady += OnAudioReplyReady;

        // Initialize audio formats for WebRTC filter (16kHz mono 16-bit)
        var recordedFormat = new AudioProcessingModuleCs.Media.AudioFormat(
            samplesPerSecond: _audioConfig.ProcessingSampleRate,  // 16000
            millisecondsPerFrame: _audioConfig.FrameSizeMs,  // 20
            channels: _audioConfig.Channels,  // 1
            bitsPerSample: _audioConfig.BitsPerSample  // 16
        );

        var playedFormat = recordedFormat;  // Same for playback

        // Initialize WebRTC filter
        _webRtcFilter = new WebRtcFilter(
            expectedAudioLatency: _audioConfig.SystemLatencyMs,
            filterLength: _audioConfig.FilterLengthMs,
            recordedAudioFormat: recordedFormat,
            playedAudioFormat: playedFormat,
            enableAec: _audioConfig.EnableEchoCancellation,
            enableDenoise: _audioConfig.EnableNoiseSuppression,
            enableAgc: _audioConfig.EnableAutomaticGainControl
        );

        // Enable WebRTC in pacer if configured
        if (_audioConfig.EnableEchoCancellation)
        {
            _audioPacer.EnableWebRtcProcessing(_webRtcFilter);
        }

        // Setup microphone and speaker
        SetupMicrophone();
        SetupSpeaker();

        // Initialize pacer for playback: Adds samples to buffered provider
        _audioPacer.Initialize((pcmFrame) =>
        {
            _bufferedWaveProvider?.AddSamples(pcmFrame, 0, pcmFrame.Length);
        });

        // Play welcome message
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
            WaveFormat = new WaveFormat(rate: 16000, bits: 16, channels: 1)
        };

        _waveIn.BufferMilliseconds = 20;
        _waveIn.DataAvailable += OnMicrophoneDataAvailable;
        _waveIn.StartRecording();

        Log.Information("Microphone capture started at {Format}.", _waveIn.WaveFormat);
    }

    private static void OnMicrophoneDataAvailable(object? sender, WaveInEventArgs e)
    {
        if (_cts.IsCancellationRequested) return;

        byte[] capturedFrame = e.Buffer;  // 640 bytes (20ms at 16kHz 16-bit mono)

        byte[] processedFrame = capturedFrame;  // Default to original if processing fails

        try
        {
            // Ensure frame is exactly 640 bytes (pad with zeros if short, though WaveIn should provide exact)
            if (capturedFrame.Length < 640)
            {
                byte[] paddedFrame = new byte[640];
                Array.Copy(capturedFrame, paddedFrame, capturedFrame.Length);
                capturedFrame = paddedFrame;  // Update for processing
            }

            // Write captured frame to filter (inputs for AEC/NS/AGC)
            _webRtcFilter?.Write(capturedFrame);

            // Prepare output buffer as short[320] (640 bytes == 320 samples)
            short[] outputShorts = new short[320];
            bool moreFrames;  // Out param (typically false for 1:1 processing, but check if API buffers)

            if (_webRtcFilter?.Read(outputShorts, out moreFrames) == true)
            {
                // Convert short[] back to byte[] (zero-copy via MemoryMarshal)
                processedFrame = MemoryMarshal.AsBytes(outputShorts.AsSpan()).ToArray();

                // Handle if more frames are available (rare, but drain if needed)
                if (moreFrames)
                {
                    Log.Warning("Additional frames available after Read; draining not implemented.");
                    // Optionally loop Read until !moreFrames, but for real-time, log and proceed
                }
            }
            else
            {
                Log.Debug("WebRtcFilter.Read returned false; using unprocessed frame.");
            }
        }
        catch (Exception ex)
        {
            Log.Error(ex, "WebRtcFilter failed on captured frame; using unprocessed audio.");
        }

        _voiceAgentCore?.ProcessIncomingAudioChunk(processedFrame);
    }

    private static void SetupSpeaker()
    {
        var playbackFormat = new WaveFormat(rate: 16000, bits: 16, channels: 1);

        _bufferedWaveProvider = new BufferedWaveProvider(playbackFormat)
        {
            DiscardOnBufferOverflow = true
        };

        _waveOut = new WaveOutEvent();
        _waveOut.Init(_bufferedWaveProvider);
        _waveOut.Play();

        Log.Information("Speaker playback started at {Format}.", playbackFormat);
    }

    private static void OnAudioReplyReady(byte[] pcmChunk)
    {
        if (_cts.IsCancellationRequested || _bufferedWaveProvider == null) return;

        try
        {
            // Enqueue full chunk to pacer (it will split, pace, filter, and play)
            _audioPacer!.EnqueueBufferForSendManual(pcmChunk);
        }
        catch (Exception ex)
        {
            Log.Error(ex, "Failed to enqueue TTS audio chunk for playback.");
        }
    }

    private static async Task ShutdownAsync()
    {
        await _cts.CancelAsync();

        _waveIn?.StopRecording();
        _waveIn?.Dispose();

        _waveOut?.Stop();
        _waveOut?.Dispose();

        _audioPacer?.Dispose();

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