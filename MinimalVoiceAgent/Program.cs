using AudioProcessingModuleCs.Media.Dsp.WebRtc;
using NAudio.Wave;
using Serilog;
using Serilog.Events;
using Serilog.Extensions.Logging;
using System;
using System.Buffers;
using System.Runtime.InteropServices;
using MinimalSileroVAD.Core;

namespace MinimalVoiceAgent;

public class Program
{
    private static WaveInEvent? _waveIn;
    private static WaveOutEvent? _waveOut;
    private static BufferedWaveProvider? _bufferedWaveProvider;
    private static VoiceAgentCore? _voiceAgentCore;
    private static CancellationTokenSource _cts = new();
    private static WebRtcFilter? _webRtcFilter = null;
    private static AudioProcessingConfig _audioConfig = AudioProcessingConfig.CreateDefault();
    private static AudioPacer? _audioPacer;  // New merged pacer
    private static long _micFrameCount = 0; // Static counter for mic events (thread-safe via Interlocked)

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
        var vad = new VadSpeechSegmenterSileroV5();
        var tts = new TtsStreamer();
        var stt = new SttProviderStreaming();
        await stt.InitializeAsync(sttConfig.SttModelUrl);
        var llm = new LlmChat(lmConfig, computerToolFunctions, kernel);
        _audioPacer = new AudioPacer();

        _voiceAgentCore = new VoiceAgentCore(stt, llm, tts, _audioPacer);
        await _voiceAgentCore.InitializeAsync(vad);

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
            enableAec: true,
            enableDenoise: false,
            enableAgc: false
        );

        _audioPacer.EnableWebRtcProcessing(_webRtcFilter);

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
        if (_cts.IsCancellationRequested)
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

        // Force mono downmix if stereo (assume _waveIn.WaveFormat.Channels == 2)
        byte[] capturedFrame;
        Span<byte> capturedSpan = e.Buffer.AsSpan(); // Zero-copy view

        if (_waveIn.WaveFormat.Channels == 2)
        {
            capturedFrame = new byte[capturedSpan.Length / 2];
            for (int i = 0; i < capturedFrame.Length / 2; i++) // Average L/R
            {
                short left = (short)((capturedSpan[i * 4 + 1] << 8) | capturedSpan[i * 4]);
                short right = (short)((capturedSpan[i * 4 + 3] << 8) | capturedSpan[i * 4 + 2]);
                short avg = (short)((left + right) / 2);
                capturedFrame[i * 2] = (byte)(avg & 0xFF);
                capturedFrame[i * 2 + 1] = (byte)((avg >> 8) & 0xFF);
            }
            Log.Debug("Downmixed stereo to mono (len {Len})", capturedFrame.Length);
        }
        else
        {
            capturedFrame = capturedSpan.ToArray();
        }

        try
        {
            // Ensure frame is exactly 640 bytes (pad with zeros if short)
            if (capturedFrame.Length < 640)
            {
                byte[] paddedFrame = new byte[640];
                Array.Copy(capturedFrame, paddedFrame, capturedFrame.Length);
                capturedFrame = paddedFrame;
            }

            // Write captured frame to filter (inputs for AEC/NS/AGC)
            _webRtcFilter?.Write(capturedFrame);

            // Rent short[] from pool (320 samples; modern: Shared pool for low-alloc)
            short[] outputArray = ArrayPool<short>.Shared.Rent(320);
            try
            {
                Span<short> outputShorts = outputArray.AsSpan(0, 320); // Span view over rented array

                bool moreFrames = false; // Out param
                if (_webRtcFilter?.Read(outputArray, out moreFrames) == true)  // Pass rented array
                {
                    // Convert Span<short> to byte[] (zero-copy via MemoryMarshal)
                    var frameBytes = MemoryMarshal.AsBytes(outputShorts[..320]).ToArray(); // Full frame

                    // Feed to voice agent (process one frame at a time for real-time)
                    _voiceAgentCore?.ProcessIncomingAudioChunk(frameBytes);

                    if (moreFrames)
                    {
                        Log.Debug("WebRTC has additional buffered frames post-Read (consider draining if backlog grows).");
                    }
                }
                else
                {
                    Log.Debug("WebRtcFilter.Read returned false; using unprocessed frame for this chunk.");
                    _voiceAgentCore?.ProcessIncomingAudioChunk(capturedFrame);
                }
            }
            finally
            {
                // Return to pool (robust: Always, even on exception)
                ArrayPool<short>.Shared.Return(outputArray, clearArray: true); // Clear for security/privacy (audio data)
            }
        }
        catch (Exception ex)
        {
            Log.Error(ex, "WebRtcFilter processing failed on captured frame; using unprocessed audio.");
            _voiceAgentCore?.ProcessIncomingAudioChunk(capturedFrame); // Ensure forwarding
        }
    }

    

    // Helper: Efficient RMS calc (inlined, no allocs beyond input)
    //private static double CalculateRmsVolume(ReadOnlySpan<byte> buffer)
    //{
    //    if (buffer.Length == 0) return 0;

    //    double sum = 0;
    //    for (int i = 0; i < buffer.Length; i += 2) // 16-bit samples
    //    {
    //        short sample = (short)(buffer[i + 1] << 8 | buffer[i]);
    //        sum += sample * sample;
    //    }
    //    return Math.Sqrt(sum / (buffer.Length / 2)); // RMS (0-32768 range)
    //}

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