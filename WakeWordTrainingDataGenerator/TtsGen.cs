using KokoroSharp;
using KokoroSharp.Core;
using KokoroSharp.Processing;
using KokoroSharp.Utilities;
using Microsoft.ML.OnnxRuntime;
using NAudio.Codecs;
using NAudio.Wave;
using Serilog;
using System.Collections.Concurrent;
using System.Diagnostics;

namespace WakeWordTrainingDataGenerator;

/// <summary>
/// Public static partial class for audio processing algorithms using NAudio where possible.
/// Handles WAV creation via NAudio.
/// </summary>
public static partial class Algos
{
    /// <summary>
    /// Creates a WAV file from raw 16-bit mono PCM bytes using NAudio's WaveFileWriter.
    /// Ensures valid RIFF header, fmt chunk, and data alignment for NAudio compatibility.
    /// </summary>
    public static byte[] CreateWavFromPcm(byte[] rawPcmBytes, int sampleRate)
    {
        if (rawPcmBytes.Length % 2 != 0)
        {
            throw new ArgumentException("Raw PCM must be even length for 16-bit samples.", nameof(rawPcmBytes));
        }

        var format = new WaveFormat(sampleRate, 16, 1); // mono 16-bit PCM
        using var ms = new MemoryStream();
        using (var writer = new WaveFileWriter(ms, format))
        {
            writer.Write(rawPcmBytes, 0, rawPcmBytes.Length); // Explicit byte[] overload for compatibility
        }
        var wavBytes = ms.ToArray();
        return wavBytes;
    }

    /// <summary>
    /// Creates a WAV file from raw 16-bit mono PCM bytes using NAudio's WaveFileWriter.
    /// Ensures valid RIFF header, fmt chunk, and data alignment for NAudio compatibility.
    /// </summary>
    public static async Task<byte[]> CreateWavFromPcmAsync(byte[] rawPcmBytes, int sampleRate)
    {
        if (rawPcmBytes.Length % 2 != 0)
        {
            throw new ArgumentException("Raw PCM must be even length for 16-bit samples.", nameof(rawPcmBytes));
        }

        var format = new WaveFormat(sampleRate, 16, 1); // mono 16-bit PCM
        using var ms = new MemoryStream();
        using (var writer = new WaveFileWriter(ms, format))
        {
            await writer.WriteAsync(rawPcmBytes, 0, rawPcmBytes.Length); // Explicit byte[] overload for compatibility
        }
        var wavBytes = ms.ToArray();
        return wavBytes;
    }

    /// <summary>
    /// Convert a WAV file (as byte array) to raw PCM (16-bit, mono) at a specified target sample rate.
    /// Note that it reads the WAV file header for information on the source audio's sample rate.
    /// </summary>
    /// <param name="wavAudio">WAV file data</param>
    /// <param name="targetSampleRateHz">Desired output sample rate (e.g. 8000, 16000)</param>
    /// <returns>Raw PCM byte array (16-bit mono)</returns>
    public static byte[] ConvertWavToPcm(byte[] wavAudio, int targetSampleRateHz)
    {
        try
        {
            using var wavStream = new MemoryStream(wavAudio);
            using var reader = new WaveFileReader(wavStream);

            var inputFormat = reader.WaveFormat;
            var targetFormat = new WaveFormat(targetSampleRateHz, 16, 1);

            using var resampler = new MediaFoundationResampler(reader, targetFormat)
            {
                ResamplerQuality = 60
            };

            using var outStream = new MemoryStream();
            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = resampler.Read(buffer, 0, buffer.Length)) > 0)
            {
                outStream.Write(buffer, 0, bytesRead);
            }

            byte[] rawPcm = outStream.ToArray();
            Log.Debug($"Converted WAV to PCM: {rawPcm.Length} bytes, {targetSampleRateHz} Hz, 16-bit mono");
            return rawPcm;
        }
        catch (Exception ex)
        {
            Log.Error(ex, "Failed to convert WAV to raw PCM.");
            return Array.Empty<byte>();
        }
    }

    /// <summary>
    /// Encodes 16 bit mono PCM data, irrespective of sample rate (this encoding algo doesn't depend on it.)
    /// </summary>
    public static byte[] EncodePcmToPcmuWithNAudio(byte[] pcmSamples)
    {
        if (pcmSamples.Length % 2 != 0)
        {
            Log.Warning($"[EncodePcmToPcmuWithNAudio] PCM samples length {pcmSamples.Length} is not even, trimming last byte.");
            Array.Resize(ref pcmSamples, pcmSamples.Length - 1);
        }

        int sampleCount = pcmSamples.Length / 2;
        byte[] pcmu = new byte[sampleCount];

        for (int i = 0; i < sampleCount; i++)
        {
            // Combine two bytes into one short (little-endian)
            short sample = (short)(pcmSamples[i * 2] | (pcmSamples[i * 2 + 1] << 8));
            pcmu[i] = MuLawEncoder.LinearToMuLawSample(sample);
        }

        return pcmu;
    }

}

/// <summary>
/// A static service class for offline, high-performance text-to-speech synthesis using KokoroSharp.
/// Outputs 8kHz mono PCMU-encoded audio bytes by default (G.711 μ-law, no header).
/// Alternatively, can output WAV (22kHz mono PCM) via parameter.
/// GPU acceleration auto-enabled with KokoroSharp.GPU.Windows NuGet + CUDA (2-5x speedup).
/// </summary>
public static class TtsProviderStreaming
{
    private static KokoroWavSynthesizer? _synthesizer;
    private static readonly SemaphoreSlim _initLock = new(1, 1);
    private static readonly object _voiceLock = new();
    private static readonly ConcurrentDictionary<string, KokoroVoice> _voiceCache = new();
    private static readonly string _modelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "kokoro.onnx");
    private static readonly Stopwatch _perfTimer = new(); // For RTF logging

    public const string DefaultVoiceKey = "af_heart"; // Natural American English

    /// <summary>
    /// Ensures initialized (internal async).
    /// </summary>
    public static async Task InitializeAsync(CancellationToken ct)
    {
        await _initLock.WaitAsync(ct).ConfigureAwait(false);
        try
        {
            if (_synthesizer is null)
            {
                ct.ThrowIfCancellationRequested();
                _ = KokoroTTS.LoadModel(); // Existing

                if (!File.Exists(_modelPath))
                    throw new InvalidOperationException($"Kokoro model not found at {_modelPath}");

                var sessionOptions = SessionOptions.MakeSessionOptionWithCudaProvider(0); // deviceId=0 for primary GPU
                //var sessionOptions = SessionOptions.MakeSessionOptionWithTensorrtProvider(0); // deviceId=0 for primary GPU
                sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL; // Modern: Full opts for perf
                sessionOptions.EnableMemoryPattern = true; // Reduce VRAM allocs
                sessionOptions.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL; // Stable for TTS
                sessionOptions.IntraOpNumThreads = Environment.ProcessorCount; // Max CPU threads for non-GPU ops
                sessionOptions.InterOpNumThreads = 1; // Single thread for inter-op to reduce context switching

                //sessionOptions.AppendExecutionProvider_CUDA(); // Ensure CUDA provider appended
                Log.Information("Attempting to initialize Kokoro with GPU acceleration...");
                try
                {
                    // Assuming KokoroWavSynthesizer accepts SessionOptions (check/fork source if not)
                    _synthesizer = new KokoroWavSynthesizer(_modelPath, sessionOptions);
                }
                catch (OnnxRuntimeException ex) when (ex.Message.Contains("CUDA") || ex.Message.Contains("cuDNN"))
                {
                    Log.Error(ex, "GPU init failed - falling back to CPU. Check CUDA/cuDNN versions and PATH.");
                    // Fallback: Create CPU session
                    var cpuOptions = new SessionOptions();
                    _synthesizer = new KokoroWavSynthesizer(_modelPath, cpuOptions);
                }
            }
        }
        finally
        {
            _initLock.Release();
        }
    }

    /// <summary>
    /// Synthesizes text to audio asynchronously, returning a seekable MemoryStream (non-streaming, full utterance).
    /// </summary>
    public static async Task<MemoryStream> TextToSpeechAsync(
        string text,
        string? voiceKey = null,
        bool outputAsWav = false,
        int outputSampleRate = 22050,
        CancellationToken ct = default)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            throw new ArgumentException("Text cannot be null or empty.", nameof(text));
        }

        voiceKey = voiceKey ?? DefaultVoiceKey;
        var voice = GetOrCacheVoice(voiceKey);

        // Synthesize to raw PCM bytes
        //var config = new KokoroTTSPipelineConfig();
        byte[] rawPcmBytes = await _synthesizer!.SynthesizeAsync(text, voice);

        if (rawPcmBytes.Length == 0)
        {
            throw new InvalidOperationException("Synthesis returned empty PCM data.");
        }
        
        if (outputAsWav)
        {
            var outputBytes = await Algos.CreateWavFromPcmAsync(rawPcmBytes, outputSampleRate); // Full WAV with valid fmt chunk
            Log.Debug($"Synthesized to WAV: {outputBytes.Length} bytes");
            var ms = new MemoryStream(outputBytes);
            ms.Position = 0;
            return ms;        
        }

        var stream = new MemoryStream(rawPcmBytes);
        stream.Position = 0; // Reset for reading
        return stream;
    }

    /// <summary>
    /// Lists available voice keys synchronously for American English (SRP: Query only).
    /// </summary>
    public static IReadOnlyList<string> ListVoices()
    {
        return KokoroVoiceManager.GetVoices(KokoroLanguage.AmericanEnglish).Select(v => v.Name).ToList();
    }

    /// <summary>
    /// Gets or caches voice (thread-safe).
    /// </summary>
    private static KokoroVoice GetOrCacheVoice(string voiceKey)
    {
        if (!_voiceCache.TryGetValue(voiceKey, out var voice))
        {
            lock (_voiceLock)
            {
                if (!_voiceCache.TryGetValue(voiceKey, out voice))
                {
                    voice = KokoroVoiceManager.GetVoice(voiceKey);
                    if (voice is null)
                    {
                        var available = string.Join(", ", KokoroVoiceManager.GetVoices(KokoroLanguage.AmericanEnglish).Select(v => v.Name));
                        throw new ArgumentException($"Voice key '{voiceKey}' not found. Available: {available}", nameof(voiceKey));
                    }
                    _voiceCache[voiceKey] = voice;
                }
            }
        }
        return voice;
    }
}