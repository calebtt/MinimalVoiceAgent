using Accord.Audio;
using Accord.Audio.Windows;  // For RaisedCosineWindow (if needed for explicit windowing)
using Microsoft.ML;
using Microsoft.ML.Data;
using Serilog;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace MinimalVoiceAgent;

public class ProcessedInput
{
    [ColumnName("Image")]
    public byte[] Image { get; set; } = Array.Empty<byte>();

    [ColumnName("LabelKey")]
    public uint LabelKey { get; set; } = 0;  // Dummy value; not used in prediction
}

public class Prediction
{
    [ColumnName("PredictedLabel")]
    public string PredictedLabel { get; set; } = string.Empty;
}

public static class WakeWordUtils
{
    private const int SampleRate = 16000;  // Fixed for wake word audio
    private const int CepstrumCount = 13;  // Number of MFCC coefficients

    /// <summary>
    /// Reusable from training: Perform ML.NET prediction on raw PNG bytes.
    /// </summary>
    public static Prediction PerformPrediction(MLContext mlContext, ITransformer model, byte[] imageBytes)
    {
        // Directly create input matching post-transform schema (no file paths)
        var input = new ProcessedInput
        {
            Image = imageBytes,
            LabelKey = 0  // Dummy; model ignores during prediction
        };
        IDataView inputData = mlContext.Data.LoadFromEnumerable(new[] { input });

        // Apply the model (no need for dataTransform—input is already "Image"/"LabelKey")
        IDataView predictions = model.Transform(inputData);

        // Extract result (exact copy from Predict)
        var predictionResult = mlContext.Data.CreateEnumerable<Prediction>(predictions, reuseRowObject: false).FirstOrDefault();
        if (predictionResult == null)
        {
            throw new InvalidOperationException("Prediction failed: No output generated.");
        }
        return predictionResult;
    }

    /// <summary>
    /// Reusable from training: MFCC extraction + PNG generation (in-memory).
    /// </summary>
    public static byte[] ExtractMfccAndGeneratePng(byte[] pcmBytes)  // Input: Raw 16kHz PCM prefix
    {
        // Step 1: Convert 16-bit PCM bytes to float[]
        float[] floatSamples = Algos.ConvertPcmToFloat(pcmBytes.AsSpan());

        // Step 2: Load as Signal (mono, float format)
        Signal signal = Signal.FromArray(floatSamples, channels: 1, SampleRate, SampleFormat.Format32BitIeeeFloat);

        // Step 3: Transform to MFCC (match training params)
        var mfccExtractor = new MelFrequencyCepstrumCoefficient(cepstrumCount: CepstrumCount);
        var descriptors = mfccExtractor.Transform(signal);

        if (!descriptors.Any())
        {
            throw new InvalidOperationException("No MFCC frames extracted.");
        }

        // Step 4: Extract coefficients: frames x coeffs (double[][] from Descriptors)
        double[][] features = descriptors.Select(d => d.Descriptor).ToArray();
        float[][] mfccMatrix = features.Select(row => row.Select(d => (float)d).ToArray()).ToArray();

        // Step 5: Convert MFCC matrix to grayscale PNG bytes (match training normalization)
        int width = mfccMatrix[0].Length;  // Coeffs (13)
        int height = mfccMatrix.Length;    // Frames (~40 for 800ms)
        using var img = new Image<L8>(width, height);
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                float coeff = mfccMatrix[y][x];
                // Training normalization: Scale MFCC range (~-10 to 10) to 0-255
                byte pixel = (byte)Math.Clamp((coeff + 10f) * (255f / 20f), 0, 255);
                img[x, y] = new L8(pixel);
            }
        }

        // Output to bytes (no file save)
        using var ms = new MemoryStream();
        img.SaveAsPng(ms);
        byte[] imageBytes = ms.ToArray();

        Log.Debug("Generated {Height}x{Width} MFCC PNG ({Size} bytes)", height, width, imageBytes.Length);
        return imageBytes;
    }
}

public class WakeWordDetector : IDisposable
{
    private readonly MLContext _mlContext = new();
    private readonly ITransformer _model;

    public WakeWordDetector(string modelPath)
    {
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"Model file not found: {modelPath}");

        _model = _mlContext.Model.Load(modelPath, out _);
        Log.Information("WakeWordDetector loaded model from {ModelPath}", modelPath);
    }

    /// <summary>
    /// Classifies the prefix of the input audio (first 800ms) as wake word present (true) or not (false).
    /// </summary>
    public bool IsWakeWord(byte[] pcm16kHzMono)
    {
        try
        {
            // Step 1: Extract MFCC features (match training: 13 coeffs, Hamming window)
            byte[] imageBytes = WakeWordUtils.ExtractMfccAndGeneratePng(pcm16kHzMono);

            if (imageBytes.Length == 0)
            {
                Log.Warning("No MFCC frames extracted; treating as negative.");
                return false;
            }

            // Step 2: Predict using in-memory byte[] (matches post-LoadRawImageBytes schema)
            var result = WakeWordUtils.PerformPrediction(_mlContext, _model, imageBytes);

            bool isPositive = result.PredictedLabel == "positive";
            Log.Information("Wake detection: {Label}", isPositive ? "POSITIVE" : "NEGATIVE");
            return isPositive;
        }
        catch (Exception ex)
        {
            Log.Error(ex, "Wake word detection failed");
            return false;  // Fail-safe: Don't trigger on error
        }
    }

    public void Dispose()
    {
        // Note: MLContext.Dispose() omitted as per version compatibility; resources released on GC
    }
}