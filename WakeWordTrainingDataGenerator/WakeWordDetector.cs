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

namespace WakeWordTrainingDataGenerator;

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

    [ColumnName("Score")]
    public float[] Score { get; set; } = Array.Empty<float>();  // [neg_prob, pos_prob]
}

public static class WakeWordUtils
{
    private const int SampleRate = 16000;  // Fixed for wake word audio
    private const int CepstrumCount = 13;  // Number of MFCC coefficients
    private const int FixedFrames = 100;

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

    public static byte[] ExtractMfccAndGeneratePng(byte[] pcmBytes)
    {
        if (pcmBytes == null || pcmBytes.Length == 0)
            return Array.Empty<byte>();

        float[] floatSamples = PcmToFloatSamples(pcmBytes);
        float[][] mfccMatrix = ComputeMfccMatrix(floatSamples);
        float[][] fixedMatrix = EnsureFixedFrameCount(mfccMatrix);
        return MfccMatrixToPngBytes(fixedMatrix);
    }

    private static float[] PcmToFloatSamples(byte[] pcmBytes)
    {
        return TrainingAlgos.ConvertPcmToFloat(pcmBytes.AsSpan());
    }

    private static float[][] ComputeMfccMatrix(float[] floatSamples)
    {
        var signal = Signal.FromArray(floatSamples, 1, SampleRate, SampleFormat.Format32BitIeeeFloat);
        var extractor = new MelFrequencyCepstrumCoefficient(cepstrumCount: CepstrumCount);
        var descriptors = extractor.Transform(signal);

        if (!descriptors.Any())
            return new float[0][];

        return descriptors
            .Select(d => d.Descriptor)
            .Select(row => row.Select(v => (float)v).ToArray())
            .ToArray();
    }

    private static float[][] EnsureFixedFrameCount(float[][] mfccMatrix)
    {
        if (mfccMatrix.Length == 0)
            return CreateZeroPaddedMatrix(FixedFrames);

        float[][] result = new float[FixedFrames][];

        int framesToCopy = Math.Min(mfccMatrix.Length, FixedFrames);
        for (int i = 0; i < framesToCopy; i++)
            result[i] = mfccMatrix[i].ToArray(); // deep copy

        // Zero-pad remaining frames
        for (int i = framesToCopy; i < FixedFrames; i++)
            result[i] = new float[CepstrumCount];

        return result;
    }

    private static float[][] CreateZeroPaddedMatrix(int frames)
    {
        var matrix = new float[frames][];
        for (int i = 0; i < frames; i++)
            matrix[i] = new float[CepstrumCount];
        return matrix;
    }

    private static byte[] MfccMatrixToPngBytes(float[][] matrix)
    {
        int width = CepstrumCount;   // 13
        int height = FixedFrames;     // 100

        using var img = new Image<L8>(width, height);

        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
            {
                float coeff = matrix[y][x];
                // Exact training normalization: ~-10..10 → 0..255
                byte pixel = (byte)Math.Clamp((coeff + 10f) * (255f / 20f), 0, 255);
                img[x, y] = new L8(pixel);
            }

        using var ms = new MemoryStream();
        img.SaveAsPng(ms);
        return ms.ToArray();
    }

}

public class WakeWordDetector : IDisposable
{
    private readonly MLContext _mlContext = new();
    private readonly ITransformer _model;
    private const float ConfidenceThreshold = 0.9f;  // Tune: for FP reduction

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
    public (bool isWake, float confidence) IsWakeWord(byte[] pcm16kHzMono)
    {
        try
        {
            byte[] imageBytes = WakeWordUtils.ExtractMfccAndGeneratePng(pcm16kHzMono);
            if (imageBytes.Length == 0)
            {
                Log.Warning("No MFCC frames extracted; treating as negative.");
                return (false, 0.0f);
            }

            var result = WakeWordUtils.PerformPrediction(_mlContext, _model, imageBytes);
            float rawPosProb = result.Score.Length > 1 ? result.Score[1] : 0f;

            const float Threshold = 0.95f;  // This is the magic number that works right now

            bool isPositive = rawPosProb > Threshold;

            Log.Information("Wake detection: {Label} (conf: {Prob:F3})",
                isPositive ? "POSITIVE" : "NEGATIVE", rawPosProb);

            return (isPositive, rawPosProb);
        }
        catch (Exception ex)
        {
            Log.Error(ex, "Wake word detection failed");
            return (false, 0.0f);
        }
    }

    public void Dispose()
    {
        // Note: MLContext.Dispose() omitted as per version compatibility; resources released on GC
    }
}