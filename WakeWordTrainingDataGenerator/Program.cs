using Accord.Audio;
using Accord.Math;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.TorchSharp;
using Microsoft.ML.Vision;
using NAudio;
using NAudio.Wave;
using NAudio.Wave.SampleProviders;
using Serilog;
using Serilog.Events;
using SixLabors.ImageSharp; // Used to convert Mel-Frequency Cepstral Coefficients (MFCC) feature data into grayscale PNG images for classification with ML.net
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SoundFlow;
using SoundFlow.Interfaces;
using SoundFlow.Modifiers;
using SoundFlow.Providers;
using SoundTouch.Net.NAudioSupport;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using TorchSharp;

namespace WakeWordTrainingDataGenerator;

public static partial class Algos
{
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

public class ImageData
{
    public string? ImagePath { get; set; }
    public string? Label { get; set; }
}

public static class WakeWordDataGenerator
{
    private static readonly string[] SimilarWords = new[]
    {
        "Elena", "Alena", "Lena", "Aline", "Arena", "Elina", "Aleena", "Allina",
        "Alyna", "Malina", "Leena", "Aaliyah", "Adina", "Aliza", "Selena",
        "Sabrina", "Katrina", "Serena", "Latina", "Medina"
    };

    public static async Task GeneratePositiveClipsAsync(string wakeWord, string outputDir, int numClips = 5000, string? noiseDir = null, bool augmentSpeed = true)
    {
        Directory.CreateDirectory(outputDir);
        var voices = TtsProviderStreaming.ListVoices().ToArray();  // Get supported voices as array

        for (int i = 0; i < numClips; i++)
        {
            string voiceKey = voices[i % voices.Length];  // Cycle through voices
            using var audioStream = await TtsProviderStreaming.TextToSpeechAsync(wakeWord, voiceKey, outputAsWav: true);
            byte[] wavBytes = audioStream.ToArray();

            // Validate TTS output
            if (wavBytes.Length < 44) // Minimum WAV header size
            {
                Serilog.Log.Error($"Invalid TTS output for {wakeWord} with voice {voiceKey}: Too small ({wavBytes.Length} bytes)");
                continue;
            }
            try
            {
                using var ms = new MemoryStream(wavBytes);
                using var reader = new WaveFileReader(ms);
                Serilog.Log.Information($"Valid TTS WAV for {wakeWord}_{i}: {reader.WaveFormat.SampleRate} Hz, Length: {reader.Length} bytes");
            }
            catch (Exception ex)
            {
                Serilog.Log.Error($"TTS validation failed for {wakeWord} with voice {voiceKey}: {ex.Message}");
                continue;
            }

            // Apply augmentations if enabled
            if (augmentSpeed || !string.IsNullOrEmpty(noiseDir))
            {
                wavBytes = AugmentAudio(wavBytes, augmentSpeed, noiseDir);
            }

            // Validate after augmentation (optional but recommended)
            try
            {
                using var ms = new MemoryStream(wavBytes);
                using var reader = new WaveFileReader(ms);
                // No log here to avoid spam
            }
            catch (Exception ex)
            {
                Serilog.Log.Error($"Augmentation validation failed for {wakeWord}_{i}: {ex.Message}");
                continue;
            }

            string clipPath = Path.Combine(outputDir, $"{wakeWord.ToLower()}_{i}.wav");
            await File.WriteAllBytesAsync(clipPath, wavBytes);
        }
        Serilog.Log.Information($"Generated {numClips} positive '{wakeWord}' clips in {outputDir}");
    }

    public static async Task GenerateNegativeClipsAsync(string outputDir, int numClipsPerWord = 250, string? noiseDir = null, bool augmentSpeed = true)
    {
        Directory.CreateDirectory(outputDir);
        var voices = TtsProviderStreaming.ListVoices().ToArray();

        int totalClips = 0;
        foreach (var negativeWord in SimilarWords)
        {
            for (int i = 0; i < numClipsPerWord; i++)
            {
                string voiceKey = voices[(totalClips + i) % voices.Length];
                using var audioStream = await TtsProviderStreaming.TextToSpeechAsync(negativeWord, voiceKey, outputAsWav: true);
                byte[] wavBytes = audioStream.ToArray();

                // Validate TTS output
                if (wavBytes.Length < 44) // Minimum WAV header size
                {
                    Serilog.Log.Error($"Invalid TTS output for {negativeWord} with voice {voiceKey}: Too small ({wavBytes.Length} bytes)");
                    continue;
                }
                try
                {
                    using var ms = new MemoryStream(wavBytes);
                    using var reader = new WaveFileReader(ms);
                    Serilog.Log.Information($"Valid TTS WAV for {negativeWord}_{i}: {reader.WaveFormat.SampleRate} Hz, Length: {reader.Length} bytes");
                }
                catch (Exception ex)
                {
                    Serilog.Log.Error($"TTS validation failed for {negativeWord} with voice {voiceKey}: {ex.Message}");
                    continue;
                }

                if (augmentSpeed || !string.IsNullOrEmpty(noiseDir))
                {
                    wavBytes = AugmentAudio(wavBytes, augmentSpeed, noiseDir);
                }

                // Validate after augmentation (optional but recommended)
                try
                {
                    using var ms = new MemoryStream(wavBytes);
                    using var reader = new WaveFileReader(ms);
                    // No log here to avoid spam
                }
                catch (Exception ex)
                {
                    Serilog.Log.Error($"Augmentation validation failed for {negativeWord}_{i}: {ex.Message}");
                    continue;
                }

                string clipPath = Path.Combine(outputDir, $"{negativeWord.ToLower()}_{i}.wav");
                await File.WriteAllBytesAsync(clipPath, wavBytes);
                totalClips++;
            }
        }
        Serilog.Log.Information($"Generated {totalClips} negative clips (similar to 'Alina') in {outputDir}");
    }

    private static byte[] AugmentAudio(byte[] inputWav, bool augmentSpeed, string? noiseDir, int silenceMsBefore = 200, int silenceMsAfter = 200)
    {
        // Read the input WAV to get format and raw samples as float[]
        float[] samples;
        WaveFormat format;
        using (var ms = new MemoryStream(inputWav))
        using (var reader = new WaveFileReader(ms))
        {
            format = reader.WaveFormat;
            var sampleProvider = reader.ToSampleProvider();
            samples = new float[reader.Length / format.BlockAlign];
            sampleProvider.Read(samples, 0, samples.Length);
        }

        // Add leading silence
        if (silenceMsBefore > 0)
        {
            int silenceSamples = (int)(format.SampleRate * silenceMsBefore / 1000.0) * format.Channels;
            float[] leadingSilence = new float[silenceSamples];
            samples = leadingSilence.Concat(samples).ToArray();
        }

        // Add trailing silence
        if (silenceMsAfter > 0)
        {
            int silenceSamples = (int)(format.SampleRate * silenceMsAfter / 1000.0) * format.Channels;
            float[] trailingSilence = new float[silenceSamples];
            samples = samples.Concat(trailingSilence).ToArray();
        }

        // Speed augmentation (random 0.8x to 1.2x, pitch-preserving)
        if (augmentSpeed)
        {
            var random = new Random();
            float speedFactor = (float)(0.8 + random.NextDouble() * 0.4);  // 0.8 to 1.2

            // Force to 16-bit PCM for SoundTouch
            WaveFormat augmentationFormat = new WaveFormat(format.SampleRate, 16, format.Channels);

            // Convert float[] to short[] (16-bit)
            short[] shortSamples = new short[samples.Length];
            for (int j = 0; j < samples.Length; j++)
            {
                shortSamples[j] = (short)(Math.Clamp(samples[j] * 32767f, -32768f, 32767f));
            }

            byte[] rawPcm = new byte[shortSamples.Length * 2];
            Buffer.BlockCopy(shortSamples, 0, rawPcm, 0, rawPcm.Length);

            using var rawStream = new RawSourceWaveStream(rawPcm, 0, rawPcm.Length, augmentationFormat);
            var floatProvider = new Wave16ToFloatProvider(rawStream);
            var soundTouchProvider = new SoundTouchWaveProvider(floatProvider);
            soundTouchProvider.Tempo = speedFactor;

            var sampleProvider = soundTouchProvider.ToSampleProvider();

            // Read processed samples
            var processedSamples = new List<float>();
            float[] buffer = new float[1024 * format.Channels];  // Adjust buffer for channels
            int read;
            while ((read = sampleProvider.Read(buffer, 0, buffer.Length)) > 0)
            {
                processedSamples.AddRange(buffer.AsSpan(0, read).ToArray());
            }
            samples = processedSamples.ToArray();

            // Update format to 16-bit PCM after augmentation
            format = augmentationFormat;
        }

        // Write samples back to WAV temporarily for noise mixing (or further processing)
        byte[] tempWav;
        using (var tempMs = new MemoryStream())
        {
            using (var writer = new WaveFileWriter(tempMs, format))
            {
                writer.WriteSamples(samples, 0, samples.Length);
            }  // Dispose writer to flush header
            tempWav = tempMs.ToArray();
        }
        inputWav = tempWav;

        // Noise addition (if noiseDir provided, mix random noise file at -20dB)
        if (!string.IsNullOrEmpty(noiseDir))
        {
            var noiseFiles = Directory.GetFiles(noiseDir, "*.wav");
            if (noiseFiles.Length > 0)
            {
                var random = new Random();
                string noisePath = noiseFiles[random.Next(noiseFiles.Length)];
                using var noiseReader = new WaveFileReader(noisePath);

                // Resample noise to match if needed
                IWaveProvider finalNoiseReader;
                if (noiseReader.WaveFormat.SampleRate != format.SampleRate || !noiseReader.WaveFormat.Equals(format))
                {
                    finalNoiseReader = new MediaFoundationResampler(noiseReader, format);
                }
                else
                {
                    finalNoiseReader = noiseReader;
                }

                using var signalMs = new MemoryStream(inputWav);
                using var signalReader = new WaveFileReader(signalMs);

                // Mix: signal + 0.1 * noise (-20dB)
                var mixer = new MixingSampleProvider(new[] { signalReader.ToSampleProvider(), new VolumeSampleProvider(finalNoiseReader.ToSampleProvider()) { Volume = 0.1f } });

                using var tempMs = new MemoryStream();
                using (var writer = new WaveFileWriter(tempMs, mixer.WaveFormat))
                {
                    float[] buffer = new float[1024 * format.Channels];
                    int read;
                    while ((read = mixer.Read(buffer, 0, buffer.Length)) > 0)
                    {
                        writer.WriteSamples(buffer, 0, read);
                    }
                }  // Dispose writer here to update header
                inputWav = tempMs.ToArray();

                if (finalNoiseReader != noiseReader)
                {
                    //finalNoiseReader.Dispose();
                }
            }
        }

        return inputWav;
    }

    public static async Task GenerateFeaturesAsync(string positiveDir, string negativeDir, string featuresDir, int fixedFrames = 100, int coeffCount = 13)
    {
        Directory.CreateDirectory(featuresDir);
        string posFeatures = Path.Combine(featuresDir, "positive");
        string negFeatures = Path.Combine(featuresDir, "negative");
        Directory.CreateDirectory(posFeatures);
        Directory.CreateDirectory(negFeatures);

        // Process positive clips
        foreach (var file in Directory.GetFiles(positiveDir, "*.wav"))
        {
            try
            {
                var mfcc = ComputeMfcc(file, coeffCount);
                var padded = PadOrTrim(mfcc, fixedFrames);
                var image = CreateImageFromMfcc(padded);
                string outPath = Path.Combine(posFeatures, Path.GetFileNameWithoutExtension(file) + ".png");
                await image.SaveAsPngAsync(outPath);
            }
            catch (Exception ex)
            {
                Serilog.Log.Warning($"Skipping invalid positive file {file}: {ex.Message}");
            }
        }

        // Process negative clips
        foreach (var file in Directory.GetFiles(negativeDir, "*.wav"))
        {
            try
            {
                var mfcc = ComputeMfcc(file, coeffCount);
                var padded = PadOrTrim(mfcc, fixedFrames);
                var image = CreateImageFromMfcc(padded);
                string outPath = Path.Combine(negFeatures, Path.GetFileNameWithoutExtension(file) + ".png");
                await image.SaveAsPngAsync(outPath);
            }
            catch (Exception ex)
            {
                Serilog.Log.Warning($"Skipping invalid negative file {file}: {ex.Message}");
            }
        }

        Serilog.Log.Information($"Generated MFCC feature images in {featuresDir}");
    }

    private static double[][] ComputeMfcc(string path, int coeffCount)
    {
        using var reader = new WaveFileReader(path);
        var format = reader.WaveFormat;
        var samples = new float[(int)(reader.Length / format.BlockAlign)];
        reader.ToSampleProvider().Read(samples, 0, samples.Length);

        // Create real-valued Signal from samples (handles multichannel, but assuming mono for wake-word clips)
        var signal = Signal.FromArray(samples, format.Channels, format.SampleRate, SampleFormat.Format32BitIeeeFloat);

        // If stereo (channels > 1), optionally extract mono channel (e.g., left) to simplify
        if (format.Channels > 1)
        {
            float[] monoSamples = new float[samples.Length / format.Channels];
            for (int i = 0; i < monoSamples.Length; i++)
            {
                monoSamples[i] = samples[i * format.Channels];  // Left channel
            }
            signal = Signal.FromArray(monoSamples, 1, format.SampleRate, SampleFormat.Format32BitIeeeFloat);
        }

        var mfcc = new MelFrequencyCepstrumCoefficient(cepstrumCount: coeffCount);
        var features = mfcc.Transform(signal);
        return features.Select(f => f.Descriptor).ToArray();
    }

    private static double[,] PadOrTrim(double[][] mfcc, int fixedFrames)
    {
        if (mfcc.Length == 0) return new double[0, 0];

        int coeffs = mfcc[0].Length;
        int frames = mfcc.Length;
        double[,] result = new double[coeffs, fixedFrames];
        int useFrames = Math.Min(frames, fixedFrames);
        for (int f = 0; f < useFrames; f++)
        {
            for (int c = 0; c < coeffs; c++)
            {
                result[c, f] = mfcc[f][c];
            }
        }
        // Padded areas remain 0
        return result;
    }

    private static Image<L8> CreateImageFromMfcc(double[,] mfcc)
    {
        int height = mfcc.GetLength(0); // coeffs
        int width = mfcc.GetLength(1); // frames
        var image = new Image<L8>(width, height);

        // Find min and max, ignoring zeros if padded
        double min = double.MaxValue;
        double max = double.MinValue;
        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                var v = mfcc[i, j];
                if (v != 0)
                {
                    min = Math.Min(min, v);
                    max = Math.Max(max, v);
                }
            }
        }
        if (min == double.MaxValue) // All zero, unlikely
        {
            min = 0;
            max = 1;
        }
        if (min == max) max = min + 1;

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                double val = mfcc[y, x];
                byte pixel = (byte)(255 * (val - min) / (max - min));
                image[x, y] = new L8(pixel);
            }
        }
        return image;
    }

    public static void TrainModel(string featuresDir, string modelPath = "wakeword_model.zip")
    {
        var mlContext = new MLContext();

        // Load image data (updated to use relative paths)
        var images = new List<ImageData>();
        foreach (var dir in Directory.GetDirectories(featuresDir))
        {
            string label = Path.GetFileName(dir);
            string relativeSubdir = label;  // e.g., "positive" or "negative"
            foreach (var file in Directory.GetFiles(dir, "*.png"))
            {
                string relativePath = Path.Combine(relativeSubdir, Path.GetFileName(file));  // e.g., "negative\alena_114.png"
                images.Add(new ImageData { ImagePath = relativePath, Label = label });
            }
        }

        if (!images.Any())
        {
            Serilog.Log.Error($"No images found in {featuresDir}. Run data generation first.");
            return;
        }

        IDataView imageData = mlContext.Data.LoadFromEnumerable(images);

        IDataView dataset = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "LabelKey", inputColumnName: "Label")
            .Append(mlContext.Transforms.LoadRawImageBytes(outputColumnName: "Image", imageFolder: featuresDir, inputColumnName: "ImagePath"))
            .Fit(imageData)
            .Transform(imageData);

        var options = new ImageClassificationTrainer.Options
        {
            FeatureColumnName = "Image",
            LabelColumnName = "LabelKey",
            Arch = ImageClassificationTrainer.Architecture.ResnetV250,
            BatchSize = 16,
            Epoch = 50,
            MetricsCallback = (metrics) =>
            {
                if (metrics != null)
                    Serilog.Log.Information($"Micro Accuracy: {metrics.Train?.Accuracy ?? 0.0f}");
            },
            WorkspacePath = Path.Combine(featuresDir, "workspace")
        };

        var pipeline = mlContext.MulticlassClassification.Trainers.ImageClassification(options)
            .Append(mlContext.Transforms.Conversion.MapKeyToValue(outputColumnName: "PredictedLabel", inputColumnName: "PredictedLabel"));

        ITransformer model = pipeline.Fit(dataset);

        // Save ML.NET model
        mlContext.Model.Save(model, dataset.Schema, modelPath);
        Serilog.Log.Information($"Model saved to {modelPath}");
    }

}

public static class Program
{

    public class Prediction
    {
        [ColumnName("PredictedLabel")]
        public string PredictedLabel { get; set; } = string.Empty;
    }

    public static void Predict(string modelPath, string featuresDir, string mfccImagePath)
    {
        var mlContext = new MLContext();

        // Load the trained model
        ITransformer model = mlContext.Model.Load(modelPath, out var schema);

        // Prepare raw input (use relative path, e.g., "positive/alina_0.png")
        string relativePath = Path.GetRelativePath(featuresDir, mfccImagePath).Replace("\\", "/");
        var rawInput = new ImageData { ImagePath = relativePath, Label = "dummy" };
        IDataView rawData = mlContext.Data.LoadFromEnumerable(new[] { rawInput });

        // Create the transform pipeline to match training schema: ImagePath/Label -> Image/LabelKey
        var dataTransform = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "LabelKey", inputColumnName: "Label")
            .Append(mlContext.Transforms.LoadRawImageBytes(outputColumnName: "Image", imageFolder: featuresDir, inputColumnName: "ImagePath"));

        // Apply transform
        IDataView transformedData = dataTransform.Fit(rawData).Transform(rawData);

        // Apply the model to get predictions
        IDataView predictions = model.Transform(transformedData);

        // Extract the prediction using enumerable (avoids cursor issues)
        var predictionResult = mlContext.Data.CreateEnumerable<Prediction>(predictions, reuseRowObject: false).FirstOrDefault();
        if (predictionResult != null)
        {
            Console.WriteLine($"Prediction for {mfccImagePath}: {predictionResult.PredictedLabel}");  // "positive" or "negative"
        }
        else
        {
            Console.WriteLine("Prediction failed: No output generated.");
        }
    }

    public static void Main(string[] args)
    {
        Algos.AddConsoleLogger();
        Log.Information($"{Environment.CurrentDirectory}");

        if (args.Length < 3)
        {
            Serilog.Log.Error("Usage: WakeWordTrainingDataGenerator <wakeWord> <positiveDir> <negativeDir> [numPositive=5000] [numNegativePerWord=250] [noiseDir]");
            return;
        }

        //RunTraining(args);

        // run inference
        string featuresDir = "features";
        string imagePath = Path.Combine(featuresDir, "positive", "alina_0.png");
        Predict("wakeword_model.zip", featuresDir, imagePath);
        imagePath = Path.Combine(featuresDir, "positive", "alina_3869.png");
        Predict("wakeword_model.zip", featuresDir, imagePath);
        imagePath = Path.Combine(featuresDir, "negative", "aleena_212.png");
        Predict("wakeword_model.zip", featuresDir, imagePath);
    }

    private static async Task RunTraining(string[] args)
    {
        string wakeWord = args[0];
        string positiveDir = args[1];
        string negativeDir = args[2];
        int numPositive = args.Length >= 4 ? int.Parse(args[3]) : 5000;
        int numNegativePerWord = args.Length >= 5 ? int.Parse(args[4]) : 250;
        string? noiseDir = args.Length >= 6 ? args[5] : null;

        // Generate positives
        await WakeWordDataGenerator.GeneratePositiveClipsAsync(wakeWord, positiveDir, numPositive, noiseDir);

        // Generate negatives
        await WakeWordDataGenerator.GenerateNegativeClipsAsync(negativeDir, numNegativePerWord, noiseDir);

        // Generate MFCC features as images
        string featuresDir = Path.Combine(Path.GetDirectoryName(positiveDir) ?? ".", "features");
        await WakeWordDataGenerator.GenerateFeaturesAsync(positiveDir, negativeDir, featuresDir);

        // Train and save model
        WakeWordDataGenerator.TrainModel(featuresDir);
    }
}