using Accord.Audio;
using Accord.Math;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Vision;
using NAudio.Wave;
using Serilog;
using Serilog.Events;
using SixLabors.ImageSharp; // Used to convert Mel-Frequency Cepstral Coefficients (MFCC) feature data into grayscale PNG images for classification with ML.net
using SixLabors.ImageSharp.PixelFormats;


namespace WakeWordTrainingDataGenerator;

internal static class TrainingAlgos
{
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

}

public class ImageData
{
    public string? ImagePath { get; set; }
    public string? Label { get; set; }
}

public static class WakeWordDataGenerator
{
    // TTS'd Alina positive command examples.
    private static readonly string[] PositiveCommands = new[]
    {
        "turn off the lights", "what's the time", "play some music", "set a timer for five minutes",
        "open the front door", "read my emails", "call mom", "how's the weather today",
        "book a flight to New York", "remind me about the meeting", "adjust the thermostat to 72",
        "send a text to Sarah", "check my calendar for tomorrow", "play the news podcast",
        "order pizza from Domino's", "find a recipe for pasta", "start a workout timer",
        "pause the movie", "increase the volume", "switch to dark mode",
        "tell me a joke", "summarize the day's news", "navigate to the nearest coffee shop",
        "create a shopping list", "translate this to Spanish", "lock the smart door",
        "dim the bedroom lights", "wake me up at seven", "queue up my playlist",
        "scan for updates", "backup my photos", "print the document",
        "reset the fitness tracker", "sync my devices", "arm the security system",
        "what time is it?", "can you dim the screen?", "dim the screen.", "dim the screen brightness."
        // Add 50+ more as needed, cycling voices and positions
    };

    // Loose-trigger training.
    // TODO utilize this.
    private static readonly string[] PositiveSimilarCommands = new[]
    {
        "Aleena, check the fridge temperature", "Elina, brew some coffee", "Aleena, lock the windows",
        "Elina, read the latest headlines", "Aleena, play jazz music", "Elina, set alarm for noon",
        "Aleena, dim the hallway light", "Elina, call the office", "Aleena, what's on Netflix",
        "Elina, order groceries online", "Aleena, track my steps", "Elina, meditate with me",
        // Expand with more similar-sounding prefixes (e.g., Alena) for loose triggering training
        "Alena, turn on the fan", "Lena, close the curtains"
    };

    private static readonly string[] NegativeSentences = new[]
    {
        "The quick brown fox jumps over the lazy dog", "I need to buy groceries tomorrow",
        "Pass the salt please", "What's for dinner tonight", "Call Sarah later",
        "The meeting starts at nine sharp", "I forgot my keys again", "Rain is forecast for afternoon",
        "Pick up milk on the way home", "The cat scratched the couch", "Traffic was terrible today",
        "Bake cookies for the party", "Charge the phone overnight", "Water the plants before leaving",
        "Fold the laundry neatly", "Sweep the kitchen floor", "Recycle the old newspapers",
        "Mow the lawn this weekend", "Fix the leaky faucet", "Paint the fence white",
        "Organize the desk drawers", "Dust the shelves thoroughly", "Vacuum under the bed",
        "Wash the car on Sunday", "Trim the hedges evenly", "Plant new flowers in the yard",
        "Read that book you borrowed", "Watch the documentary tonight", "Listen to the radio show",
        "Cook stir-fry for lunch", "Brew tea in the afternoon", "Slice the vegetables thinly",
        "Boil pasta al dente", "Grill chicken for dinner", "Bake bread from scratch",
        "Stir the soup gently", "Chop onions finely", "Peel the potatoes first",
        "Mash the garlic cloves", "Season with herbs", "Simmer on low heat",
        // Dissimilar/distant negatives (no Alina-like sounds)
        "Quantum physics fascinates me", "Solve the puzzle quickly", "Debate the ethics deeply",
        "Explore the ancient ruins", "Invent a new gadget", "Compose a symphony",
        "Photograph the sunset", "Sculpt the clay model", "Paint the landscape vividly",
        "Dance under the stars", "Sing a lullaby softly", "Whisper secrets quietly",
        "Shout with joy loudly", "Laugh heartily together", "Cry rivers of tears",
        "Run a marathon swiftly", "Swim laps endlessly", "Climb mountains boldly",
        "Dive into the ocean", "Fly kites on windy days", "Sail boats across lakes"
    };

    public static async Task GeneratePositiveClipsAsync(string wakeWord, string outputDir, int numClips = 5000, string? noiseDir = null, bool augment = true)
    {
        Directory.CreateDirectory(outputDir);
        var voices = TtsProviderStreaming.ListVoices().ToArray();
        int voicesCount = voices.Length;

        for (int i = 0; i < numClips; i++)
        {
            string voiceKey = voices[i % voicesCount];
            string command = PositiveCommands[i % PositiveCommands.Length];

            // Full utterance: Wake + random pause + command
            using var wakeStream = await TtsProviderStreaming.TextToSpeechAsync(wakeWord, voiceKey, true);
            byte[] wakeWav = wakeStream.ToArray();
            byte[] wakePcm = Algos.ConvertWavToPcm(wakeWav, 16000);

            using var cmdStream = await TtsProviderStreaming.TextToSpeechAsync(command, voiceKey, true);
            byte[] cmdWav = cmdStream.ToArray();
            byte[] cmdPcm = Algos.ConvertWavToPcm(cmdWav, 16000);

            // Random pause: 200-1000ms silence
            int pauseMs = 200 + (i % 4 * 200);  // 200, 400, 600, 800
            byte[] pause = new byte[(16000 * pauseMs / 1000 * 2)];

            // Random wake offset: 0-1500ms (for mid-embedding)
            int offsetMs = (i / 5 % 4) * 500;  // 0, 500, 1000, 1500
            int offsetBytes = (16000 * offsetMs / 1000 * 2);

            // Total <7s: Wake + offset silence + pause + command
            int totalBytes = Math.Min(offsetBytes + wakePcm.Length + pause.Length + cmdPcm.Length, 16000 * 6 * 2);  // 6s cap
            byte[] clipPcm = new byte[totalBytes];

            // Place components
            int pos = offsetBytes;
            Array.Copy(wakePcm, 0, clipPcm, pos, Math.Min(wakePcm.Length, totalBytes - pos)); pos += wakePcm.Length;
            Array.Copy(pause, 0, clipPcm, pos, Math.Min(pause.Length, totalBytes - pos)); pos += pause.Length;
            Array.Copy(cmdPcm, 0, clipPcm, pos, Math.Min(cmdPcm.Length, totalBytes - pos));

            // Augment if enabled
            if (augment && i % 3 != 0)  // 2/3 augmented
            {
                clipPcm = AugmentAudio(clipPcm, noiseDir, speedFactor: 0.8 + (new Random(i).NextDouble() * 0.8), pitchShift: -2 + (new Random(i + 1).Next(5)), noiseLevel: 0.1);
            }

            // Save WAV
            byte[] wav = Algos.CreateWavFromPcm(clipPcm, 16000);
            string filename = $"positive_long_{i:D5}.wav";
            await File.WriteAllBytesAsync(Path.Combine(outputDir, filename), wav);

            if (i % 500 == 0) Log.Information("Generated {i}/{numClips} long positives", i, numClips);
        }
    }

    public static async Task GenerateNegativeClipsAsync(string outputDir, int numPerSentence = 250, string? noiseDir = null, bool augment = true)
    {
        Directory.CreateDirectory(outputDir);
        var voices = TtsProviderStreaming.ListVoices().ToArray();
        int voicesCount = voices.Length;

        for (int sentIdx = 0; sentIdx < NegativeSentences.Length; sentIdx++)
        {
            string sentence = NegativeSentences[sentIdx];
            for (int i = 0; i < numPerSentence; i++)
            {
                string voiceKey = voices[(sentIdx * numPerSentence + i) % voicesCount];

                using var stream = await TtsProviderStreaming.TextToSpeechAsync(sentence, voiceKey, true);
                byte[] wav = stream.ToArray();
                byte[] pcm = Algos.ConvertWavToPcm(wav, 16000);

                // Random length trim (2-6s)
                int trimEndMs = 2000 + (new Random(i).Next(4000));
                int trimEndBytes = Math.Min((16000 * trimEndMs / 1000 * 2), pcm.Length);
                byte[] trimmedPcm = new byte[trimEndBytes];
                Array.Copy(pcm, 0, trimmedPcm, 0, trimEndBytes);

                if (augment && i % 3 != 0)
                {
                    trimmedPcm = AugmentAudio(trimmedPcm, noiseDir, speedFactor: 0.9 + (new Random(i).NextDouble() * 0.4), pitchShift: -1 + (new Random(i + 1).Next(3)), noiseLevel: 0.15);
                }

                byte[] outWav = Algos.CreateWavFromPcm(trimmedPcm, 16000);
                string filename = $"negative_long_{sentIdx:D2}_{i:D3}.wav";
                await File.WriteAllBytesAsync(Path.Combine(outputDir, filename), outWav);
            }
        }
    }

    public static byte[] AugmentAudio(byte[] pcm16k, string? noiseDir, double speedFactor = 1.0, int pitchShift = 0, double noiseLevel = 0.1)
    {
        if (pcm16k.Length == 0) return pcm16k;

        try
        {
            // Step 1: Load PCM as float samples (your fixed TrainingAlgos.ConvertPcmToFloat)
            float[] inputSamples = TrainingAlgos.ConvertPcmToFloat(pcm16k.AsSpan());

            // Step 2: Apply speed/pitch with SoundTouchProcessor
            var soundTouch = new SoundTouch.SoundTouchProcessor();
            soundTouch.SampleRate = 16000;
            soundTouch.Channels = 1;  // Mono
            soundTouch.PutSamples(inputSamples, inputSamples.Length);

            // Set parameters
            soundTouch.Tempo = (float)speedFactor;  // e.g., 0.9 for slower
            soundTouch.Pitch = (float)Math.Pow(2, pitchShift / 12.0);  // Semitones

            // Flush and read processed samples (use List for dynamic sizing)
            soundTouch.Flush();
            var outputSamplesList = new List<float>(inputSamples.Length * 2);  // Initial capacity with headroom
            const int chunkSize = 1024;
            float[] chunk = new float[chunkSize];
            while (true)
            {
                int samplesReceived = soundTouch.ReceiveSamples(chunk, chunkSize);
                if (samplesReceived == 0) break;
                outputSamplesList.AddRange(chunk.Take(samplesReceived));
            }
            float[] outputSamples = outputSamplesList.ToArray();

            // Step 3: Convert back to PCM bytes (your fixed TrainingAlgos.ConvertFloatToPcm)
            byte[] augmentedPcm = TrainingAlgos.ConvertFloatToPcm(outputSamples.AsSpan());

            // Step 4: Mix noise if requested
            if (noiseLevel > 0 && !string.IsNullOrEmpty(noiseDir) && augmentedPcm.Length > 0)
            {
                var noiseFiles = Directory.GetFiles(noiseDir, "*.wav").Where(f => new FileInfo(f).Length < 1000000).ToArray();
                if (noiseFiles.Length > 0)
                {
                    byte[] noisePcm = Algos.ConvertWavToPcm(File.ReadAllBytes(noiseFiles[new Random().Next(noiseFiles.Length)]), 16000);
                    if (noisePcm.Length > 0)
                    {
                        float[] augFloat = TrainingAlgos.ConvertPcmToFloat(augmentedPcm.AsSpan());
                        float[] noiseFloat = TrainingAlgos.ConvertPcmToFloat(noisePcm.AsSpan());
                        for (int j = 0; j < augFloat.Length; j++)
                        {
                            augFloat[j] += (float)(noiseFloat[j % noiseFloat.Length] * noiseLevel);
                        }
                        augmentedPcm = TrainingAlgos.ConvertFloatToPcm(augFloat.AsSpan());  // Reuse your fixed version
                    }
                }
            }

            return augmentedPcm;
        }
        catch (Exception ex)
        {
            Log.Error(ex, "Audio augmentation failed; returning original");
            return pcm16k;  // Fallback
        }
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

    // Standalone test method
    //public static void TestModel(string modelPath, string featuresDir, string[] testWavPaths, string[] expectedLabels)
    //{
    //    var mlContext = new MLContext();
    //    ITransformer model = mlContext.Model.Load(modelPath, out _);

    //    int correct = 0;
    //    for (int i = 0; i < testWavPaths.Length; i++)
    //    {
    //        // Generate MFCC PNG for this WAV (reuse your ComputeMfcc + CreateImageFromMfcc + save)
    //        string tempPng = Path.Combine(Path.GetTempPath(), $"test_{i}.png");
    //        var mfcc = ComputeMfcc(testWavPaths[i], 13);
    //        var padded = PadOrTrim(mfcc, 100);
    //        var image = CreateImageFromMfcc(padded);
    //        image.SaveAsPng(tempPng);

    //        // Predict (as in your existing Predict method)
    //        string relativePath = $"dummy_dir/{Path.GetFileName(tempPng)}"; // Hack: Use a temp subdir
    //                                                                        // ... (copy your prediction logic)
    //        var pred = /* extracted PredictedLabel */;

    //        Log.Information($"Test {i}: {testWavPaths[i]} -> Predicted: {pred} (Expected: {expectedLabels[i]})");
    //        if (pred == expectedLabels[i]) correct++;

    //        File.Delete(tempPng); // Cleanup
    //    }
    //    Log.Information($"Overall: {correct}/{testWavPaths.Length} correct ({correct * 100.0 / testWavPaths.Length:F1}%)");
    //}

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

    public async static Task Main(string[] args)
    {
        TrainingAlgos.AddConsoleLogger();
        Log.Information($"{Environment.CurrentDirectory}");

        if (args.Length < 3)
        {
            Serilog.Log.Error("Usage: WakeWordTrainingDataGenerator <wakeWord> <positiveDir> <negativeDir> <noiseDir> [numPositive=5000] [numNegativePerWord=250]");
            return;
        }

        // Training may take a couple hours, can run with 'Alina goodaudio badaudio'
        await RunTraining(args);
    }

    private static async Task RunTraining(string[] args)
    {
        string wakeWord = args[0];
        string positiveDir = args[1];
        string negativeDir = args[2];
        string noiseDir = args[3];
        int numPositive = args.Length >= 5 ? int.Parse(args[4]) : 5000;
        int numNegativePerWord = args.Length >= 6 ? int.Parse(args[5]) : 250;

        Log.Information("Loaded with args: {wakeWord} {positiveDir} {negativeDir} {noiseDir} numPositive={numPositive} numNegativePerWord={numNegativePerWord}",
            wakeWord, positiveDir, negativeDir, noiseDir, numPositive, numNegativePerWord);

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