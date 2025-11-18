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
                path: "training_log.txt",
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


public static class Program
{

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

    private static async Task TestModelOnFiles(
        WakeWordDetector detector,
        string[] filePaths,
        bool expectedPositive,
        string testSetName = "Custom Set")
    {
        if (filePaths == null || filePaths.Length == 0)
        {
            Log.Warning("No files provided for testing in set '{testSetName}'.", testSetName);
            return;
        }

        int errorCount = 0;
        int total = filePaths.Length;

        Log.Information("Testing {total} files from '{testSetName}' (expected: {expect})",
            total, testSetName, expectedPositive ? "POSITIVE" : "NEGATIVE");

        foreach (var file in filePaths)
        {
            byte[] wavBytes = await File.ReadAllBytesAsync(file);
            byte[] pcm = Algos.ConvertWavToPcm(wavBytes, 16000);

            var (isWake, confidence) = detector.IsWakeWord(pcm);

            bool correct = isWake == expectedPositive;
            if (!correct) errorCount++;

            string result = correct ? "Correct" : (expectedPositive ? "FN" : "FP");

            Log.Information("Test {file}: {result} -> {detected} (conf: {conf:F3})",
                Path.GetFileName(file), result, isWake ? "POSITIVE" : "NEGATIVE", confidence);
        }

        float errorRate = (float)errorCount / total;
        Log.Information("=== {testSetName} Summary: {errorCount}/{total} errors ({errorRate:P1}) ===",
            testSetName, errorCount, total, errorRate);
    }

    private static async Task RunTraining(string[] args)
    {
        await TtsProviderStreaming.InitializeAsync(new());

        string wakeWord = args[0];
        string positiveDir = args[1];
        string negativeDir = args[2];
        string noiseDir = args[3];
        int numPositive = args.Length >= 5 ? int.Parse(args[4]) : 5000;
        int numNegativePerWord = args.Length >= 6 ? int.Parse(args[5]) : 400;

        Log.Information("Loaded with args: {wakeWord} {positiveDir} {negativeDir} {noiseDir} numPositive={numPositive} numNegativePerWord={numNegativePerWord}",
            wakeWord, positiveDir, negativeDir, noiseDir, numPositive, numNegativePerWord);

        // Generate positives
        //await WakeWordDataGenerator.GeneratePositiveClipsAsync(wakeWord, positiveDir, numPositive, noiseDir);

        //// Generate negatives
        //await WakeWordDataGenerator.GenerateNegativeClipsAsync(negativeDir, numNegativePerWord, noiseDir);

        //// Generate MFCC features as images
        //string featuresDir = Path.Combine(Path.GetDirectoryName(positiveDir) ?? ".", "features");
        //await WakeWordDataGenerator.GenerateFeaturesAsync(positiveDir, negativeDir, featuresDir);

        //// Train and save model
        //WakeWordDataGenerator.TrainModel(featuresDir);

        // New: Test on 30 random badaudio files (expected negatives)
        const int numToTest = 30;
        const string modelPath = "wakeword_model.zip";
        Log.Information("Model path: {modelPath}", modelPath);
        using var detector = new WakeWordDetector(modelPath);
        Log.Information("Created WakeWordDetector.");
        
        var allBadaudioFiles = Directory.GetFiles(negativeDir, "*.wav");
        var allGoodAudioFiles = Directory.GetFiles(positiveDir, "*.wav");

        if (allBadaudioFiles.Length == 0)
        {
            Log.Error("No badaudio files for testing.");
            return;
        }
        if(allGoodAudioFiles.Length == 0)
        {
            Log.Error("No goodaudio files for testing.");
            return;
        }

        var randomNegativeFiles = allBadaudioFiles.OrderBy(_ => Guid.NewGuid()).Take(numToTest).ToArray();
        var randomPositiveFiles = allGoodAudioFiles.OrderBy(_ => Guid.NewGuid()).Take(numToTest).ToArray();

        //int fpCount = 0;
        Log.Information("Testing on {numToTest} random badaudio files (expected: all negative)...", numToTest);
        await TestModelOnFiles(detector, randomNegativeFiles, false, "badaudio");
        Log.Information("Testing on {numToTest} random goodaudio files (expected: all positive)...", numToTest);
        await TestModelOnFiles(detector, randomPositiveFiles, true, "goodaudio");

    }
}