using Accord.Audio;
using Microsoft.ML;
using Microsoft.ML.Vision;
using NAudio.Wave;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Serilog;

namespace WakeWordTrainingDataGenerator;

public class ImageData
{
    public string? ImagePath { get; set; }
    public string? Label { get; set; }
}

public static class WakeWordDataGenerator
{
    private const double NoiseLevel = 0.05;

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
        "Dive into the ocean", "Fly kites on windy days", "Sail boats across lakes",
        "good morning everyone", "turn it up a bit", "what's going on here", "call the office now",
        "a linear path ahead", "Elena is calling", "align the picture", "I'll lean on this",
        "olive and lemon", "a lion in the room", "helen a favor", "ale in the barrel",
        "all in a line", "I'll need a loan", "open the window slightly", "read the latest news",
        "what's that", "hold on", "oh well", "someone do something",
        "a linear path", "align it now", "ale and lager", "a leaner cut",
        "all in a row", "I'll need a lift", "olive and feta", "a lion's roar", "helen a hand",
        "open a window", "I'll lean in", "a lemon aid", "a lien on the house",
        "align the stars", "ale ina bottle", "a leaner machine", "helen a favor", "all ena time",
        "I'll need a loan", "open the line", "a line in the sand", "align the team"
    };

    // New TTS-only generation function: Handles all synthesis in batches, returns metadata + PCMs for later assembly
    public static async Task<List<(byte[] wakePcm, byte[] cmdPcm, string voiceKey, string command, int index)>> GenerateTtsClipsAsync(
        string wakeWord,
        int numClips = 5000,
        int batchSize = 4)
    {
        var voices = TtsProviderStreaming.ListVoices().ToArray();
        int voicesCount = voices.Length;
        var ttsResults = new List<(byte[] wakePcm, byte[] cmdPcm, string voiceKey, string command, int index)>();

        int startIndex = 0;
        while (startIndex < numClips)
        {
            int currentBatchSize = Math.Min(batchSize, numClips - startIndex);
            var batchTasks = new List<Task<(byte[] wakePcm, byte[] cmdPcm)>>();

            for (int j = 0; j < currentBatchSize; j++)
            {
                int localI = startIndex + j;
                string voiceKey = voices[localI % voicesCount];
                string command = PositiveCommands[localI % PositiveCommands.Length];

                var task = Task.Run(async () =>
                {
                    // Synthesize wake and command
                    using var wakeStream = await TtsProviderStreaming.TextToSpeechAsync(wakeWord, voiceKey, true);
                    byte[] wakeWav = wakeStream.ToArray();
                    byte[] wakePcm = Algos.ConvertWavToPcm(wakeWav, 16000);

                    using var cmdStream = await TtsProviderStreaming.TextToSpeechAsync(command, voiceKey, true);
                    byte[] cmdWav = cmdStream.ToArray();
                    byte[] cmdPcm = Algos.ConvertWavToPcm(cmdWav, 16000);

                    return (wakePcm, cmdPcm);
                });

                batchTasks.Add(task);
            }

            // Wait for batch TTS to complete
            var batchPcms = await Task.WhenAll(batchTasks);

            // Collect with metadata
            for (int j = 0; j < currentBatchSize; j++)
            {
                int i = startIndex + j;
                var (wakePcm, cmdPcm) = batchPcms[j];
                string voiceKey = voices[i % voicesCount];
                string command = PositiveCommands[i % PositiveCommands.Length];

                ttsResults.Add((wakePcm, cmdPcm, voiceKey, command, i));
            }

            startIndex += currentBatchSize;

            if (startIndex % (batchSize * 10) == 0)
                Log.Information("Generated TTS for {startIndex}/{numClips} clips", startIndex, numClips);
        }

        Log.Information("Completed TTS generation for {numClips} clips", numClips);
        return ttsResults;
    }

    // New augmentation/assembly function: Takes TTS results, assembles full clips, augments, returns WAV bytes list
    public static async Task<List<byte[]>> AugmentAndAssembleClipsAsync(
        List<(byte[] wakePcm, byte[] cmdPcm, string voiceKey, string command, int index)> ttsResults,
        string? noiseDir = null,
        bool augment = true,
        CancellationToken ct = default)
    {
        var finalWavs = new List<byte[]>();

        foreach (var (wakePcm, cmdPcm, voiceKey, command, i) in ttsResults)
        {
            ct.ThrowIfCancellationRequested();

            // Assemble full clip (wake + offset + pause + command)
            int pauseMs = 200 + (i % 4 * 200);  // 200, 400, 600, 800
            byte[] pause = new byte[(16000 * pauseMs / 1000 * 2)];

            int offsetMs = (i / 5 % 4) * 500;  // 0, 500, 1000, 1500
            int offsetBytes = (16000 * offsetMs / 1000 * 2);

            int totalBytes = Math.Min(offsetBytes + wakePcm.Length + pause.Length + cmdPcm.Length, 16000 * 6 * 2);  // 6s cap
            byte[] clipPcm = new byte[totalBytes];

            int pos = offsetBytes;
            Array.Copy(wakePcm, 0, clipPcm, pos, Math.Min(wakePcm.Length, totalBytes - pos)); pos += wakePcm.Length;
            Array.Copy(pause, 0, clipPcm, pos, Math.Min(pause.Length, totalBytes - pos)); pos += pause.Length;
            Array.Copy(cmdPcm, 0, clipPcm, pos, Math.Min(cmdPcm.Length, totalBytes - pos));

            // Augment if enabled
            if (augment && i % 2 != 0)  // 1/2 augmented
            {
                double speedFactor = 0.8 + (new Random(i).NextDouble() * 0.8);
                int pitchShift = -2 + (new Random(i + 1).Next(5));
                clipPcm = AugmentAudio(clipPcm, noiseDir, speedFactor, pitchShift, NoiseLevel);
            }

            // Convert to WAV
            byte[] wav = await Algos.CreateWavFromPcmAsync(clipPcm, 16000);
            finalWavs.Add(wav);

            if (finalWavs.Count % 500 == 0)
                Log.Information("Assembled and augmented {count}/{total} clips", finalWavs.Count, ttsResults.Count);
        }

        Log.Information("Completed augmentation and assembly for {total} clips", ttsResults.Count);
        return finalWavs;
    }


    // Updated GeneratePositiveClipsAsync in WakeWordDataGenerator.cs: Mix 70% full, 30% wake-only
    public static async Task GeneratePositiveClipsAsync(string wakeWord, string outputDir, int numClips = 5000, string? noiseDir = null, bool augment = true)
    {
        Directory.CreateDirectory(outputDir);
        var voices = TtsProviderStreaming.ListVoices().ToArray();
        int voicesCount = voices.Length;

        int fullClips = (int)(numClips * 0.7);  // 70% full utterances
        int wakeOnlyClips = numClips - fullClips;  // 30% wake-only

        // Gen full utterances
        await GenerateFullUtterancesAsync(wakeWord, outputDir, fullClips, voices, PositiveCommands, noiseDir, augment, false);  // False for full

        // Gen wake-only
        await GenerateWakeOnlyAsync(wakeWord, outputDir, wakeOnlyClips, voices, noiseDir, augment);  // No isWakeOnly (uses default false, but logic differs)

        Log.Information("Generated {full} full + {wake} wake-only = {total} positives", fullClips, wakeOnlyClips, numClips);
    }

    private static async Task GenerateFullUtterancesAsync(string wakeWord, string outputDir, int numClips, string[] voices, string[] commands, string? noiseDir, bool augment, bool isWakeOnly = false)
    {
        var clipPool = new List<(string path, byte[] data)>();  // Pool for batched writes

        int startIndex = 0;
        while (startIndex < numClips)
        {
            int currentBatchSize = Math.Min(4, numClips - startIndex);  // Batch size 4 for TTS overlap
            var batchTasks = new List<Task<(byte[] wakePcm, byte[] cmdPcm)>>();

            for (int j = 0; j < currentBatchSize; j++)
            {
                int localI = startIndex + j;
                string voiceKey = voices[localI % voices.Length];
                string command = commands[localI % commands.Length];

                var task = Task.Run(async () =>
                {
                    using var wakeStream = await TtsProviderStreaming.TextToSpeechAsync(wakeWord, voiceKey, true);
                    byte[] wakeWav = wakeStream.ToArray();
                    byte[] wakePcm = Algos.ConvertWavToPcm(wakeWav, 16000);

                    using var cmdStream = await TtsProviderStreaming.TextToSpeechAsync(command, voiceKey, true);
                    byte[] cmdWav = cmdStream.ToArray();
                    byte[] cmdPcm = Algos.ConvertWavToPcm(cmdWav, 16000);

                    return (wakePcm, cmdPcm);
                });

                batchTasks.Add(task);
            }

            // Wait for batch TTS to complete
            var batchResults = await Task.WhenAll(batchTasks);

            // Assemble and augment for this batch
            for (int j = 0; j < currentBatchSize; j++)
            {
                int i = startIndex + j;
                var (wakePcm, cmdPcm) = batchResults[j];

                // Random pause: 200-800ms silence
                int pauseMs = 200 + (i % 4 * 200);  // 200, 400, 600, 800
                byte[] pause = new byte[(16000 * pauseMs / 1000 * 2)];

                // Random wake offset: 0-1500ms (for mid-embedding)
                int offsetMs = (i / 5 % 4) * 500;  // 0, 500, 1000, 1500
                int offsetBytes = (16000 * offsetMs / 1000 * 2);

                // Total <7s: Offset + wake + pause + command
                int totalBytes = Math.Min(offsetBytes + wakePcm.Length + pause.Length + cmdPcm.Length, 16000 * 6 * 2);  // 6s cap
                byte[] clipPcm = new byte[totalBytes];

                // Place components
                int pos = offsetBytes;
                int copied = Math.Min(wakePcm.Length, totalBytes - pos);
                Array.Copy(wakePcm, 0, clipPcm, pos, copied); pos += copied;
                copied = Math.Min(pause.Length, totalBytes - pos);
                Array.Copy(pause, 0, clipPcm, pos, copied); pos += copied;
                copied = Math.Min(cmdPcm.Length, totalBytes - pos);
                Array.Copy(cmdPcm, 0, clipPcm, pos, copied);  // pos += not needed, end

                // Augment if enabled (2/3 for full)
                if (augment && i % 3 != 0)
                {
                    var rng = new Random(i);  // Seeded for repro
                    double speedFactor = 0.8 + (rng.NextDouble() * 0.8);  // 0.8-1.6x
                    int pitchShift = -2 + (rng.Next(5));  // -2 to +2 semitones
                    double noiseLevel = NoiseLevel;
                    clipPcm = AugmentAudio(clipPcm, noiseDir, speedFactor, pitchShift, noiseLevel);
                }

                // Generate WAV and pool
                byte[] wav = Algos.CreateWavFromPcm(clipPcm, 16000);
                string prefix = isWakeOnly ? "wake_only" : "full";
                string filename = $"positive_{prefix}_{i:D5}.wav";
                string fullPath = Path.Combine(outputDir, filename);
                clipPool.Add((fullPath, wav));

                if (i % 500 == 0) Log.Information("Assembled {i}/{numClips} {type} utterances (pooled)", 
                    i,
                    numClips,
                    isWakeOnly ? "wake-only" : "full");
            }

            startIndex += currentBatchSize;
        }

        // Batch async writes
        var writeTasks = clipPool.Select(clip => File.WriteAllBytesAsync(clip.path, clip.data)).ToArray();
        await Task.WhenAll(writeTasks);

        Log.Information("Batched and saved {numClips} {type} utterances", numClips, isWakeOnly ? "wake-only" : "full");
    }

    // New helper for wake-only
    private static async Task GenerateWakeOnlyAsync(string wakeWord, string outputDir, int numClips, string[] voices, string? noiseDir, bool augment)
    {
        for (int i = 0; i < numClips; i++)
        {
            string voiceKey = voices[i % voices.Length];
            using var wakeStream = await TtsProviderStreaming.TextToSpeechAsync(wakeWord, voiceKey, true);
            byte[] wakeWav = wakeStream.ToArray();
            byte[] wakePcm = Algos.ConvertWavToPcm(wakeWav, 16000);

            // Target 1s clip (wake ~0.5s + pad)
            int targetBytes = 16000 * 1 * 2;
            byte[] clipPcm = new byte[Math.Max(targetBytes, wakePcm.Length)];
            Array.Copy(wakePcm, 0, clipPcm, 0, Math.Min(wakePcm.Length, targetBytes));

            // Augment heavily for robustness
            if (augment)
            {
                clipPcm = AugmentAudio(clipPcm, noiseDir, speedFactor: 0.9 + (new Random(i).NextDouble() * 0.2), pitchShift: -1 + new Random(i + 1).Next(3), noiseLevel: 0.05);
            }

            byte[] wav = Algos.CreateWavFromPcm(clipPcm, 16000);
            string filename = $"positive_wake_only_{i:D5}.wav";
            await File.WriteAllBytesAsync(Path.Combine(outputDir, filename), wav);
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
                    trimmedPcm = AugmentAudio(trimmedPcm, noiseDir, speedFactor: 0.9 + (new Random(i).NextDouble() * 0.4), pitchShift: -1 + (new Random(i + 1).Next(3)), noiseLevel: NoiseLevel);
                }

                byte[] outWav = Algos.CreateWavFromPcm(trimmedPcm, 16000);
                string filename = $"negative_long_{sentIdx:D2}_{i:D3}.wav";
                await File.WriteAllBytesAsync(Path.Combine(outputDir, filename), outWav);
            }
        }
    }

    public static byte[] AugmentAudio(byte[] pcm16k, string? noiseDir, double speedFactor = 1.0, int pitchShift = 0, double noiseLevel = 0.05)
    {
        if (pcm16k.Length == 0) return pcm16k;
        try
        {
            // Step 1: Load PCM as float samples (your fixed TrainingAlgos.ConvertPcmToFloat)
            float[] inputSamples = TrainingAlgos.ConvertPcmToFloat(pcm16k.AsSpan());
            // Step 2: Apply speed/pitch with SoundTouchProcessor
            var soundTouch = new SoundTouch.SoundTouchProcessor();
            soundTouch.SampleRate = 16000;
            soundTouch.Channels = 1; // Mono
            soundTouch.PutSamples(inputSamples, inputSamples.Length);
            // Set parameters
            soundTouch.Tempo = (float)speedFactor; // e.g., 0.9 for slower
            soundTouch.Pitch = (float)Math.Pow(2, pitchShift / 12.0); // Semitones
                                                                      // Flush and read processed samples (use List for dynamic sizing)
            soundTouch.Flush();
            var outputSamplesList = new List<float>(inputSamples.Length * 2); // Initial capacity with headroom
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
                var noiseFiles = Directory.GetFiles(noiseDir, "*.wav", SearchOption.AllDirectories)
                    .Where(f => new FileInfo(f).Length < 1000000).ToArray();
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
                        augmentedPcm = TrainingAlgos.ConvertFloatToPcm(augFloat.AsSpan()); // Reuse your fixed version
                    }
                }
            }
            return augmentedPcm;
        }
        catch (Exception ex)
        {
            Log.Error(ex, "Audio augmentation failed; returning original");
            return pcm16k; // Fallback
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
            string relativeSubdir = label; // e.g., "positive" or "negative"
            foreach (var file in Directory.GetFiles(dir, "*.png"))
            {
                string relativePath = Path.Combine(relativeSubdir, Path.GetFileName(file)); // e.g., "negative\alena_114.png"
                images.Add(new ImageData { ImagePath = relativePath, Label = label });
            }
        }
        if (!images.Any())
        {
            Serilog.Log.Error($"No images found in {featuresDir}. Run data generation first.");
            return;
        }

        // Shuffle and split data for validation (80/20)
        var dataArray = images.ToArray();
        var random = new Random(42); // Reproducible shuffle
        random.Shuffle(dataArray); // Accord.Math extension

        int trainSplit = (int)(dataArray.Length * 0.8);
        var trainImages = dataArray.Take(trainSplit).ToList();
        var valImages = dataArray.Skip(trainSplit).ToList();

        Serilog.Log.Information("Split: {trainCount} train, {valCount} val images", trainImages.Count, valImages.Count);

        IDataView trainData = mlContext.Data.LoadFromEnumerable(trainImages);
        IDataView valData = mlContext.Data.LoadFromEnumerable(valImages);

        // Define transforms pipeline
        var transforms = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "LabelKey", inputColumnName: "Label")
            .Append(mlContext.Transforms.LoadRawImageBytes(outputColumnName: "Image", imageFolder: featuresDir, inputColumnName: "ImagePath"));

        // Fit transforms on train data to get ITransformer
        var fittedTransforms = transforms.Fit(trainData);

        // Transform both datasets
        IDataView trainDataset = fittedTransforms.Transform(trainData);
        IDataView valDataset = fittedTransforms.Transform(valData);

        var options = new ImageClassificationTrainer.Options
        {
            FeatureColumnName = "Image",
            LabelColumnName = "LabelKey",
            Arch = ImageClassificationTrainer.Architecture.ResnetV250,
            BatchSize = 16,
            Epoch = 50,
            ValidationSet = valDataset, // Enable validation metrics
            MetricsCallback = (metrics) =>
            {
                if (metrics != null)
                    Serilog.Log.Information($"Micro Accuracy: {metrics.Train?.Accuracy ?? 0.0f}");
            },
            WorkspacePath = Path.Combine(featuresDir, "workspace"),
            LearningRate = 0.001f
        };
        var pipeline = mlContext.MulticlassClassification.Trainers.ImageClassification(options)
            .Append(mlContext.Transforms.Conversion.MapKeyToValue(outputColumnName: "PredictedLabel", inputColumnName: "PredictedLabel"));
        ITransformer model = pipeline.Fit(trainDataset);



        // Post-training validation evaluation
        var valPredictions = model.Transform(valDataset);
        var valMetrics = mlContext.MulticlassClassification.Evaluate(valPredictions, labelColumnName: "LabelKey");
        Serilog.Log.Information("Final Validation Metrics - MicroAccuracy: {valAcc:F3}, MacroAccuracy: {macroAcc:F3}, LogLoss: {logLoss:F3}",
            valMetrics.MicroAccuracy, valMetrics.MacroAccuracy, valMetrics.LogLoss);

        // Save ML.NET model
        mlContext.Model.Save(model, trainDataset.Schema, modelPath);
        Serilog.Log.Information($"Model saved to {modelPath}");
    }

}