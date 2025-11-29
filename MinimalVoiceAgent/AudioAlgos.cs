using NAudio.Codecs;
using NAudio.Wave;
using Serilog;
using System.Runtime.InteropServices;

namespace MinimalVoiceAgent;
public static class AudioAlgos
{
    /// <summary>
    /// Resamples PCM audio to the target sample rate, handling variable input sizes.
    /// </summary>
    public static byte[] ResamplePcmWithNAudio(byte[] inputPcm, int inputSampleRate, int outputSampleRate)
    {
        // Handle odd-length input by padding
        if (inputPcm.Length % 2 != 0)
        {
            Log.Warning($"Invalid input PCM length: {inputPcm.Length} bytes (must be even for 16-bit), padding with 1 byte");
            byte[] paddedInput = new byte[inputPcm.Length + 1];
            Array.Copy(inputPcm, paddedInput, inputPcm.Length);
            inputPcm = paddedInput;
        }

        // Calculate expected output size
        int inputSamples = inputPcm.Length / 2; // 16-bit mono
        double sampleRateRatio = (double)outputSampleRate / inputSampleRate;
        int expectedOutputSamples = (int)Math.Ceiling(inputSamples * sampleRateRatio);
        // Ensure even output samples for 16-bit audio
        if (expectedOutputSamples % 2 != 0)
            expectedOutputSamples++;

        using var inputStream = new MemoryStream(inputPcm);
        using var rawSource = new RawSourceWaveStream(inputStream, new WaveFormat(inputSampleRate, 16, 1));
        var outFormat = new WaveFormat(outputSampleRate, 16, 1);

        using var conversionStream = new WaveFormatConversionStream(outFormat, rawSource);
        using var outputStream = new MemoryStream();
        byte[] buffer = new byte[4096];
        int bytesRead;

        while ((bytesRead = conversionStream.Read(buffer, 0, buffer.Length)) > 0)
        {
            outputStream.Write(buffer, 0, bytesRead);
        }

        byte[] resampled = outputStream.ToArray();

        // Ensure output is even-length and padded to expected size
        if (resampled.Length % 2 != 0)
        {
            byte[] paddedOutput = new byte[resampled.Length + 1];
            Array.Copy(resampled, paddedOutput, resampled.Length);
            resampled = paddedOutput;
        }

        if (resampled.Length < expectedOutputSamples * 2)
        {
            byte[] paddedOutput = new byte[expectedOutputSamples * 2];
            Array.Copy(resampled, paddedOutput, resampled.Length);
            resampled = paddedOutput;
        }
        else if (resampled.Length > expectedOutputSamples * 2)
        {
            byte[] trimmedOutput = new byte[expectedOutputSamples * 2];
            Array.Copy(resampled, trimmedOutput, trimmedOutput.Length);
            resampled = trimmedOutput;
        }

        return resampled;
    }

    /// <summary>
    /// Converts PCMU (G.711 μ-law) encoded audio to 16-bit 16kHz mono PCM.
    /// </summary>
    /// <param name="pcmuAudio">PCMU-encoded byte array</param>
    /// <returns>16-bit 16kHz mono PCM byte array</returns>
    public static byte[] ConvertPcmuToPcm16kHz(byte[] pcmuAudio)
    {
        try
        {
            // Step 1: Decode PCMU to 16-bit PCM at 8kHz
            int sampleCount = pcmuAudio.Length;
            byte[] pcm8kHz = new byte[sampleCount * 2]; // 16-bit = 2 bytes per sample

            for (int i = 0; i < sampleCount; i++)
            {
                // Decode mu-law sample to 16-bit linear PCM
                short linearSample = MuLawDecoder.MuLawToLinearSample(pcmuAudio[i]);
                // Write as little-endian 16-bit sample
                pcm8kHz[i * 2] = (byte)(linearSample & 0xFF);
                pcm8kHz[i * 2 + 1] = (byte)(linearSample >> 8);
            }

            // Step 2: Resample from 8kHz to 16kHz
            byte[] pcm16kHz = ResamplePcmWithNAudio(pcm8kHz, 8000, 16000);
            return pcm16kHz;
        }
        catch (Exception ex)
        {
            Log.Error(ex, "Failed to convert PCMU to 16-bit 16kHz PCM.");
            return Array.Empty<byte>();
        }
    }

    /// <summary>
    /// Converts 16-bit PCM audio (e.g., at 16kHz) to PCMU (G.711 μ-law) encoded audio at the specified output rate (default 8kHz for telephony).
    /// </summary>
    /// <param name="pcmAudio">16-bit mono PCM byte array (e.g., 16kHz)</param>
    /// <param name="inputSampleRate">Input sample rate (e.g., 16000)</param>
    /// <param name="outputSampleRate">Output sample rate for PCMU (default 8000)</param>
    /// <returns>PCMU-encoded byte array at outputSampleRate</returns>
    public static byte[] ConvertPcmToPcmu(byte[] pcmAudio, int inputSampleRate, int outputSampleRate = 8000)
    {
        try
        {
            // Step 1: Ensure even length
            if (pcmAudio.Length % 2 != 0)
            {
                Log.Warning($"Invalid input PCM length: {pcmAudio.Length} bytes (must be even for 16-bit), padding with 1 byte");
                byte[] paddedInput = new byte[pcmAudio.Length + 1];
                Array.Copy(pcmAudio, paddedInput, pcmAudio.Length);
                pcmAudio = paddedInput;
            }

            // Step 2: Resample to outputSampleRate if needed
            byte[] resampledPcm;
            if (inputSampleRate == outputSampleRate)
            {
                resampledPcm = pcmAudio;
            }
            else
            {
                resampledPcm = ResamplePcmWithNAudio(pcmAudio, inputSampleRate, outputSampleRate);
            }

            // Step 3: Encode 16-bit PCM samples to μ-law bytes
            int sampleCount = resampledPcm.Length / 2; // 16-bit samples
            byte[] pcmuOutput = new byte[sampleCount]; // 1 byte per μ-law sample

            for (int i = 0; i < sampleCount; i++)
            {
                // Read little-endian 16-bit sample
                short linearSample = (short)((resampledPcm[i * 2 + 1] << 8) | resampledPcm[i * 2]);
                // Encode to μ-law
                pcmuOutput[i] = MuLawEncoder.LinearToMuLawSample(linearSample);
            }

            return pcmuOutput;
        }
        catch (Exception ex)
        {
            Log.Error(ex, "Failed to convert PCM to PCMU.");
            return Array.Empty<byte>();
        }
    }

    public static byte[] AdjustPcmVolume(byte[] pcmFrame, float factor)
    {
        var samples = MemoryMarshal.Cast<byte, short>(pcmFrame.AsSpan());
        for (int i = 0; i < samples.Length; i++)
        {
            samples[i] = (short)Math.Clamp(samples[i] * factor, short.MinValue, short.MaxValue);
        }
        return pcmFrame;
    }

    /// <summary>
    /// Generates mu-law silence audio for the specified duration in seconds.
    /// </summary>
    public static byte[] GeneratePcmuSilence(int durationSeconds, int sampleRate = 8000)
    {
        const byte silenceSample = 0xFF; // Mu-law silence
        int byteCount = sampleRate * durationSeconds;
        byte[] silence = new byte[byteCount];
        Array.Fill(silence, silenceSample);
        return silence;
    }

    /// <summary>
    /// Generates PCM 16kHz silence audio for the specified duration in seconds.
    /// </summary>
    public static byte[] GeneratePcm16kHzSilence(int durationSeconds, int sampleRate = 16000)
    {
        const int bytesPerSample = 2; // 16-bit
        int byteCount = sampleRate * durationSeconds * bytesPerSample;
        byte[] silence = new byte[byteCount]; // Zero-initialized for silence
        return silence;
    }
}