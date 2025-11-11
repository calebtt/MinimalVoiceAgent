using Serilog;

namespace MinimalVoiceAgent;

public class TtsStreamer : IDisposable
{
    private CancellationTokenSource? _cancellationSource;
    private bool _isPlaying;
    private readonly object _lock = new();

    public event EventHandler<byte[]>? OnAudioChunkReady;

    public bool IsPlaying
    {
        get
        {
            lock (_lock)
            {
                return _isPlaying;
            }
        }
    }

    public async Task StartStreamingAsync(string text, string? voiceKey = null, CancellationToken ct = default)
    {
        lock (_lock)
        {
            if (_isPlaying)
            {
                Log.Warning("TTS already generating, stopping previous.");
                Stop();
            }

            _isPlaying = true;
        }

        if (_cancellationSource != null)
            await _cancellationSource!.CancelAsync();
        _cancellationSource = CancellationTokenSource.CreateLinkedTokenSource(ct);

        try
        {
            // Collect streaming chunks into full PCM buffer
            var chunks = new List<byte[]>();
            await foreach (var pcmChunk in TtsProviderStreaming.TextToSpeechStreamAsync(text, voiceKey, _cancellationSource.Token))
            {
                if (pcmChunk.Length > 0 && !_cancellationSource.Token.IsCancellationRequested)
                {
                    chunks.Add(pcmChunk);
                }
            }

            if (_cancellationSource.Token.IsCancellationRequested)
            {
                Log.Debug("TTS generation cancelled.");
                return;
            }

            if (chunks.Count == 0)
            {
                Log.Warning("No TTS audio generated; skipping.");
                return;
            }

            // Concatenate
            int totalLength = chunks.Sum(c => c.Length);
            byte[] fullPcm = new byte[totalLength];
            int offset = 0;
            foreach (var chunk in chunks)
            {
                Buffer.BlockCopy(chunk, 0, fullPcm, offset, chunk.Length);
                offset += chunk.Length;
            }

            // Send full PCM to pacer
            OnAudioChunkReady?.Invoke(this, fullPcm);

            Log.Information($"Sent full TTS PCM audio ({fullPcm.Length} bytes) for '{text.Substring(0, Math.Min(50, text.Length))}...'");
        }
        catch (OperationCanceledException)
        {
            Log.Debug("TTS generation cancelled.");
        }
        catch (Exception ex)
        {
            Log.Error(ex, "Error during TTS generation.");
        }
        finally
        {
            lock (_lock)
            {
                _isPlaying = false;
            }
        }
    }

    public void Stop()
    {
        lock (_lock)
        {
            if (!_isPlaying) return;

            _cancellationSource?.Cancel();
            _isPlaying = false;
            Log.Information("TTS generation cancelled.");
        }
    }

    public void Dispose()
    {
        Stop();
        _cancellationSource?.Dispose();
        OnAudioChunkReady = null;
    }
}