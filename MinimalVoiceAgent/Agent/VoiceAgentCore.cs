using Serilog;
using MinimalSileroVAD.Core;

namespace MinimalVoiceAgent;

public class VoiceAgentCore : IAsyncDisposable
{
    private const float VolumeLoweringFactor = 0.35f;

    private readonly SttProviderStreaming _streamingSttClient;
    private readonly LlmChat _llmChat;
    private readonly TtsStreamer _ttsStreamer;
    private readonly AudioPacer _audioPacer;

    private CancellationTokenSource? _cancellationTokenSource;
    private IVadSpeechSegmenter? _vad;

    private readonly object _processingLock = new();
    private bool _isProcessingTranscription;
    private bool _doUseInterruption;
    private bool _ignoreCurrentSegment;

    private bool _isDisposed;

    /// <summary>
    /// Buffer attaches to this, used for audio FROM TTS.
    /// </summary>
    public event Action<byte[]>? OnAudioReplyReady;

    public VoiceAgentCore(
        SttProviderStreaming streamingSttClient,
        LlmChat llmChat,
        TtsStreamer ttsStreamer,
        AudioPacer audioPacer,
        bool doUseInterruption = true)
    {
        _streamingSttClient = streamingSttClient ?? throw new ArgumentNullException(nameof(streamingSttClient));
        _llmChat = llmChat ?? throw new ArgumentNullException(nameof(llmChat));
        _ttsStreamer = ttsStreamer ?? throw new ArgumentNullException(nameof(ttsStreamer));
        _audioPacer = audioPacer ?? throw new ArgumentNullException(nameof(audioPacer));
        _ttsStreamer.OnAudioChunkReady += (sender, chunk) => OnAudioReplyReady?.Invoke(chunk!);

        _streamingSttClient.TranscriptionComplete += OnTranscriptionComplete;

        Log.Information("VoiceAgentCore created.");
        _doUseInterruption = doUseInterruption;
    }

    public bool IsProcessingTranscription => _isProcessingTranscription;

    public Task InitializeAsync(IVadSpeechSegmenter vadSegmenter)
    {
        try
        {
            _cancellationTokenSource = new CancellationTokenSource();

            _vad = vadSegmenter ?? throw new ArgumentNullException(nameof(vadSegmenter));

            bool volumeFilterActive = false;

            _vad.SentenceBegin += (sender, e) =>
            {
                Log.Information("VAD: Speech segment started.");

                _ignoreCurrentSegment = false;  // Reset the ignore segment flag at start
                volumeFilterActive = false;

                if (_audioPacer.IsAudioPlaying)
                {
                    if (_doUseInterruption)
                    {
                        Log.Information("VAD: Applying volume filter for interruption.");
                        _audioPacer.ApplyFilter(chunk => AudioAlgos.AdjustPcmVolume(chunk, VolumeLoweringFactor));
                        volumeFilterActive = true;
                    }
                    else
                    {
                        _ignoreCurrentSegment = true;  // Flag to ignore this segment (potential echo)
                        Log.Information("VAD: Flagged segment for ignore (started during playback, interruption disabled).");
                    }
                }
            };

            _vad.SentenceCompleted += (sender, pcmStream) =>
            {
                if (_cancellationTokenSource?.IsCancellationRequested == true)
                    return;

                Log.Information($"VAD: Speech segment completed, {pcmStream.Length} bytes");

                if (volumeFilterActive)
                {
                    _audioPacer.ClearFilter();
                    volumeFilterActive = false;
                    Log.Information("VAD: Volume filter cleared after speech segment.");
                }

                // Ignore if flagged (skips STT processing)
                if (!_doUseInterruption && _ignoreCurrentSegment)
                {
                    _ignoreCurrentSegment = false;
                    Log.Information("VAD: Ignored speech segment that started during playback (interruption disabled).");
                    return;
                }

                _streamingSttClient.ProcessAudioChunkAsync(pcmStream).Wait();
            };

            Log.Information("VoiceAgentCore initialized.");
            return Task.CompletedTask;
        }
        catch (Exception ex)
        {
            Log.Error(ex, "Failed to initialize VoiceAgentCore");
            throw;
        }
    }

    public async Task ShutdownAsync()
    {
        try
        {
            if(_cancellationTokenSource != null)
                await _cancellationTokenSource!.CancelAsync();

            _vad?.Dispose();
            _vad = null;

            _ttsStreamer.Stop();
            _ttsStreamer.Dispose();
            _audioPacer.ResetBuffer();
            Log.Information("VoiceAgentCore shutdown complete.");
            return;
        }
        catch (Exception ex)
        {
            Log.Error(ex, "Error during VoiceAgentCore shutdown");
        }
        return;
    }

    /// <summary>
    /// Audio received for internal processing (from cellphone/microphone, GotAudioRtp forwards audio to this.)
    /// Audio is 16khz 16bit mono PCM. 20ms frames.
    /// </summary>
    public void ProcessIncomingAudioChunk(byte[] pcm16Khz)
    {
        if (_vad == null || _cancellationTokenSource?.IsCancellationRequested == true)
            return;

        _vad.PushFrame(pcm16Khz, 16_000, 20);
    }

    public void InterruptPlayback()
    {
        // If _doUseInterruption is false, this should not be called.
        if (_audioPacer.IsAudioPlaying)
        {
            Log.Information("VoiceAgentCore: Interrupting current TTS playback.");
            _audioPacer.ResetBuffer();
        }
    }

    /// <summary>
    /// Called when complete transcription is ready.
    /// </summary>
    private void OnTranscriptionComplete(object? sender, string transcription)
    {
        if (!_doUseInterruption && _audioPacer.IsAudioPlaying)
        {
            // Ignore new transcription if not using interruption and TTS is playing
            Log.Information("VoiceAgentCore: Ignoring transcription due to ongoing TTS playback (interruption disabled).");
            return;
        }

        Log.Information($"VoiceAgentCore: STT Complete transcription: '{transcription}'");

        bool shouldProcess;
        lock (_processingLock)
        {
            if (_isProcessingTranscription)
            {
                return;
            }
            _isProcessingTranscription = true;
            shouldProcess = true;
        }

        if (!shouldProcess)
            return;

        try
        {
            ProcessCompleteTranscriptionAsync(transcription).Wait();
        }
        finally
        {
            lock (_processingLock)
            {
                _isProcessingTranscription = false;
            }
        }
    }

    /// <summary>
    /// Requests LLM chat completion and starts TTS streaming for the given transcription.
    /// </summary>
    private async Task ProcessCompleteTranscriptionAsync(string transcription)
    {
        try
        {
            Log.Information($"VoiceAgentCore: Processing complete transcription: {transcription}");

            // Updated: Use LlmChat.ProcessMessageAsync (no maxTokens; CT propagated)
            string response = await _llmChat.ProcessMessageAsync(transcription, _cancellationTokenSource?.Token ?? CancellationToken.None);
            Log.Information($"VoiceAgentCore: LLM Response: {response}");

            if (_cancellationTokenSource?.IsCancellationRequested == true)
                return;

            // Interrupt if playing
            if (_doUseInterruption)
            {
                InterruptPlayback();
            }

            // Start streaming TTS (voiceKey null by default)
            await _ttsStreamer.StartStreamingAsync(response, ct: _cancellationTokenSource?.Token ?? CancellationToken.None);
        }
        catch (TaskCanceledException tcex)
        {
            Log.Debug($"VoiceAgentCore: Task canceled, info: {tcex.Message}");
        }
        catch (Exception ex)
        {
            Log.Error(ex, "VoiceAgentCore: Error processing complete transcription.");
        }
    }

    protected virtual void Dispose(bool disposing)
    {
        if (_isDisposed)
            return;

        if (disposing)
        {
            // Sync disposal: Only managed resources; no async here to avoid blocks
            try
            {
                _streamingSttClient.TranscriptionComplete -= OnTranscriptionComplete;
                // Other sync cleanups (e.g., _vad?.Dispose();)
            }
            catch (Exception ex)
            {
                // Log but don't re-throw in Dispose (per guidelines)
                Log.Error(ex, "Error during managed disposal in VoiceAgentCore.");
            }
        }

        // Unmanaged cleanup here if needed (e.g., native handles)

        _isDisposed = true;
    }

    protected virtual async ValueTask DisposeAsyncCore()  // Private async core
    {
        if (_isDisposed)
            return;

        if (_cancellationTokenSource?.IsCancellationRequested != true)  // Avoid redundant shutdown if already canceling
        {
            try
            {
                // Await async shutdown non-blockingly
                await ShutdownAsync().ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
                Log.Debug("Shutdown canceled during async disposal.");
            }
            catch (Exception ex)
            {
                Log.Error(ex, "Error during async shutdown in VoiceAgentCore disposal.");
            }
        }

        // Managed cleanup (events, etc.)—can be async if needed
        _streamingSttClient.TranscriptionComplete -= OnTranscriptionComplete;

        _isDisposed = true;
    }

    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    public async ValueTask DisposeAsync()
    {
        await DisposeAsyncCore().ConfigureAwait(false);
        GC.SuppressFinalize(this);
    }

    ~VoiceAgentCore()
    {
        Dispose(false);
    }
}