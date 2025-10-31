﻿using AudioProcessingModuleCs.Media.Dsp.WebRtc;
using Serilog;
using System.Collections.Concurrent;
using System.Diagnostics;

namespace MinimalVoiceAgent;

public class AudioPacer : IDisposable
{
    private const int FrameDurationMs = 20;
    private const int FrameSizeBytes = 640;  // 16000 Hz * 2 bytes/sample * 0.02s = 640 bytes
    private readonly byte[] _silenceFrame = new byte[FrameSizeBytes];  // Zeros for silence
    private readonly ConcurrentQueue<byte[]> _queue = new();
    private CancellationTokenSource? _cts;
    private Task? _pacerTask;
    private volatile Func<byte[], byte[]>? _currentFilter;
    private volatile bool _hasAudioPending;
    private Action<byte[]>? _playAction;  // Callback to play a single frame (e.g., add to BufferedWaveProvider)

    public event Action? SendingComplete;

    public bool IsAudioPlaying => !_queue.IsEmpty;

    public void Initialize(Action<byte[]> playAction)
    {
        ArgumentNullException.ThrowIfNull(playAction);
        _playAction = playAction;

        if (_pacerTask != null && !_pacerTask.IsCompleted)
        {
            throw new InvalidOperationException("The pacer is already started.");
        }

        _cts?.Dispose();
        _cts = new CancellationTokenSource();
        _pacerTask = Task.Run(() => RunAsync(_cts.Token), _cts.Token);
    }

    public async Task StopAsync()
    {
        if (_cts == null) return;

        await _cts.CancelAsync();

        // Flush queue
        while (_queue.TryDequeue(out _)) { }

        if (_hasAudioPending)
        {
            _hasAudioPending = false;
            SendingComplete?.Invoke();
        }

        if (_pacerTask != null)
        {
            try
            {
                await _pacerTask;
            }
            catch (TaskCanceledException) { }
            _pacerTask = null;
        }

        _playAction = null;
        ClearFilter();
        _cts.Dispose();
        _cts = null;
    }

    public void ResetBuffer()
    {
        if (_pacerTask == null || _pacerTask.IsCompleted)
        {
            Log.Warning("Cannot reset buffer: pacer is not running.");
            return;
        }

        if (_hasAudioPending)
        {
            _hasAudioPending = false;
            SendingComplete?.Invoke();
        }

        while (_queue.TryDequeue(out _)) { }

        Log.Information("Buffer reset, queue cleared.");
    }

    public void ApplyFilter(Func<byte[], byte[]> filter)
    {
        ArgumentNullException.ThrowIfNull(filter);
        _currentFilter = filter;
        Log.Information("Applied audio filter.");
    }

    public void ClearFilter()
    {
        _currentFilter = null;
        Log.Debug("Audio filter cleared.");
    }

    private async Task RunAsync(CancellationToken token)
    {
        var stopwatch = Stopwatch.StartNew();
        long expectedElapsedMs = 0;

        while (!token.IsCancellationRequested)
        {
            if (_playAction != null)
            {
                if (!_queue.TryDequeue(out var frame))
                {
                    frame = _silenceFrame;
                }

                var frameToPlay = frame.AsSpan().ToArray();  // Copy for safety

                // Apply filter if active
                var filter = _currentFilter;
                if (filter != null)
                {
                    try
                    {
                        frameToPlay = filter(frameToPlay);
                    }
                    catch (Exception ex)
                    {
                        Log.Error(ex, "Error applying audio filter; playing unfiltered frame.");
                    }
                }

                _playAction(frameToPlay);

                // Detect completion
                if (_hasAudioPending && _queue.IsEmpty && !frame.AsSpan().SequenceEqual(_silenceFrame))
                {
                    _hasAudioPending = false;
                    SendingComplete?.Invoke();
                    Log.Debug("Sending complete: All real audio frames sent.");
                }

                expectedElapsedMs += FrameDurationMs;
            }

            var actualElapsed = stopwatch.ElapsedMilliseconds;
            var delayMs = expectedElapsedMs - actualElapsed;
            if (delayMs > 0)
            {
                await Task.Delay((int)delayMs, token);
            }
            else
            {
                await Task.Yield();
            }
        }
    }

    public void EnqueueBufferForSendManual(byte[] pcmBytes)
    {
        if (pcmBytes == null || pcmBytes.Length == 0 || _pacerTask == null)
            return;

        var offset = 0;
        while (offset + FrameSizeBytes <= pcmBytes.Length)
        {
            var frame = new byte[FrameSizeBytes];
            Array.Copy(pcmBytes, offset, frame, 0, FrameSizeBytes);
            _hasAudioPending = true;  // Mark pending for real audio
            _queue.Enqueue(frame);
            offset += FrameSizeBytes;
        }

        if (offset < pcmBytes.Length)
        {
            var remaining = pcmBytes.Length - offset;
            Log.Debug($"Discarded {remaining} PCM bytes (incomplete frame).");
        }
    }

    public void EnableWebRtcProcessing(WebRtcFilter webRtcFilter)
    {
        ApplyFilter(pcmFrame =>
        {
            try
            {
                // Ensure frame is exactly 640 bytes (pad if needed, though Enqueue should ensure)
                if (pcmFrame.Length < FrameSizeBytes)
                {
                    byte[] paddedFrame = new byte[FrameSizeBytes];
                    Array.Copy(pcmFrame, paddedFrame, pcmFrame.Length);
                    pcmFrame = paddedFrame;
                }

                // Register this frame as played for AEC reference
                webRtcFilter.RegisterFramePlayed(pcmFrame);

                return pcmFrame;  // Return original (or padded); API doesn't modify played frames
            }
            catch (Exception ex)
            {
                Log.Error(ex, "WebRtc processing failed on played frame; skipping.");
                return pcmFrame;  // Fallback
            }
        });

        Log.Information("Enabled WebRtc processing in AudioPacer filter chain.");
    }

    public void Dispose()
    {
        StopAsync().GetAwaiter().GetResult();  // Sync for Dispose
        GC.SuppressFinalize(this);
    }
}