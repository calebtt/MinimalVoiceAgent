# Clean-Speech-Daemon Capture

The agent can take its microphone audio from the external
[`clean-speech-daemon`](https://github.com/calebtt) instead of capturing the local mic directly.
The daemon references the **system playback monitor**, so it removes *all* speaker output
(the agent's own TTS, music, video, any app) plus background noise before the audio reaches STT —
something the agent's built-in WebRTC APM cannot do, because APM's only echo reference is the
agent's own render stream (its TTS).

## How it works

```
clean-speech-daemon  ──(Unix socket: JSON header + s16le PCM @48k)──►  CleanSpeechDaemonCaptureSource
                                                                          │  resample 48k → 16k, 20ms frames
                                                                          ▼
                                                                    VoiceAgentCore → VAD → STT → LLM → TTS
```

When this source is active, `SoundFlowAudioRouter` runs in **playback-only** mode: it opens no
microphone and constructs **no WebRTC APM**, since the daemon already performs echo cancellation
and noise suppression. SoundFlow is used only to play TTS.

## Enabling it

1. Run the daemon (see its repo): `clean-speech-daemon run`. It publishes the cleaned stream on
   `/tmp/clean-speech-daemon.sock`.
2. In `MinimalVoiceAgent/sttsettings.json`:

   ```json
   "Capture": {
     "UseCleanSpeechDaemon": true,
     "SocketPath": "/tmp/clean-speech-daemon.sock"
   }
   ```

3. Start the agent. You should see:

   ```
   clean-speech-daemon connected: 48000 Hz 1 ch s16le -> resampling to 16000 Hz
   Microphone source: clean-speech-daemon (...). Internal capture and APM disabled.
   Audio initialized (playback-only): 16000Hz 1ch S16
   ```

If the socket is unavailable at startup, the agent logs a warning and **falls back to the local
microphone** (with the built-in APM), so it still runs without the daemon. Default is off.

## Notes

- The daemon emits little-endian 16-bit mono PCM at 48 kHz (per its socket header); the source
  resamples to the agent's 16 kHz pipeline rate.
- The daemon also does its own VAD gating; the agent still runs its Silero VAD to segment
  utterances. This is complementary, not conflicting.
