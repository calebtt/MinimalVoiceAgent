# Clean-Speech-Daemon Capture

The agent can take its microphone audio from the
[`clean-speech-daemon`](https://github.com/calebtt/clean-speech) (bundled as the `clean-speech`
submodule) instead of capturing the local mic directly.
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

## One-time setup

The daemon is a Python service bundled as a submodule. Initialize it and create its virtualenv:

```bash
git submodule update --init clean-speech
scripts/setup-clean-speech-daemon.sh
```

The setup script creates `clean-speech/.venv`, installs the daemon into it, and writes a default
daemon config if you don't already have one. (Neural echo cancellers need extra deps — the script
prints the one-liner to add them.)

## Configuration

Daemon capture and auto-start are **on by default** in `sttsettings.json`:

```json
"Capture": {
  "UseCleanSpeechDaemon": true,
  "AutoStartDaemon": true,
  "SocketPath": "/tmp/clean-speech-daemon.sock",
  "DaemonDirectory": "../clean-speech",
  "DaemonStartupTimeoutSeconds": 20
}
```

- **`AutoStartDaemon: true`** (default) — the agent launches the daemon itself (from `DaemonDirectory/.venv`),
  waits for its socket, and stops it on shutdown. If a daemon is already running, the agent uses it
  and leaves it alone.
- **`AutoStartDaemon: false`** — start it yourself (`clean-speech/.venv/bin/clean-speech-daemon run`)
  and the agent just connects.
- Set **`UseCleanSpeechDaemon: false`** to use the local microphone with WebRTC APM instead.

On startup you should see:

```
Starting clean-speech-daemon: .../clean-speech/.venv/bin/clean-speech-daemon run
clean-speech-daemon is up (socket /tmp/clean-speech-daemon.sock).
clean-speech-daemon connected: 48000 Hz 1 ch s16le -> resampling to 16000 Hz
Microphone source: clean-speech-daemon (...). Internal capture and APM disabled.
Audio initialized (playback-only): 16000Hz 1ch S16
```

If the daemon can't be started or reached, the agent logs a warning and **falls back to the local
microphone** (with the built-in APM), so it still runs without the daemon. If the daemon disconnects
mid-session, the capture source **reconnects automatically** with exponential backoff. Requires
Linux or macOS (Unix domain sockets).

## Notes

- The daemon emits little-endian 16-bit mono PCM at 48 kHz (per its socket header); the source
  resamples to the agent's 16 kHz pipeline rate.
- When daemon capture is active the agent uses an energy-based segmenter instead of Silero VAD,
  so utterances are not gated twice. Auto-start uses the bundled `daemon/clean-speech-agent.toml`
  profile, which keeps echo cancellation and noise suppression on but sets `enable_vad = false`.
