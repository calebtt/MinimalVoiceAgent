using Serilog;
using System.Diagnostics;
using System.Net.Sockets;

namespace MinimalVoiceAgent;

/// <summary>
/// Optionally launches and supervises the external <c>clean-speech-daemon</c> process (bundled as
/// the <c>clean-speech</c> submodule). If a daemon is already running (its socket accepts
/// connections) this does nothing and reports that it does not own the process. Stale socket files
/// left after a crash are ignored. When it does start the daemon, it waits for
/// the socket to appear and terminates the process on disposal.
/// </summary>
public sealed class CleanSpeechDaemonProcess : IAsyncDisposable
{
    private readonly string _workingDirectory;
    private readonly string _socketPath;
    private readonly string? _configPath;
    private Process? _process;

    public CleanSpeechDaemonProcess(string workingDirectory, string socketPath, string? configPath = null)
    {
        if (string.IsNullOrWhiteSpace(workingDirectory))
            throw new ArgumentException("Daemon directory is required.", nameof(workingDirectory));
        if (string.IsNullOrWhiteSpace(socketPath))
            throw new ArgumentException("Socket path is required.", nameof(socketPath));
        _workingDirectory = workingDirectory;
        _socketPath = socketPath;
        _configPath = string.IsNullOrWhiteSpace(configPath) ? null : configPath;
    }

    /// <summary>
    /// Ensures the daemon is running. Returns <c>true</c> if this call started it (and therefore owns
    /// its lifecycle), or <c>false</c> if a daemon was already running.
    /// </summary>
    public async Task<bool> EnsureStartedAsync(TimeSpan timeout, CancellationToken ct)
    {
        if (IsSocketLive(_socketPath))
        {
            Log.Information("clean-speech-daemon already running (socket {Path} accepting connections); not starting another.", _socketPath);
            return false;
        }

        if (File.Exists(_socketPath))
        {
            Log.Warning(
                "clean-speech-daemon socket {Path} exists but is not accepting connections (stale); starting a fresh daemon.",
                _socketPath);
        }

        var dir = Path.GetFullPath(_workingDirectory);
        if (!Directory.Exists(dir))
            throw new DirectoryNotFoundException(
                $"clean-speech-daemon directory '{dir}' not found. Did you init the submodule (git submodule update --init)?");

        var (exe, args) = ResolveLaunchCommand(dir, _configPath);
        Log.Information("Starting clean-speech-daemon: {Exe} {Args} (cwd {Dir})", exe, args, dir);

        var psi = new ProcessStartInfo
        {
            FileName = exe,
            Arguments = args,
            WorkingDirectory = dir,
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
        };
        _process = Process.Start(psi) ?? throw new InvalidOperationException("Failed to start clean-speech-daemon process.");
        _process.OutputDataReceived += (_, e) => { if (e.Data is { } line) Log.Debug("[daemon] {Line}", line); };
        _process.ErrorDataReceived += (_, e) => { if (e.Data is { } line) Log.Debug("[daemon] {Line}", line); };
        _process.BeginOutputReadLine();
        _process.BeginErrorReadLine();

        await WaitForSocketAsync(timeout, ct);
        Log.Information("clean-speech-daemon is up (socket {Path}).", _socketPath);
        return true;
    }

    /// <summary>Resolves how to launch the daemon from its venv. Prefers the console script.</summary>
    internal static (string exe, string args) ResolveLaunchCommand(string daemonDir, string? configPath = null)
    {
        var runArgs = BuildRunArguments(configPath);
        var venvBin = Path.Combine(daemonDir, ".venv", "bin");
        var consoleScript = Path.Combine(venvBin, "clean-speech-daemon");
        if (File.Exists(consoleScript))
            return (consoleScript, runArgs);

        var python = Path.Combine(venvBin, "python");
        if (File.Exists(python))
            return (python, $"-m clean_speech_daemon {runArgs}");

        throw new FileNotFoundException(
            $"No clean-speech-daemon virtualenv found under '{Path.Combine(daemonDir, ".venv")}'. " +
            "Run scripts/setup-clean-speech-daemon.sh once to create it.");
    }

    internal static string BuildRunArguments(string? configPath)
    {
        if (string.IsNullOrWhiteSpace(configPath))
            return "run";

        var fullPath = Path.GetFullPath(configPath);
        if (!File.Exists(fullPath))
            return "run";

        return $"run --config \"{fullPath}\"";
    }

    /// <summary>Returns true when a Unix socket path has a live listener (not a stale file).</summary>
    internal static bool IsSocketLive(string socketPath)
    {
        if (string.IsNullOrWhiteSpace(socketPath) || !File.Exists(socketPath))
            return false;

        if (!OperatingSystem.IsLinux() && !OperatingSystem.IsMacOS())
            return true;

        try
        {
            using var socket = new Socket(AddressFamily.Unix, SocketType.Stream, ProtocolType.Unspecified);
            socket.Connect(new UnixDomainSocketEndPoint(socketPath));
            return true;
        }
        catch (SocketException ex) when (ex.SocketErrorCode == SocketError.ConnectionRefused)
        {
            return false;
        }
        catch
        {
            return false;
        }
    }

    private async Task WaitForSocketAsync(TimeSpan timeout, CancellationToken ct)
    {
        var deadline = DateTime.UtcNow + timeout;
        while (DateTime.UtcNow < deadline)
        {
            if (IsSocketLive(_socketPath))
                return;
            if (_process is { HasExited: true })
                throw new InvalidOperationException($"clean-speech-daemon exited early (code {_process.ExitCode}) before publishing its socket.");
            await Task.Delay(200, ct);
        }
        throw new TimeoutException($"clean-speech-daemon socket '{_socketPath}' did not appear within {timeout.TotalSeconds:0}s.");
    }

    public async ValueTask DisposeAsync()
    {
        if (_process is null)
            return;

        try
        {
            if (!_process.HasExited)
            {
                _process.Kill(entireProcessTree: true);
                await _process.WaitForExitAsync();
            }
        }
        catch (Exception ex)
        {
            Log.Debug(ex, "Error stopping clean-speech-daemon process.");
        }
        finally
        {
            _process.Dispose();
            _process = null;
        }
    }
}
