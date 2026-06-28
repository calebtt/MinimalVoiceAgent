using System.Net.Sockets;
using Xunit;

namespace MinimalVoiceAgent.Tests;

/// <summary>
/// Tests for how <see cref="CleanSpeechDaemonProcess"/> resolves the launch command from the
/// daemon's virtualenv layout.
/// </summary>
public class CleanSpeechDaemonProcessTests : IDisposable
{
    private readonly string _dir;

    public CleanSpeechDaemonProcessTests()
    {
        _dir = Path.Combine(Path.GetTempPath(), "csd-proc-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(Path.Combine(_dir, ".venv", "bin"));
    }

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* best effort */ }
    }

    private string VenvBin => Path.Combine(_dir, ".venv", "bin");

    [Fact]
    public void ResolveLaunchCommand_PrefersConsoleScript()
    {
        var script = Path.Combine(VenvBin, "clean-speech-daemon");
        File.WriteAllText(script, "#!/bin/sh\n");

        var (exe, args) = CleanSpeechDaemonProcess.ResolveLaunchCommand(_dir);

        Assert.Equal(script, exe);
        Assert.Equal("run", args);
    }

    [Fact]
    public void BuildRunArguments_IncludesConfigWhenFileExists()
    {
        var config = Path.Combine(_dir, "agent.toml");
        File.WriteAllText(config, "enable_vad = false\n");

        var args = CleanSpeechDaemonProcess.BuildRunArguments(config);

        Assert.StartsWith("run --config \"", args);
        Assert.Contains("agent.toml", args);
    }

    [Fact]
    public void BuildRunArguments_FallsBackWhenConfigMissing()
    {
        Assert.Equal("run", CleanSpeechDaemonProcess.BuildRunArguments(Path.Combine(_dir, "missing.toml")));
        Assert.Equal("run", CleanSpeechDaemonProcess.BuildRunArguments(null));
    }

    [Fact]
    public void ResolveLaunchCommand_FallsBackToPythonModule()
    {
        var python = Path.Combine(VenvBin, "python");
        File.WriteAllText(python, "#!/bin/sh\n");

        var (exe, args) = CleanSpeechDaemonProcess.ResolveLaunchCommand(_dir);

        Assert.Equal(python, exe);
        Assert.Equal("-m clean_speech_daemon run", args);
    }

    [Fact]
    public void ResolveLaunchCommand_ThrowsWhenVenvMissing()
    {
        // .venv/bin exists but contains neither the console script nor python.
        Assert.Throws<FileNotFoundException>(() => CleanSpeechDaemonProcess.ResolveLaunchCommand(_dir));
    }

    [Fact]
    public void Constructor_RejectsBlankArguments()
    {
        Assert.Throws<ArgumentException>(() => new CleanSpeechDaemonProcess("  ", "/tmp/x.sock"));
        Assert.Throws<ArgumentException>(() => new CleanSpeechDaemonProcess("/tmp/dir", " "));
    }

    [Fact]
    public void TryProbeHandshake_ReturnsFalseForMissingPath()
    {
        Assert.False(CleanSpeechDaemonProcess.TryProbeHandshake(Path.Combine(_dir, "missing.sock")));
    }

    [Fact]
    public void TryProbeHandshake_ReturnsFalseForStaleSocketFile()
    {
        if (!OperatingSystem.IsLinux() && !OperatingSystem.IsMacOS())
            return;

        var stale = Path.Combine(_dir, "stale.sock");
        File.WriteAllText(stale, string.Empty);

        Assert.False(CleanSpeechDaemonProcess.TryProbeHandshake(stale));
    }

    [Fact]
    public void TryProbeHandshake_ReturnsTrueWhenMetadataLineIsSent()
    {
        if (!OperatingSystem.IsLinux() && !OperatingSystem.IsMacOS())
            return;

        var path = Path.Combine(_dir, "live.sock");
        using var server = new Socket(AddressFamily.Unix, SocketType.Stream, ProtocolType.Unspecified);
        server.Bind(new UnixDomainSocketEndPoint(path));
        server.Listen(1);

        var acceptTask = Task.Run(() =>
        {
            using var client = server.Accept();
            var metadata = "{\"format\":\"s16le\",\"sample_rate\":48000,\"channels\":1}\n"u8.ToArray();
            client.Send(metadata);
        });

        Assert.True(CleanSpeechDaemonProcess.TryProbeHandshake(path));
        acceptTask.Wait(TimeSpan.FromSeconds(2));
    }
}
