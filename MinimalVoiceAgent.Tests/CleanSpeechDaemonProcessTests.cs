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
}
