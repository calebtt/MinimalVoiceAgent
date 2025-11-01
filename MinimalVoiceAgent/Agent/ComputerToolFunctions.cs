using Microsoft.SemanticKernel;
using NAudio.CoreAudioApi;
using Serilog;
using System.ComponentModel;
using System.Diagnostics; // For Process.Start
using System.Management; // For system info
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization;

namespace MinimalVoiceAgent;

public static partial class Algos
{
    public static bool IsFullPathOrAppID(string input)
    {
        return Path.IsPathRooted(input) || input.Contains('!') || File.Exists(input); // AppID has '!', paths rooted
    }

    public static async Task<string> LaunchDirectlyAsync(string target, string arguments, JsonSerializerOptions jsonOptions)
    {
        var startInfo = new ProcessStartInfo
        {
            FileName = Algos.IsAppID(target) ? "explorer.exe" : target,
            Arguments = Algos.IsAppID(target) ? $"shell:AppsFolder\\{target}!App" : arguments,
            UseShellExecute = true
        };
        using var proc = Process.Start(startInfo);
        if (proc == null) throw new InvalidOperationException("Failed to start app.");

        Log.Information("Launched {Target} with args '{Args}' (PID: {Pid}).", target, arguments, proc.Id);
        await Task.Delay(500); // Yield for launch
        return JsonSerializer.Serialize(new { status = "success", message = $"Launched {target} successfully." }, jsonOptions);
    }

    public static bool IsAppID(string input) => input.Contains('!'); // UWP AppID format

    public static async Task<List<AppCandidate>> SearchStartAppsAsync(string query, int topN)
    {
        var candidates = new List<AppCandidate>();
        var cmd = $@"Get-StartApps | Where-Object {{ $_.Name -like '*{query}*' }} | Select-Object -First {topN} | ForEach-Object {{ [PSCustomObject]@{{""Name"": $_.Name, ""AppID"": $_.AppID, ""TilePath"": $_.TilePath }} }} | ConvertTo-Json -Compress";

        var startInfo = new ProcessStartInfo
        {
            FileName = "powershell.exe",
            Arguments = $"-NoProfile -ExecutionPolicy Bypass -Command \"{cmd}\"",
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true
        };

        using var process = new Process { StartInfo = startInfo };
        process.Start();
        var output = await process.StandardOutput.ReadToEndAsync();
        var error = await process.StandardError.ReadToEndAsync();
        await process.WaitForExitAsync();

        if (process.ExitCode != 0)
        {
            Log.Error("PowerShell search failed: {Error}", error);
            return candidates;
        }

        try
        {
            var jsonArray = JsonSerializer.Deserialize<JsonElement[]>(output);
            foreach (var item in jsonArray)
            {
                candidates.Add(new AppCandidate(
                    Name: item.GetProperty("Name").GetString() ?? string.Empty,
                    AppID: item.TryGetProperty("AppID", out var appId) ? appId.GetString() : null,
                    TilePath: item.TryGetProperty("TilePath", out var path) ? path.GetString() : null
                ));
            }
        }
        catch (JsonException ex)
        {
            Log.Error(ex, "Failed to parse PowerShell JSON: {Output}", output);
        }

        return candidates;
    }

    public static async Task<string?> RunPythonAdDetectorAsync(string mode, string pythonScriptPath, string pythonEnv)
    {
        if (!File.Exists(pythonScriptPath))
        {
            Log.Warning("Python script missing; skipping analysis.");
            return null;
        }

        try
        {
            var startInfo = new ProcessStartInfo
            {
                FileName = pythonEnv,
                Arguments = $"\"{pythonScriptPath}\" --mode {mode}",
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true
            };

            using var process = new Process { StartInfo = startInfo };
            process.Start();
            var outputTask = process.StandardOutput.ReadToEndAsync();
            var errorTask = process.StandardError.ReadToEndAsync();
            await process.WaitForExitAsync(); // .NET 8+ async

            var output = await outputTask;
            var error = await errorTask;

            if (process.ExitCode != 0)
            {
                Log.Error("Python analysis failed (code {Code}): {Error}", process.ExitCode, error);
                return null;
            }

            // Trim whitespace, validate JSON
            output = output.Trim();
            if (string.IsNullOrWhiteSpace(output) || !output.StartsWith("{"))
            {
                Log.Warning("Invalid Python output: {Output}", output);
                return null;
            }

            return output;
        }
        catch (Exception ex)
        {
            Log.Error(ex, "Error running Python for ad detection.");
            return null;
        }
    }

    public record AppCandidate(string Name, string? AppID, string? TilePath);
}

/// <summary>
/// Standalone class with tool functions tailored for common tasks on a nerdy man's computer.
/// Focuses on local system operations like volume control, system info, timers, calculations, reminders, file ops, media automation, and more.
/// Best practices: Async methods where possible, immutable data, explicit error handling, modern C# features (e.g., primary constructors, null-forgiving).
/// Dependencies: NAudio (audio), System.Management (hardware), WindowsInput (simulation), System.Drawing.Common (screenshots).
/// New: Python subprocess for intelligent ad detection via LLaVA captioning (requires Python + torch/transformers installed locally).
/// </summary>
public class ComputerToolFunctions
{
    private readonly JsonSerializerOptions _jsonOptions = new() { WriteIndented = true, DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull };
    private readonly Dictionary<DateTimeOffset, string> _localReminders = new(); // In-memory reminders
    private readonly List<System.Timers.Timer> _activeTimers = new(); // For managing timers
    private readonly string _notificationLogPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData), "VoiceAgent", "notifications.log");
    private readonly string _pythonScriptPath; // Path to the local Python analysis script
    private const int VolumeStep = 10; // Fixed step size for raise/lower (10% increments)
    private const string PythonEnv = "python"; // Assume 'python' in PATH; configurable via env var

    /// <summary>
    /// Primary constructor: Initializes simulation, log dir, and Python script path.
    /// Expects a local Python script at ./python/ad_detector.py (adapted from provided script).
    /// </summary>
    public ComputerToolFunctions()
    {
        Directory.CreateDirectory(Path.GetDirectoryName(_notificationLogPath)!); // Ensure log dir
        _pythonScriptPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "python", "ad_detector.py");

        if (!File.Exists(_pythonScriptPath))
        {
            Log.Warning("Python ad_detector.py missing—ad skip tool will fallback to basic detection. Place it at {Path}", _pythonScriptPath);
        }
    }

    [KernelFunction("get_current_datetime")]
    [Description("Retrieve the current local date and time, useful for scheduling or time-based queries.")]
    public virtual async Task<string> GetCurrentDateTimeAsync(
        [Description("Time zone ID (default: local system time zone)")] string? timeZoneId = null)
    {
        try
        {
            TimeZoneInfo timeZone = string.IsNullOrEmpty(timeZoneId)
                ? TimeZoneInfo.Local
                : TimeZoneInfo.FindSystemTimeZoneById(timeZoneId);

            var currentDateTime = TimeZoneInfo.ConvertTimeFromUtc(DateTime.UtcNow, timeZone);

            Log.Information("GetCurrentDateTime: TimeZone={TimeZone}, Result={DateTime}", timeZone.DisplayName, currentDateTime);

            await Task.CompletedTask;

            return JsonSerializer.Serialize(new
            {
                status = "success",
                dateTime = currentDateTime.ToString("yyyy-MM-dd HH:mm:ss zzz"),
                timeZone = timeZone.DisplayName
            }, _jsonOptions);
        }
        catch (TimeZoneNotFoundException ex)
        {
            Log.Error(ex, "Invalid time zone: {TimeZoneId}", timeZoneId);
            return JsonSerializer.Serialize(new { error = "Invalid time zone", details = ex.Message }, _jsonOptions);
        }
        catch (Exception ex)
        {
            Log.Error(ex, "Tool execution failed for get_current_datetime: {Message}", ex.Message);
            return JsonSerializer.Serialize(new { error = "Execution failed", details = ex.Message }, _jsonOptions);
        }
    }

    [KernelFunction("perform_calculation")]
    [Description("Perform a simple mathematical calculation, e.g., for quick arithmetic or code snippets.")]
    public virtual async Task<string> PerformCalculationAsync(
        [Description("Mathematical expression (e.g., '2 + 3 * 4')")] string expression)
    {
        try
        {
            if (string.IsNullOrWhiteSpace(expression))
            {
                throw new ArgumentException("Expression cannot be empty.");
            }

            var result = new System.Data.DataTable().Compute(expression, null)?.ToString()
                ?? throw new InvalidOperationException("Calculation failed.");

            Log.Information("PerformCalculation: Expression={Expression}, Result={Result}", expression, result);

            await Task.CompletedTask;

            return JsonSerializer.Serialize(new { status = "success", result }, _jsonOptions);
        }
        catch (Exception ex)
        {
            Log.Error(ex, "Tool execution failed for perform_calculation: {Message}", ex.Message);
            return JsonSerializer.Serialize(new { error = "Execution failed", details = ex.Message }, _jsonOptions);
        }
    }

    [KernelFunction("set_local_reminder")]
    [Description("Set a simple in-memory reminder (persists only during runtime; e.g., 'buy new GPU next week').")]
    public virtual async Task<string> SetLocalReminderAsync(
        [Description("Reminder message")] string message,
        [Description("Due time offset from now (e.g., 'in 30 minutes', 'tomorrow at 9AM')")] string dueTime)
    {
        try
        {
            var parsedDueTime = DateTimeOffset.Now + ParseTimeOffset(dueTime);

            _localReminders[parsedDueTime] = message;

            Log.Information("SetLocalReminder: Message={Message}, DueTime={DueTime}", message, parsedDueTime);

            await Task.CompletedTask;

            return JsonSerializer.Serialize(new
            {
                status = "success",
                message = $"Reminder set for {parsedDueTime:yyyy-MM-dd HH:mm:ss}."
            }, _jsonOptions);
        }
        catch (Exception ex)
        {
            Log.Error(ex, "Tool execution failed for set_local_reminder: {Message}", ex.Message);
            return JsonSerializer.Serialize(new { error = "Execution failed", details = ex.Message }, _jsonOptions);
        }
    }

    [KernelFunction("lower_volume")]
    [Description("Lower the system volume by a fixed step (10%). Supported on Windows only.")]
    public virtual async Task<string> LowerVolumeAsync()
    {
        try
        {
            if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                throw new PlatformNotSupportedException("Volume control supported on Windows only.");
            }

            using var deviceEnumerator = new MMDeviceEnumerator();
            using var defaultDevice = deviceEnumerator.GetDefaultAudioEndpoint(DataFlow.Render, Role.Multimedia);

            var currentLevel = (int)(defaultDevice.AudioEndpointVolume.MasterVolumeLevelScalar * 100);
            var newLevel = Math.Max(0, currentLevel - VolumeStep);
            defaultDevice.AudioEndpointVolume.MasterVolumeLevelScalar = newLevel / 100f;
            defaultDevice.AudioEndpointVolume.Mute = false;  // Unmute if adjusting

            Log.Information("System volume lowered from {Current}% to {New}%.", currentLevel, newLevel);

            await Task.CompletedTask;

            return JsonSerializer.Serialize(new { status = "success", message = $"System volume lowered to {newLevel}%." }, _jsonOptions);
        }
        catch (Exception ex)
        {
            Log.Error(ex, "Tool execution failed for lower_volume: {Message}", ex.Message);
            return JsonSerializer.Serialize(new { error = "Execution failed", details = ex.Message }, _jsonOptions);
        }
    }

    [KernelFunction("raise_volume")]
    [Description("Raise the system volume by a fixed step (10%). Supported on Windows only.")]
    public virtual async Task<string> RaiseVolumeAsync()
    {
        try
        {
            if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                throw new PlatformNotSupportedException("Volume control supported on Windows only.");
            }

            using var deviceEnumerator = new MMDeviceEnumerator();
            using var defaultDevice = deviceEnumerator.GetDefaultAudioEndpoint(DataFlow.Render, Role.Multimedia);

            var currentLevel = (int)(defaultDevice.AudioEndpointVolume.MasterVolumeLevelScalar * 100);
            var newLevel = Math.Min(100, currentLevel + VolumeStep);
            defaultDevice.AudioEndpointVolume.MasterVolumeLevelScalar = newLevel / 100f;
            defaultDevice.AudioEndpointVolume.Mute = false;  // Unmute if adjusting

            Log.Information("System volume raised from {Current}% to {New}%.", currentLevel, newLevel);

            await Task.CompletedTask;

            return JsonSerializer.Serialize(new { status = "success", message = $"System volume raised to {newLevel}%." }, _jsonOptions);
        }
        catch (Exception ex)
        {
            Log.Error(ex, "Tool execution failed for raise_volume: {Message}", ex.Message);
            return JsonSerializer.Serialize(new { error = "Execution failed", details = ex.Message }, _jsonOptions);
        }
    }

    [KernelFunction("mute_volume")]
    [Description("Mute the system volume (sets mute to true). Supported on Windows only.")]
    public virtual async Task<string> MuteVolumeAsync()
    {
        try
        {
            if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                throw new PlatformNotSupportedException("Volume control supported on Windows only.");
            }

            using var deviceEnumerator = new MMDeviceEnumerator();
            using var defaultDevice = deviceEnumerator.GetDefaultAudioEndpoint(DataFlow.Render, Role.Multimedia);

            var wasMuted = defaultDevice.AudioEndpointVolume.Mute;
            defaultDevice.AudioEndpointVolume.Mute = true;

            Log.Information("System volume muted (was already muted: {WasMuted}).", wasMuted);

            await Task.CompletedTask;

            return JsonSerializer.Serialize(new { status = "success", message = "System volume muted." }, _jsonOptions);
        }
        catch (Exception ex)
        {
            Log.Error(ex, "Tool execution failed for mute_volume: {Message}", ex.Message);
            return JsonSerializer.Serialize(new { error = "Execution failed", details = ex.Message }, _jsonOptions);
        }
    }

    [KernelFunction("get_system_volume")]
    [Description("Retrieve the current system volume level (0-100) and mute status. Supported on Windows only.")]
    public virtual async Task<string> GetSystemVolumeAsync()
    {
        try
        {
            if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                throw new PlatformNotSupportedException("Volume control supported on Windows only.");
            }

            using var deviceEnumerator = new MMDeviceEnumerator();
            using var defaultDevice = deviceEnumerator.GetDefaultAudioEndpoint(DataFlow.Render, Role.Multimedia);

            var level = (int)(defaultDevice.AudioEndpointVolume.MasterVolumeLevelScalar * 100);
            var isMuted = defaultDevice.AudioEndpointVolume.Mute;

            Log.Information("Current system volume: {Level}% (muted: {IsMuted}).", level, isMuted);

            await Task.CompletedTask;

            return JsonSerializer.Serialize(new
            {
                status = "success",
                level = level,
                isMuted = isMuted,
                message = $"Current volume: {level}% (muted: {isMuted})."
            }, _jsonOptions);
        }
        catch (Exception ex)
        {
            Log.Error(ex, "Tool execution failed for get_system_volume: {Message}", ex.Message);
            return JsonSerializer.Serialize(new { error = "Execution failed", details = ex.Message }, _jsonOptions);
        }
    }

    [KernelFunction("get_system_info")]
    [Description("Get basic system information like CPU usage, memory, or hardware specs for troubleshooting.")]
    public virtual async Task<string> GetSystemInfoAsync(
        [Description("Type of info (e.g., 'cpu', 'memory', 'all')")] string infoType = "all")
    {
        try
        {
            var info = new Dictionary<string, object>();

            if (infoType == "cpu" || infoType == "all")
            {
                using var searcher = new ManagementObjectSearcher("select * from Win32_Processor");
                foreach (ManagementObject? obj in searcher.Get())
                {
                    if (obj != null)
                    {
                        info["CPU Name"] = obj["Name"]?.ToString();
                        info["CPU Cores"] = obj["NumberOfCores"]?.ToString();
                    }
                }
            }

            if (infoType == "memory" || infoType == "all")
            {
                using var searcher = new ManagementObjectSearcher("select * from Win32_OperatingSystem");
                foreach (ManagementObject? obj in searcher.Get())
                {
                    if (obj != null)
                    {
                        info["Total Memory (MB)"] = Convert.ToUInt64(obj["TotalVisibleMemorySize"]) / 1024;
                        info["Free Memory (MB)"] = Convert.ToUInt64(obj["FreePhysicalMemory"]) / 1024;
                    }
                }
            }

            Log.Information("GetSystemInfo: Type={InfoType}, Result={Result}", infoType, JsonSerializer.Serialize(info));

            await Task.CompletedTask;

            return JsonSerializer.Serialize(new { status = "success", info }, _jsonOptions);
        }
        catch (Exception ex)
        {
            Log.Error(ex, "Tool execution failed for get_system_info: {Message}", ex.Message);
            return JsonSerializer.Serialize(new { error = "Execution failed", details = ex.Message }, _jsonOptions);
        }
    }

    [KernelFunction("set_timer")]
    [Description("Set a timer that logs or alerts after a duration (e.g., '5 minutes for compile').")]
    public virtual async Task<string> SetTimerAsync(
        [Description("Duration in seconds")] int durationSeconds,
        [Description("Message for when timer ends")] string? message = "Timer ended.")
    {
        try
        {
            if (durationSeconds <= 0)
            {
                throw new ArgumentException("Duration must be positive.");
            }

            var timer = new System.Timers.Timer(durationSeconds * 1000) { AutoReset = false };
            timer.Elapsed += (sender, e) =>
            {
                Log.Information("Timer ended: {Message}", message);
                // TODO: Add alert, e.g., play sound or notify via send_notification
                _activeTimers.Remove((System.Timers.Timer)sender!);
            };
            timer.Start();
            _activeTimers.Add(timer);

            Log.Information("SetTimer: Duration={Duration}s, Message={Message}", durationSeconds, message);

            await Task.CompletedTask;

            return JsonSerializer.Serialize(new { status = "success", message = $"Timer set for {durationSeconds} seconds." }, _jsonOptions);
        }
        catch (Exception ex)
        {
            Log.Error(ex, "Tool execution failed for set_timer: {Message}", ex.Message);
            return JsonSerializer.Serialize(new { error = "Execution failed", details = ex.Message }, _jsonOptions);
        }
    }

    [KernelFunction("send_notification")]
    [Description("Log a message or issue for follow-up (treat 'message' as issue for general interactions).")]
    public virtual async Task<string> SendNotificationAsync(
        [Description("Message content or issue description")] string issue,
        [Description("Location or context (optional)")] string? location = "",
        [Description("Urgency level (default 'medium')")] string? urgency = "medium",
        [Description("User's name")] string? callerName = "")
    {
        try
        {
            var entry = new
            {
                Timestamp = DateTimeOffset.Now,
                Issue = issue,
                Location = location ?? string.Empty,
                Urgency = Enum.TryParse<UrgencyLevel>(urgency, true, out var parsedUrgency) ? parsedUrgency : UrgencyLevel.Medium,
                CallerName = callerName ?? string.Empty
            };

            var jsonEntry = JsonSerializer.Serialize(entry, _jsonOptions);
            await File.AppendAllTextAsync(_notificationLogPath, jsonEntry + Environment.NewLine);

            Log.Information("Notification logged: {Issue} (urgency: {Urgency})", issue, entry.Urgency);

            await Task.CompletedTask;

            return JsonSerializer.Serialize(new { status = "success", message = "Notification logged successfully." }, _jsonOptions);
        }
        catch (Exception ex)
        {
            Log.Error(ex, "Tool execution failed for send_notification: {Message}", ex.Message);
            return JsonSerializer.Serialize(new { error = "Execution failed", details = ex.Message }, _jsonOptions);
        }
    }

    [KernelFunction("launch_application")]
    [Description("Launch a desktop application by executable path, AppID, or partial name. Uses Windows Start menu search for fuzzy matching; returns candidates if topN > 1 for LLM selection.")]
    public virtual async Task<string> LaunchApplicationAsync(
            [Description("Path, AppID, or friendly name (e.g., 'notepad', 'Microsoft.WindowsCalculator_8wekyb3d8bbwe!App')")] string appPathOrName,
            [Description("Optional command-line args (e.g., '--new-window')")] string? arguments = "",
            [Description("Top N search results to return (default 1: launch top; >1: return list without launching)")] int topN = 1)
    {
        if (topN < 1 || topN > 10) throw new ArgumentOutOfRangeException(nameof(topN), "TopN must be 1-10.");

        try
        {
            // If full path or AppID provided, launch directly
            if (Algos.IsFullPathOrAppID(appPathOrName))
            {
                return await Algos.LaunchDirectlyAsync(appPathOrName, arguments ?? string.Empty, _jsonOptions);
            }

            // Else, search Start menu
            var candidates = await Algos.SearchStartAppsAsync(appPathOrName, topN);
            if (!candidates.Any())
            {
                throw new FileNotFoundException($"No apps found matching '{appPathOrName}'. Try a full path or AppID.");
            }

            if (topN == 1)
            {
                // Launch top match
                var top = candidates.First();
                return await Algos.LaunchDirectlyAsync(top.AppID ?? top.TilePath ?? top.Name, arguments ?? string.Empty, _jsonOptions);
            }
            else
            {
                // Return list for LLM
                var result = candidates.Select(c => new { c.Name, c.AppID, c.TilePath }).ToList();
                Log.Information("Returned {Count} app candidates for '{Query}'.", candidates.Count, appPathOrName);
                return JsonSerializer.Serialize(new { status = "candidates", results = result, message = $"Found {candidates.Count} matching apps." }, _jsonOptions);
            }
        }
        catch (Exception ex)
        {
            Log.Error(ex, "Launch failed for {App}.", appPathOrName);
            return JsonSerializer.Serialize(new { error = "Execution failed", details = ex.Message }, _jsonOptions);
        }
    }

    [KernelFunction("open_file")]
    [Description("Open a file or folder using the default system handler (e.g., for PDFs, images).")]
    public virtual async Task<string> OpenFileAsync(
        [Description("Full path to file or folder")] string filePath)
    {
        try
        {
            if (!File.Exists(filePath) && !Directory.Exists(filePath))
                throw new FileNotFoundException($"Path not found: {filePath}");

            Process.Start(new ProcessStartInfo(filePath) { UseShellExecute = true });

            Log.Information("Opened {Path}.", filePath);

            await Task.CompletedTask;

            return JsonSerializer.Serialize(new { status = "success", message = $"Opened {Path.GetFileName(filePath)}." }, _jsonOptions);
        }
        catch (Exception ex)
        {
            Log.Error(ex, "Open failed for {Path}.", filePath);
            return JsonSerializer.Serialize(new { error = "Execution failed", details = ex.Message }, _jsonOptions);
        }
    }

    [KernelFunction("search_local_files")]
    [Description("Search for files in a directory by pattern (e.g., '*.txt' in 'Documents'). Returns top matches.")]
    public virtual async Task<string> SearchLocalFilesAsync(
        [Description("Root directory (default: user Documents)")] string? directory = null,
        [Description("File pattern (e.g., '*.py', 'report.*')")] string pattern = "",
        [Description("Max files to return")] int maxResults = 10)
    {
        try
        {
            if (string.IsNullOrWhiteSpace(pattern)) throw new ArgumentException("Pattern required.");

            var dir = string.IsNullOrEmpty(directory) ? Environment.ExpandEnvironmentVariables("%USERPROFILE%\\Documents") : Environment.ExpandEnvironmentVariables(directory);
            if (!Directory.Exists(dir)) throw new DirectoryNotFoundException($"Directory not found: {dir}");

            var matches = Directory.EnumerateFiles(dir, pattern, SearchOption.AllDirectories)
                .Take(maxResults)
                .Select(f => Path.GetFileName(f))
                .ToList();

            var result = matches.Any() ? string.Join(Environment.NewLine, matches) : "No matches found.";

            Log.Information("Searched {Dir} for '{Pattern}': {Count} results.", dir, pattern, matches.Count);

            await Task.CompletedTask;

            return JsonSerializer.Serialize(new { status = "success", results = matches, message = result }, _jsonOptions);
        }
        catch (Exception ex)
        {
            Log.Error(ex, "Search failed for pattern '{Pattern}' in {Dir}.", pattern, directory);
            return JsonSerializer.Serialize(new { error = "Execution failed", details = ex.Message }, _jsonOptions);
        }
    }

    [KernelFunction("skip_youtube_ad")]
    [Description("Skip YouTube ads: Runs Python script to capture/analyze screen with Florence-2, detect 'Skip' button, click if found, verify success.")]
    public virtual async Task<string> SkipYoutubeAdAsync(
        [Description("Screenshot interval (default 2s)")] int pollIntervalSeconds = 2,
        [Description("Max wait time for button (default 30s)")] int maxPollSeconds = 30)
    {
        try
        {
            if (!File.Exists(_pythonScriptPath))
            {
                Log.Warning("Python script missing; ad skip unavailable.");
                return JsonSerializer.Serialize(new { error = "Script missing", details = "Place adskip5.py at {Path}" }, _jsonOptions);
            }

            var args = $"\"{_pythonScriptPath}\" --mode live --interval {pollIntervalSeconds} --max-poll-seconds {maxPollSeconds}";
            var startInfo = new ProcessStartInfo
            {
                FileName = PythonEnv,
                Arguments = args,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true
            };

            using var process = new Process { StartInfo = startInfo };
            process.Start();
            var output = await process.StandardOutput.ReadToEndAsync();
            var error = await process.StandardError.ReadToEndAsync();
            await process.WaitForExitAsync();

            if (process.ExitCode != 0)
            {
                Log.Error("Python ad skip failed (code {Code}): {Error}", process.ExitCode, error);
                return JsonSerializer.Serialize(new { error = "Execution failed", details = error }, _jsonOptions);
            }

            // Parse JSON result
            var jsonNode = JsonNode.Parse(output.Trim());
            if (jsonNode == null)
            {
                Log.Warning("Invalid Python JSON: {Output}", output);
                return JsonSerializer.Serialize(new { status = "partial", message = "Detection failed—check logs." }, _jsonOptions);
            }

            var status = jsonNode["status"]?.ToString() ?? "partial";
            var message = jsonNode["message"]?.ToString() ?? "Unknown result.";
            Log.Information("Ad skip result: {Status} - {Message}", status, message);

            return JsonSerializer.Serialize(new { status, message }, _jsonOptions);
        }
        catch (Exception ex)
        {
            Log.Error(ex, "Ad skip failed.");
            return JsonSerializer.Serialize(new { error = "Execution failed", details = ex.Message }, _jsonOptions);
        }
    }

    [KernelFunction("lock_screen")]
    [Description("Lock the Windows screen immediately (user session).")]
    public virtual async Task<string> LockScreenAsync()
    {
        try
        {
            if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                throw new PlatformNotSupportedException("Lock screen supported on Windows only.");
            }

            LockWorkStation();

            Log.Information("Screen locked.");

            await Task.CompletedTask;

            return JsonSerializer.Serialize(new { status = "success", message = "Screen locked." }, _jsonOptions);
        }
        catch (Exception ex)
        {
            Log.Error(ex, "Lock screen failed: {Message}", ex.Message);
            return JsonSerializer.Serialize(new { error = "Execution failed", details = ex.Message }, _jsonOptions);
        }
    }

    // Helpers (private, robust, modern)
    private static TimeSpan ParseTimeOffset(string dueTime)
    {
        // Simplified parser; use Humanizer or NodaTime for prod robustness
        var parts = dueTime.ToLowerInvariant().Split(' ');
        if (parts.Length < 2) throw new ArgumentException($"Unable to parse due time: {dueTime}");

        var unit = parts[1];
        if (!int.TryParse(parts[0].Replace("in ", ""), out var value)) throw new ArgumentException($"Unable to parse due time: {dueTime}");

        return unit switch
        {
            "minute" or "minutes" => TimeSpan.FromMinutes(value),
            "hour" or "hours" => TimeSpan.FromHours(value),
            "day" or "days" => TimeSpan.FromDays(value),
            "tomorrow" => TimeSpan.FromDays(1),
            _ => throw new ArgumentException($"Unsupported unit in due time: {dueTime}")
        };
    }

    // P/Invoke for LockWorkStation and screen metrics
    [DllImport("user32.dll")]
    private static extern bool LockWorkStation();

    [DllImport("user32.dll")]
    private static extern int GetSystemMetrics(int nIndex);

    private const int SM_CXSCREEN = 0;
    private const int SM_CYSCREEN = 1;

    // Enum for urgency
    private enum UrgencyLevel { Low, Medium, High }
}