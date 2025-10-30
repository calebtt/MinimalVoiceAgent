using Microsoft.SemanticKernel;
using Serilog;
using System.ComponentModel;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Runtime.InteropServices;
using NAudio.CoreAudioApi; // Assuming NAudio is available for volume control; install via NuGet if needed
using System.Management; // For system info queries

namespace MinimalVoiceAgent;

/// <summary>
/// Standalone class with tool functions tailored for common tasks on a nerdy man's computer.
/// Focuses on local system operations like volume control, system info, timers, calculations, and reminders.
/// Best practices: Async methods, immutable data, explicit error handling, modern C# features (e.g., primary constructors).
/// Dependencies: NAudio for audio (volume), System.Management for hardware queries.
/// </summary>
public class ComputerToolFunctions
{
    private readonly JsonSerializerOptions _jsonOptions = new() { WriteIndented = true, DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull };
    private readonly Dictionary<DateTimeOffset, string> _localReminders = new(); // In-memory reminders
    private readonly List<System.Timers.Timer> _activeTimers = new(); // For managing timers

    /// <summary>
    /// Primary constructor: No external dependencies; uses system APIs directly.
    /// </summary>
    public ComputerToolFunctions()
    {
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

    [KernelFunction("adjust_system_volume")]
    [Description("Adjust the system volume, e.g., mute while watching TV or lower for a call.")]
    public virtual async Task<string> AdjustSystemVolumeAsync(
        [Description("Volume level (0-100; negative for mute)")] int level)
    {
        try
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                using var deviceEnumerator = new MMDeviceEnumerator();
                using var defaultDevice = deviceEnumerator.GetDefaultAudioEndpoint(DataFlow.Render, Role.Multimedia);

                if (level < 0)
                {
                    defaultDevice.AudioEndpointVolume.Mute = true;
                    Log.Information("System volume muted.");
                    return JsonSerializer.Serialize(new { status = "success", message = "System volume muted." }, _jsonOptions);
                }
                else
                {
                    level = Math.Clamp(level, 0, 100);
                    defaultDevice.AudioEndpointVolume.MasterVolumeLevelScalar = level / 100f;
                    defaultDevice.AudioEndpointVolume.Mute = false;
                    Log.Information("System volume set to {Level}%.", level);
                    return JsonSerializer.Serialize(new { status = "success", message = $"System volume set to {level}%." }, _jsonOptions);
                }
            }
            else
            {
                throw new PlatformNotSupportedException("Volume control supported on Windows only.");
            }

            await Task.CompletedTask;
        }
        catch (Exception ex)
        {
            Log.Error(ex, "Tool execution failed for adjust_system_volume: {Message}", ex.Message);
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
                foreach (var obj in searcher.Get())
                {
                    info["CPU Name"] = obj["Name"];
                    info["CPU Cores"] = obj["NumberOfCores"];
                }
            }

            if (infoType == "memory" || infoType == "all")
            {
                using var searcher = new ManagementObjectSearcher("select * from Win32_OperatingSystem");
                foreach (var obj in searcher.Get())
                {
                    info["Total Memory (MB)"] = Convert.ToUInt64(obj["TotalVisibleMemorySize"]) / 1024;
                    info["Free Memory (MB)"] = Convert.ToUInt64(obj["FreePhysicalMemory"]) / 1024;
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
                // TODO: Add alert, e.g., play sound or notify
                _activeTimers.Remove(timer);
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

    private static TimeSpan ParseTimeOffset(string dueTime)
    {
        // Simplified; expand with libraries like Humanizer or NodaTime for production
        if (dueTime.Contains("minutes"))
        {
            if (int.TryParse(dueTime.Split(' ')[1], out int minutes))
            {
                return TimeSpan.FromMinutes(minutes);
            }
        }
        // Add hours, days, etc.
        throw new ArgumentException($"Unable to parse due time: {dueTime}");
    }
}