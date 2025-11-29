#pragma once
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include <mutex>
#include <unordered_map>

struct BrightnessRenderInfo {
    uint8_t alpha{ std::numeric_limits<decltype(alpha)>::max() };
    std::wstring label;
};

BrightnessRenderInfo GetBrightnessRenderInfoFromPercentage(const float brightnessPercentage) {
    const float MinPercentage = 0.1f; // 10%
    const float MaxPercentage = 1.0f; // 100%

    // Clamp percentage for safety
    const auto clampedPercentage = std::max(MinPercentage, std::min(MaxPercentage, brightnessPercentage));
    const auto alpha = static_cast<uint8_t>(clampedPercentage * 255.0f + 0.5f);  // adds the 0.5f to round to nearest
    const auto displayPercentage = static_cast<int>(clampedPercentage * 100.0f + 0.5f); // Convert to 0-100%, round
    const std::wstring label = std::to_wstring(displayPercentage) + L"%";  // format
    return { alpha, label };
}


static HWND overlayWindow = nullptr;
static std::mutex overlayMutex;
static BrightnessRenderInfo currentRenderInfo;
static std::thread windowThread; // message loop thread
static DWORD windowThreadId = 0;  // Store background thread ID for targeted PostThreadMessage

RECT GetPrimaryMonitorRect() {
    RECT rc;
    //SystemParametersInfo(SPI_GETWORKAREA, 0, &rc, 0);
    // Or use full screen:
    GetWindowRect(GetDesktopWindow(), &rc);
    return rc;
}

LRESULT CALLBACK OverlayWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    return DefWindowProc(hwnd, msg, wParam, lParam);
}

HWND CreateOverlayForMonitor(const RECT& monitorRect, HINSTANCE hInstance, uint8_t alpha, std::wstring_view labelText = L"")
{
    static const wchar_t* className = L"BlueLightOverlayClass";
    static bool classRegistered = false;

    if (!classRegistered) {
        WNDCLASS wc = {};
        wc.lpfnWndProc = OverlayWndProc;
        wc.hInstance = hInstance;
        wc.lpszClassName = className;
        wc.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);  // Set background to black
        RegisterClass(&wc);
        classRegistered = true;
    }

    int width = monitorRect.right - monitorRect.left;
    int height = monitorRect.bottom - monitorRect.top;

    HWND hwnd = CreateWindowEx(
        WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST,
        className,
        L"",
        0,
        monitorRect.left, monitorRect.top, width, height,
        nullptr, nullptr, hInstance, nullptr);

    if (hwnd) {
        SetLayeredWindowAttributes(hwnd, 0, alpha, LWA_ALPHA);
        ShowWindow(hwnd, SW_SHOW);
    }

    if (!labelText.empty()) {
        std::thread([hwnd, monitorRect, alpha, labelText]() {
            std::this_thread::sleep_for(std::chrono::seconds(2));

            HDC hdcScreen = GetDC(NULL);
            HDC hdcMem = CreateCompatibleDC(hdcScreen);

            int width = monitorRect.right - monitorRect.left;
            int height = monitorRect.bottom - monitorRect.top;

            BITMAPINFO bmi = {};
            bmi.bmiHeader.biSize = sizeof(bmi.bmiHeader);
            bmi.bmiHeader.biWidth = width;
            bmi.bmiHeader.biHeight = -height;
            bmi.bmiHeader.biPlanes = 1;
            bmi.bmiHeader.biBitCount = 32;
            bmi.bmiHeader.biCompression = BI_RGB;

            void* pvBits = nullptr;
            HBITMAP hBitmap = CreateDIBSection(hdcMem, &bmi, DIB_RGB_COLORS, &pvBits, nullptr, 0);
            SelectObject(hdcMem, hBitmap);

            UINT32* pixels = reinterpret_cast<UINT32*>(pvBits);
            UINT32 bgra = (alpha << 24) | (0 << 16) | (0 << 8) | 0;
            for (int i = 0; i < width * height; ++i) {
                pixels[i] = bgra;
            }

            SIZE size = { width, height };
            POINT src = { 0, 0 };
            BLENDFUNCTION blend = { AC_SRC_OVER, 0, 255, AC_SRC_ALPHA };

            UpdateLayeredWindow(hwnd, hdcScreen, nullptr, &size, hdcMem, &src, 0, &blend, ULW_ALPHA);
            ReleaseDC(NULL, hdcScreen);
            DeleteObject(hBitmap);
            DeleteDC(hdcMem);
            }).detach();
    }

    return hwnd;

}

// Helper function for the thread's main loop
void WindowMessageLoop(HINSTANCE hInstance, uint8_t alpha, std::wstring_view labelText) {
    RECT monitorRect = GetPrimaryMonitorRect();
    {
        std::scoped_lock lock(overlayMutex);
        overlayWindow = CreateOverlayForMonitor(monitorRect, hInstance, alpha, labelText);
    }

    if (overlayWindow) {
        // Run message loop to keep window responsive
        MSG msg;
        while (GetMessage(&msg, nullptr, 0, 0)) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
    }
}

// 0.0f - 1.0f
void SetSingleMonitorOverlayBrightness(const float brightnessPercent) {
    std::unique_lock<std::mutex> lock(overlayMutex);

    // Destroy previous overlay if exists
    if (overlayWindow) {
        DestroyWindow(overlayWindow);
        overlayWindow = nullptr;
    }
    if (windowThread.joinable()) {
        if (windowThreadId != 0) {
            PostThreadMessage(windowThreadId, WM_QUIT, 0, 0);  // Post to background thread's queue
        }
        lock.unlock();  // Unlock before join to avoid deadlock
        windowThread.join();
        lock.lock();  // Relock if needed
        windowThreadId = 0;
    }

    currentRenderInfo = GetBrightnessRenderInfoFromPercentage(brightnessPercent);

    if (currentRenderInfo.alpha == std::numeric_limits<decltype(currentRenderInfo.alpha)>::max())
        return;

    lock.unlock();  // Unlock before spawning new thread

    // Spawn new thread for window creation and message loop
    windowThread = std::thread([alpha = currentRenderInfo.alpha, label = currentRenderInfo.label]() {
        windowThreadId = GetCurrentThreadId();  // Capture thread ID inside the thread
        WindowMessageLoop(GetModuleHandle(nullptr), alpha, label);
        });
}

void DestroySingleMonitorOverlay() {
    std::unique_lock<std::mutex> lock(overlayMutex);

    if (overlayWindow) {
        DestroyWindow(overlayWindow);
        overlayWindow = nullptr;
    }

    if (windowThread.joinable()) {
        if (windowThreadId != 0) {
            PostThreadMessage(windowThreadId, WM_QUIT, 0, 0);  // Post to background thread's queue
        }
        lock.unlock();  // Unlock before join to avoid deadlock
        windowThread.join();
        windowThreadId = 0;
    }

    // Reset state
    currentRenderInfo = BrightnessRenderInfo{};
}

float GetSingleMonitorOverlayBrightness() {
    std::scoped_lock lock(overlayMutex);
    if (currentRenderInfo.alpha == std::numeric_limits<decltype(currentRenderInfo.alpha)>::max())
        return 1.0f; // 100%
    return static_cast<float>(currentRenderInfo.alpha) / 255.0f;
}