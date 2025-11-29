#pragma once
#include "Win32Overlay.h"
using namespace System;

namespace ScreenyDimmery {
    public ref class Dimmer
    {
    public:
        static void DestroySingleMonitorOverlay()
        {
			::DestroySingleMonitorOverlay();
        }
        static void SetSingleMonitorOverlayBrightness(float brightnessPercent)
        {
            ::SetSingleMonitorOverlayBrightness(brightnessPercent);
		}
        static float GetCurrentBrightnessPercent()
        {
			return ::GetSingleMonitorOverlayBrightness();
		}

        // Add more if needed, e.g., GetCurrentState()
    };
}
