"""VR Flicker Analyzer - Local Capture Agent.

通过 adb + scrcpy 直接从连接的设备录屏，自动化跑完整套闪烁分析。
"""

from .device_manager import list_devices, get_device_info, DeviceInfo
from .capturer import Capturer, CaptureError

__all__ = ['list_devices', 'get_device_info', 'DeviceInfo', 'Capturer', 'CaptureError']
