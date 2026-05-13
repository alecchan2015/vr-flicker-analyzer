"""ADB 设备发现与信息查询。

只依赖系统 `adb` 二进制；不引入额外 Python 包。
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass, asdict
from typing import List, Optional


class ADBError(RuntimeError):
    pass


def _adb_path() -> str:
    p = shutil.which('adb')
    if not p:
        raise ADBError(
            "未找到 adb 可执行文件。macOS 请执行: brew install android-platform-tools"
        )
    return p


def _run(args: List[str], timeout: float = 10.0) -> str:
    proc = subprocess.run(
        [_adb_path(), *args],
        capture_output=True, text=True, timeout=timeout,
    )
    if proc.returncode != 0:
        raise ADBError(f"adb {' '.join(args)} 失败: {proc.stderr.strip() or proc.stdout.strip()}")
    return proc.stdout


@dataclass
class DeviceInfo:
    serial: str
    state: str                 # device / unauthorized / offline
    model: Optional[str] = None
    manufacturer: Optional[str] = None
    android_version: Optional[str] = None
    resolution: Optional[str] = None    # "1920x1080"
    density: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)


def list_devices() -> List[DeviceInfo]:
    """返回当前 adb 可见的设备列表（含基本信息）。"""
    out = _run(['devices', '-l'])
    devices: List[DeviceInfo] = []
    for line in out.splitlines()[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        serial, state = parts[0], parts[1]
        info = DeviceInfo(serial=serial, state=state)
        if state == 'device':
            try:
                info = get_device_info(serial)
            except ADBError:
                pass
        devices.append(info)
    return devices


def _getprop(serial: str, prop: str) -> str:
    return _run(['-s', serial, 'shell', 'getprop', prop]).strip()


def get_device_info(serial: str) -> DeviceInfo:
    """查询指定设备的详细信息。"""
    model = _getprop(serial, 'ro.product.model') or None
    manufacturer = _getprop(serial, 'ro.product.manufacturer') or None
    android_version = _getprop(serial, 'ro.build.version.release') or None

    resolution: Optional[str] = None
    try:
        wm = _run(['-s', serial, 'shell', 'wm', 'size']).strip()
        m = re.search(r'(\d+x\d+)', wm)
        if m:
            resolution = m.group(1)
    except ADBError:
        pass

    density: Optional[int] = None
    try:
        dm = _run(['-s', serial, 'shell', 'wm', 'density']).strip()
        m = re.search(r'(\d+)', dm)
        if m:
            density = int(m.group(1))
    except ADBError:
        pass

    return DeviceInfo(
        serial=serial,
        state='device',
        model=model,
        manufacturer=manufacturer,
        android_version=android_version,
        resolution=resolution,
        density=density,
    )
