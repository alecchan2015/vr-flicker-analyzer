"""scrcpy 录制封装。

调用系统 `scrcpy --record` 子进程把设备屏幕保存为 mp4。
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


class CaptureError(RuntimeError):
    pass


@dataclass
class CaptureConfig:
    serial: str
    output_path: str
    duration_s: float = 10.0
    max_size: Optional[int] = 1280       # scrcpy --max-size，限制长边
    bit_rate: Optional[str] = '8M'        # scrcpy --video-bit-rate
    fps: Optional[int] = None             # scrcpy --max-fps；None 表示不限制
    no_audio: bool = True
    no_display: bool = True               # scrcpy --no-window（仅录制不显示窗口）


def _scrcpy_path() -> str:
    p = shutil.which('scrcpy')
    if not p:
        raise CaptureError("未找到 scrcpy。macOS 请执行: brew install scrcpy")
    return p


class Capturer:
    """单次录制任务。线程安全地启动 / 停止 / 等待 scrcpy 子进程。"""

    def __init__(self, config: CaptureConfig):
        self.config = config
        self._proc: Optional[subprocess.Popen] = None
        self._stop_evt = threading.Event()
        self._start_ts: float = 0.0
        self._end_ts: float = 0.0

    def _build_cmd(self) -> list:
        c = self.config
        cmd = [_scrcpy_path(), '-s', c.serial, '--record', c.output_path]
        if c.max_size:
            cmd += ['--max-size', str(c.max_size)]
        if c.bit_rate:
            cmd += ['--video-bit-rate', c.bit_rate]
        if c.fps:
            cmd += ['--max-fps', str(c.fps)]
        if c.no_audio:
            cmd += ['--no-audio']
        if c.no_display:
            cmd += ['--no-window']
        return cmd

    def start(self) -> None:
        if self._proc is not None:
            raise CaptureError("录制已在进行中")
        out_dir = os.path.dirname(self.config.output_path) or '.'
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        cmd = self._build_cmd()
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                start_new_session=True,
            )
        except FileNotFoundError as e:
            raise CaptureError(f"启动 scrcpy 失败: {e}") from e
        self._start_ts = time.time()

    def wait_for_duration(self) -> None:
        """阻塞等到达预设时长或被外部 stop()。"""
        if self._proc is None:
            raise CaptureError("录制未启动")
        deadline = self._start_ts + self.config.duration_s
        while time.time() < deadline:
            if self._stop_evt.is_set():
                break
            if self._proc.poll() is not None:
                stderr = (self._proc.stderr.read() or b'').decode('utf-8', 'ignore')
                raise CaptureError(f"scrcpy 提前退出: {stderr.strip()}")
            time.sleep(0.1)
        self.stop()

    def stop(self) -> None:
        if self._proc is None or self._proc.poll() is not None:
            self._end_ts = time.time()
            return
        self._stop_evt.set()
        try:
            os.killpg(os.getpgid(self._proc.pid), signal.SIGINT)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            self._proc.wait(timeout=3)
        self._end_ts = time.time()

    @property
    def elapsed(self) -> float:
        if self._end_ts:
            return self._end_ts - self._start_ts
        if self._start_ts:
            return time.time() - self._start_ts
        return 0.0

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def run_blocking(self) -> str:
        """启动 → 录满时长 → 停止；返回输出文件路径。"""
        self.start()
        self.wait_for_duration()
        if not os.path.exists(self.config.output_path) or os.path.getsize(self.config.output_path) < 1024:
            raise CaptureError(f"录制文件无效: {self.config.output_path}")
        return self.config.output_path
