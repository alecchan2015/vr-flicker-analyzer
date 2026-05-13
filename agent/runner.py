"""CLI 入口：抓视频 → 分析 → 终端打印 + 落盘。

用法:
    python -m agent list
    python -m agent capture --device <serial> --duration 10 --out ./report
    python -m agent analyze <video_path> --out ./report
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# 让 backend 包可被 import
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / 'backend'))

from agent.device_manager import list_devices, ADBError
from agent.capturer import Capturer, CaptureConfig, CaptureError
from backend.analyzer import analyze_video, SEVERITY_LEVELS


SEV_ANSI = {
    'Excellent': '\033[92m',  # green
    'Good':      '\033[36m',  # cyan
    'Moderate':  '\033[93m',  # yellow
    'Severe':    '\033[33m',  # orange-ish
    'Critical':  '\033[91m',  # red
}
RESET = '\033[0m'


def _color(level: str) -> str:
    return f"{SEV_ANSI.get(level, '')}{level}{RESET}"


def cmd_list(args):
    try:
        devices = list_devices()
    except ADBError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 2
    if not devices:
        print("未发现设备。请确认 USB 连接 + 开启 USB 调试。")
        return 0
    print(f"发现 {len(devices)} 台设备:")
    for d in devices:
        info = d.to_dict()
        print(f"  - {info['serial']:24s} {info['state']:12s} "
              f"{info.get('manufacturer') or '?'} {info.get('model') or '?'} "
              f"Android {info.get('android_version') or '?'} "
              f"{info.get('resolution') or '?'}")
    return 0


def _resolve_device(serial: str | None) -> str:
    devices = [d for d in list_devices() if d.state == 'device']
    if not devices:
        raise CaptureError("没有可用设备 (state=device)。")
    if serial:
        for d in devices:
            if d.serial == serial:
                return d.serial
        raise CaptureError(f"指定的 serial 不存在或未授权: {serial}")
    if len(devices) > 1:
        raise CaptureError(
            "检测到多台设备，请用 --device <serial> 显式指定。"
        )
    return devices[0].serial


def _print_report(result: dict):
    print()
    print("─" * 64)
    print(f"  设备:        {result['device']}")
    print(f"  帧数 / FPS:  {result['n_frames']} @ {result['fps']:.2f}")
    print(f"  双目视频:    {'是 (取右眼)' if result['is_stereo'] else '否'}")
    print("─" * 64)
    sev = result.get('severity', {})
    rows = [
        ('Percent Flicker (%)',  result['percent_flicker'],      sev.get('percent_flicker')),
        ('Edge Flicker (%)',     result['edge_percent_flicker'], sev.get('edge_percent_flicker')),
        ('HF Stability CV (%)',  result['hf_stability_cv'],      sev.get('hf_stability_cv')),
        ('Aliasing Ratio (%)',   result['aliasing_ratio_mean'],  sev.get('aliasing_ratio_mean')),
        ('Tearing Score',        result['tearing_score'],        sev.get('tearing_score')),
    ]
    for name, val, level in rows:
        print(f"  {name:<22s} {val:>10.3f}    {_color(level or '-')}")
    print("─" * 64)
    print(f"  撕裂帧占比:  {result['tear_frame_ratio']:.1f}%   "
          f"平均接缝幅值: {result['mean_seam_magnitude_px']:.2f} px")
    print(f"  综合评级:    {_color(result['overall_severity'])}  "
          f"(score={result['overall_score']:.1f}/100)")
    print("─" * 64)


def cmd_capture(args):
    try:
        serial = _resolve_device(args.device)
    except CaptureError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 2

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    video_path = out_dir / f'capture_{serial}_{ts}.mp4'

    cfg = CaptureConfig(
        serial=serial,
        output_path=str(video_path),
        duration_s=args.duration,
        max_size=args.max_size,
        bit_rate=args.bit_rate,
        fps=args.fps,
    )
    print(f"[1/2] 开始录制设备 {serial}，时长 {args.duration}s → {video_path}")
    cap = Capturer(cfg)
    try:
        cap.run_blocking()
    except CaptureError as e:
        print(f"[ERROR] 录制失败: {e}", file=sys.stderr)
        return 3
    print(f"      实际耗时 {cap.elapsed:.1f}s，文件 {video_path.stat().st_size/1e6:.1f} MB")

    print(f"[2/2] 开始分析...")
    return _analyze_and_print(str(video_path), args.name or serial, str(out_dir))


def cmd_analyze(args):
    if not os.path.exists(args.video):
        print(f"[ERROR] 视频不存在: {args.video}", file=sys.stderr)
        return 2
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    return _analyze_and_print(args.video, args.name or 'Device', str(out_dir))


def _analyze_and_print(video_path: str, name: str, out_dir: str) -> int:
    last_p = [-5]
    def cb(p):
        pct = int(p * 100)
        if pct - last_p[0] >= 5:
            sys.stdout.write(f"\r      进度 {pct:3d}%")
            sys.stdout.flush()
            last_p[0] = pct
    result = analyze_video(video_path, out_dir, device_name=name,
                           max_frames=300, progress_callback=cb)
    sys.stdout.write("\r" + " " * 20 + "\r")
    _print_report(result)
    print(f"\n  图表:  {result.get('chart_path')}")
    print(f"  JSON:  {result.get('json_path')}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='agent',
        description='VR Flicker Analyzer - 本地录制 + 分析 Agent',
    )
    sub = p.add_subparsers(dest='cmd', required=True)

    sp = sub.add_parser('list', help='列出已连接的 ADB 设备')
    sp.set_defaults(func=cmd_list)

    sp = sub.add_parser('capture', help='抓取设备屏幕并自动分析')
    sp.add_argument('--device', help='设备 serial（多设备时必填）')
    sp.add_argument('--duration', type=float, default=10.0, help='录制时长(秒)，默认 10')
    sp.add_argument('--max-size', type=int, default=1280, help='scrcpy --max-size，默认 1280')
    sp.add_argument('--bit-rate', default='8M', help='scrcpy --video-bit-rate，默认 8M')
    sp.add_argument('--fps', type=int, default=None, help='scrcpy --max-fps，默认不限')
    sp.add_argument('--name', help='报告中显示的设备名')
    sp.add_argument('--out', default='./report', help='输出目录，默认 ./report')
    sp.set_defaults(func=cmd_capture)

    sp = sub.add_parser('analyze', help='分析已有的视频文件')
    sp.add_argument('video', help='视频路径')
    sp.add_argument('--name', help='报告中显示的设备名')
    sp.add_argument('--out', default='./report', help='输出目录，默认 ./report')
    sp.set_defaults(func=cmd_analyze)

    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    raise SystemExit(main())
