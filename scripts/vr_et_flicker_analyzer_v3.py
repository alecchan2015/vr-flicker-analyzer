#!/usr/bin/env python3
"""
VR ET-Center Panel Flicker Analyzer v3.0
=========================================
在 ET 中心热力图定位的基础上，新增：
  - 文字边缘闪烁评估 (Edge Flicker)
  - 高频纹理细节稳定性评估 (HF Texture Stability)

用法:
  python3 vr_et_flicker_analyzer_v3.py <video_path> [--output-dir <dir>]
"""

import cv2
import numpy as np
import json
import argparse
import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import signal as scipy_signal

# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def calc_percent_flicker(lum_series):
    """Percent Flicker = (max - min) / (max + min) * 100"""
    mx, mn = np.max(lum_series), np.min(lum_series)
    if mx + mn < 1e-6:
        return 0.0
    return float((mx - mn) / (mx + mn) * 100)

def calc_flicker_index(lum_series):
    """IES Flicker Index = 正半周期面积 / 总面积"""
    mean_val = np.mean(lum_series)
    above = np.sum(np.maximum(lum_series - mean_val, 0))
    total = np.sum(np.abs(lum_series - mean_val))
    if total < 1e-6:
        return 0.0
    return float(above / total)

def calc_dominant_freq(lum_series, fps):
    """FFT 主频 (Hz)"""
    n = len(lum_series)
    if n < 4:
        return 0.0
    fft_vals = np.abs(np.fft.rfft(lum_series - np.mean(lum_series)))
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    fft_vals[0] = 0  # 去除直流分量
    idx = np.argmax(fft_vals)
    return float(freqs[idx])

def calc_fft_peak_ratio(lum_series):
    """FFT 主频能量占比"""
    n = len(lum_series)
    if n < 4:
        return 0.0
    fft_vals = np.abs(np.fft.rfft(lum_series - np.mean(lum_series)))
    fft_vals[0] = 0
    total = np.sum(fft_vals)
    if total < 1e-6:
        return 0.0
    return float(np.max(fft_vals) / total)

def calc_temporal_contrast(lum_series):
    """时域对比度 = σ / μ"""
    mu = np.mean(lum_series)
    if mu < 1e-6:
        return 0.0
    return float(np.std(lum_series) / mu)

def calc_frame_diff_mean(lum_series):
    """帧间差分均值"""
    diffs = np.abs(np.diff(lum_series))
    return float(np.mean(diffs)) if len(diffs) > 0 else 0.0

def calc_flicker_score(pf, fi, tc, fft_ratio, fd_mean):
    """综合闪烁评分 (0-100, 越低越好)"""
    s = (pf / 100 * 40) + (fi * 20) + (tc * 1000 * 15) + (fft_ratio * 15) + (min(fd_mean / 50, 1) * 10)
    return float(min(s, 100))

def severity_label(score):
    if score < 10:
        return "Excellent"
    elif score < 25:
        return "Good"
    elif score < 45:
        return "Moderate"
    elif score < 65:
        return "Severe"
    else:
        return "Critical"

def extract_roi_lum(frame_gray, x, y, w, h):
    """提取矩形 ROI 的平均亮度"""
    roi = frame_gray[y:y+h, x:x+w]
    return float(np.mean(roi)) if roi.size > 0 else 0.0

def extract_circle_lum(frame_gray, cx, cy, r):
    """提取圆形 ROI 的平均亮度"""
    h, w = frame_gray.shape
    Y, X = np.ogrid[:h, :w]
    mask = (X - cx)**2 + (Y - cy)**2 <= r**2
    vals = frame_gray[mask]
    return float(np.mean(vals)) if vals.size > 0 else 0.0

def extract_ring_lum(frame_gray, cx, cy, r_inner, r_outer):
    """提取环形 ROI 的平均亮度"""
    h, w = frame_gray.shape
    Y, X = np.ogrid[:h, :w]
    dist2 = (X - cx)**2 + (Y - cy)**2
    mask = (dist2 >= r_inner**2) & (dist2 <= r_outer**2)
    vals = frame_gray[mask]
    return float(np.mean(vals)) if vals.size > 0 else 0.0

# ─────────────────────────────────────────────
# 新增模块 1：文字边缘闪烁评估
# ─────────────────────────────────────────────

def extract_edge_lum(frame_gray, x, y, w, h, canny_low=30, canny_high=100, dilate_px=2):
    """
    提取 ROI 内文字边缘像素的平均亮度。
    使用 Canny 边缘检测 + 膨胀，只统计边缘像素的亮度均值。
    """
    roi = frame_gray[y:y+h, x:x+w]
    if roi.size == 0:
        return 0.0
    edges = cv2.Canny(roi, canny_low, canny_high)
    if dilate_px > 0:
        kernel = np.ones((dilate_px * 2 + 1, dilate_px * 2 + 1), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
    edge_pixels = roi[edges > 0]
    return float(np.mean(edge_pixels)) if edge_pixels.size > 0 else 0.0

def extract_edge_lum_circle(frame_gray, cx, cy, r, canny_low=30, canny_high=100, dilate_px=2):
    """提取圆形 ROI 内文字边缘像素的平均亮度"""
    h, w = frame_gray.shape
    # 裁出 bounding box
    x1 = max(0, cx - r)
    y1 = max(0, cy - r)
    x2 = min(w, cx + r)
    y2 = min(h, cy + r)
    roi = frame_gray[y1:y2, x1:x2].copy()
    if roi.size == 0:
        return 0.0
    # 圆形 mask
    roi_h, roi_w = roi.shape
    Y, X = np.ogrid[:roi_h, :roi_w]
    circle_mask = (X - (cx - x1))**2 + (Y - (cy - y1))**2 <= r**2
    # 边缘检测
    edges = cv2.Canny(roi, canny_low, canny_high)
    if dilate_px > 0:
        kernel = np.ones((dilate_px * 2 + 1, dilate_px * 2 + 1), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
    combined_mask = (edges > 0) & circle_mask
    edge_pixels = roi[combined_mask]
    return float(np.mean(edge_pixels)) if edge_pixels.size > 0 else 0.0

def calc_edge_flicker_score(pf, tc, fd_mean):
    """
    文字边缘闪烁专项评分 (0-100)。
    评分设计原则：
      - Percent Flicker 是主要驱动因子（占 60%）
      - Temporal Contrast 辅助（占 30%），阈值调整为 0.02（即 TC>2% 才开始显著影响）
      - Frame Diff 辅助（占 10%），阈值调整为 5.0（降低对编码噪声的敏感度）
    参考分段：
      PF < 1%   且 TC < 0.005  -> 通常 < 10 （Excellent）
      PF 1-5%   且 TC 0.005-0.02 -> 通常 10-30 （Good/Moderate）
      PF > 10%  或 TC > 0.05    -> 通常 > 50 （Severe/Critical）
    """
    s = (min(pf, 100) / 100 * 60) + (min(tc / 0.05, 1) * 30) + (min(fd_mean / 5.0, 1) * 10)
    return float(min(s, 100))

# ─────────────────────────────────────────────
# 新增模块 2：高频纹理细节稳定性评估
# ─────────────────────────────────────────────

def extract_hf_energy(frame_gray, x, y, w, h):
    """
    提取 ROI 内的高频纹理能量 (Laplacian 方差)。
    Laplacian 方差越大，说明纹理越丰富/越锐利。
    时域上该值的波动反映了纹理细节的稳定性。
    """
    roi = frame_gray[y:y+h, x:x+w]
    if roi.size == 0:
        return 0.0
    lap = cv2.Laplacian(roi, cv2.CV_64F)
    return float(np.var(lap))

def extract_hf_energy_circle(frame_gray, cx, cy, r):
    """提取圆形 ROI 内的高频纹理能量"""
    h, w = frame_gray.shape
    x1 = max(0, cx - r)
    y1 = max(0, cy - r)
    x2 = min(w, cx + r)
    y2 = min(h, cy + r)
    roi = frame_gray[y1:y2, x1:x2].copy()
    if roi.size == 0:
        return 0.0
    roi_h, roi_w = roi.shape
    Y, X = np.ogrid[:roi_h, :roi_w]
    circle_mask = (X - (cx - x1))**2 + (Y - (cy - y1))**2 <= r**2
    # 对圆外区域填充均值，避免边缘影响 Laplacian
    mean_val = int(np.mean(roi[circle_mask])) if np.any(circle_mask) else 128
    roi_masked = roi.copy()
    roi_masked[~circle_mask] = mean_val
    lap = cv2.Laplacian(roi_masked, cv2.CV_64F)
    return float(np.var(lap[circle_mask]))

def calc_hf_stability_score(hf_series):
    """
    高频纹理稳定性评分 (0-100, 越低越好)。
    使用 HF 能量时序的变异系数 (CV = σ/μ) 来量化纹理细节的时域抖动。
    CV 越大，说明纹理细节在帧间抖动越剧烈（如 TAA 鬼影、渲染噪点）。
    """
    mu = np.mean(hf_series)
    if mu < 1e-6:
        return 0.0
    cv = np.std(hf_series) / mu
    # 归一化：CV=0 -> 0分, CV=0.1 -> ~50分, CV>=0.2 -> 100分
    score = min(cv / 0.2 * 100, 100)
    return float(score)

def calc_hf_stability_label(score):
    if score < 5:
        return "极稳定 (Excellent)"
    elif score < 15:
        return "稳定 (Good)"
    elif score < 35:
        return "轻微抖动 (Moderate)"
    elif score < 60:
        return "明显抖动 (Severe)"
    else:
        return "严重抖动 (Critical)"

# ─────────────────────────────────────────────
# 面板检测
# ─────────────────────────────────────────────

def detect_panel(frame_gray, frame_w, frame_h):
    """通过亮度梯度自动检测面板边界"""
    # 行列均值
    row_mean = np.mean(frame_gray, axis=1).astype(float)
    col_mean = np.mean(frame_gray, axis=0).astype(float)

    # 平滑
    row_smooth = np.convolve(row_mean, np.ones(20)/20, mode='same')
    col_smooth = np.convolve(col_mean, np.ones(20)/20, mode='same')

    # 梯度
    row_grad = np.abs(np.gradient(row_smooth))
    col_grad = np.abs(np.gradient(col_smooth))

    # 边界候选（排除最外5%）
    margin_r = int(frame_h * 0.05)
    margin_c = int(frame_w * 0.05)

    def find_boundary(grad, margin, size, direction='top'):
        search = grad[margin:size - margin]
        offset = margin
        if direction in ('top', 'left'):
            idx = np.argmax(search[:len(search)//2]) + offset
        else:
            idx = np.argmax(search[len(search)//2:]) + len(search)//2 + offset
        return int(idx)

    top = find_boundary(row_grad, margin_r, frame_h, 'top')
    bottom = find_boundary(row_grad, margin_r, frame_h, 'bottom')
    left = find_boundary(col_grad, margin_c, frame_w, 'left')
    right = find_boundary(col_grad, margin_c, frame_w, 'right')

    # 合理性校验
    if bottom - top < frame_h * 0.2:
        top = int(frame_h * 0.1)
        bottom = int(frame_h * 0.9)
    if right - left < frame_w * 0.2:
        left = int(frame_w * 0.1)
        right = int(frame_w * 0.9)

    return left, top, right - left, bottom - top

# ─────────────────────────────────────────────
# ET 中心定位
# ─────────────────────────────────────────────

def build_contrast_heatmap(video_path, panel, max_frames=60, step=5):
    """Pass 1: 累积局部对比度热力图，定位 ET 中心"""
    px, py, pw, ph = panel
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    heatmap = np.zeros((ph, pw), dtype=np.float64)
    count = 0
    frame_idx = 0
    while cap.isOpened() and count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            roi = gray[py:py+ph, px:px+pw]
            lap = cv2.Laplacian(roi.astype(np.float64), cv2.CV_64F)
            heatmap += np.abs(lap)
            count += 1
        frame_idx += 1
    cap.release()

    # 高斯平滑
    heatmap_smooth = cv2.GaussianBlur(heatmap.astype(np.float32), (51, 51), 0)
    # ET 中心 = 热力图最大值位置（在面板坐标系内）
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(heatmap_smooth)
    et_cx = px + max_loc[0]
    et_cy = py + max_loc[1]
    return et_cx, et_cy, heatmap_smooth

# ─────────────────────────────────────────────
# 主分析流程
# ─────────────────────────────────────────────

def analyze(video_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    print(f"视频: {frame_w}x{frame_h}, {total_frames}帧, {fps:.2f}FPS")

    # ── Step 1: 提取中间帧，检测面板 ──
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
    ret, mid_frame = cap.read()
    cap.release()
    if not ret:
        print("ERROR: 无法读取视频帧")
        return

    mid_gray = cv2.cvtColor(mid_frame, cv2.COLOR_BGR2GRAY)
    px, py, pw, ph = detect_panel(mid_gray, frame_w, frame_h)
    print(f"[Step 1] 面板: x={px}, y={py}, w={pw}, h={ph}")

    # ── Step 2: 建热力图，定位 ET 中心 ──
    print("[Step 2] 建立对比度热力图，定位 ET 中心...")
    step = max(1, total_frames // 60)
    et_cx, et_cy, heatmap = build_contrast_heatmap(video_path, (px, py, pw, ph), max_frames=60, step=step)
    r_hd = int(min(pw, ph) * 0.22)      # 高清区半径 ≈ 面板短边 22%
    r_peri = int(min(pw, ph) * 0.48)    # 余光区外径 ≈ 面板短边 48%
    print(f"[Step 2] ET 中心: ({et_cx}, {et_cy}), HD半径: {r_hd}px, 余光外径: {r_peri}px")

    # ── Step 3: Pass 2 流式读帧，计算所有指标 ──
    print("[Step 3] 流式计算所有指标...")

    # 亮度时序
    lum_panel, lum_hd, lum_peri = [], [], []
    # 文字边缘亮度时序
    edge_lum_panel, edge_lum_hd, edge_lum_peri = [], [], []
    # 高频纹理能量时序
    hf_panel, hf_hd, hf_peri = [], [], []

    cap = cv2.VideoCapture(video_path)
    fi = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 整体亮度
        lum_panel.append(extract_roi_lum(gray, px, py, pw, ph))
        lum_hd.append(extract_circle_lum(gray, et_cx, et_cy, r_hd))
        lum_peri.append(extract_ring_lum(gray, et_cx, et_cy, r_hd, r_peri))

        # 文字边缘亮度
        edge_lum_panel.append(extract_edge_lum(gray, px, py, pw, ph))
        edge_lum_hd.append(extract_edge_lum_circle(gray, et_cx, et_cy, r_hd))
        # 余光区边缘：用面板边缘减去高清区边缘近似
        edge_lum_peri.append(extract_edge_lum(gray, px, py, pw, ph, canny_low=20, canny_high=80))

        # 高频纹理能量
        hf_panel.append(extract_hf_energy(gray, px, py, pw, ph))
        hf_hd.append(extract_hf_energy_circle(gray, et_cx, et_cy, r_hd))
        hf_peri.append(extract_hf_energy(gray, px, py, pw, ph))  # 面板整体近似余光

        fi += 1
        if fi % 30 == 0:
            print(f"  {fi}/{total_frames}")
    cap.release()
    print("  完成")

    lum_panel = np.array(lum_panel)
    lum_hd = np.array(lum_hd)
    lum_peri = np.array(lum_peri)
    edge_lum_panel = np.array(edge_lum_panel)
    edge_lum_hd = np.array(edge_lum_hd)
    edge_lum_peri = np.array(edge_lum_peri)
    hf_panel = np.array(hf_panel)
    hf_hd = np.array(hf_hd)
    hf_peri = np.array(hf_peri)

    # ── Step 4: 计算所有指标 ──
    def compute_zone_metrics(lum_series, edge_series, hf_series, label):
        pf = calc_percent_flicker(lum_series)
        fi_val = calc_flicker_index(lum_series)
        tc = calc_temporal_contrast(lum_series)
        fft_r = calc_fft_peak_ratio(lum_series)
        fd = calc_frame_diff_mean(lum_series)
        dom_freq = calc_dominant_freq(lum_series, fps)
        score = calc_flicker_score(pf, fi_val, tc, fft_r, fd)

        # 文字边缘闪烁
        edge_pf = calc_percent_flicker(edge_series)
        edge_tc = calc_temporal_contrast(edge_series)
        edge_fd = calc_frame_diff_mean(edge_series)
        edge_score = calc_edge_flicker_score(edge_pf, edge_tc, edge_fd)

        # 高频纹理稳定性
        hf_score = calc_hf_stability_score(hf_series)
        hf_label = calc_hf_stability_label(hf_score)

        return {
            "label": label,
            # 整体亮度闪烁
            "percent_flicker": round(pf, 4),
            "flicker_index": round(fi_val, 6),
            "temporal_contrast": round(tc, 6),
            "fft_peak_ratio": round(fft_r, 6),
            "fft_dominant_freq_hz": round(dom_freq, 2),
            "frame_diff_mean": round(fd, 4),
            "flicker_score": round(score, 2),
            "severity": severity_label(score),
            "mean_luminance": round(float(np.mean(lum_series)), 2),
            # 文字边缘闪烁
            "edge_percent_flicker": round(edge_pf, 4),
            "edge_temporal_contrast": round(edge_tc, 6),
            "edge_frame_diff_mean": round(edge_fd, 4),
            "edge_flicker_score": round(edge_score, 2),
            "edge_severity": severity_label(edge_score),
            # 高频纹理稳定性
            "hf_stability_score": round(hf_score, 2),
            "hf_stability_label": hf_label,
        }

    results = {
        "panel_full": compute_zone_metrics(lum_panel, edge_lum_panel, hf_panel, "面板整体"),
        "hd_zone": compute_zone_metrics(lum_hd, edge_lum_hd, hf_hd, "高清区（ET中心圆形）"),
        "peripheral": compute_zone_metrics(lum_peri, edge_lum_peri, hf_peri, "余光区（ET中心环形）"),
    }

    # ── Step 5: 保存 JSON ──
    json_path = output_dir / "et_flicker_metrics_v3.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[Step 5] JSON 已保存: {json_path}")

    # ── Step 6: 生成标注图 ──
    annotation = mid_frame.copy()
    # 面板框
    cv2.rectangle(annotation, (px, py), (px+pw, py+ph), (0, 165, 255), 6)
    cv2.putText(annotation, "Panel", (px+10, py+60), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 165, 255), 4)
    # 高清区（红色圆形）
    cv2.circle(annotation, (et_cx, et_cy), r_hd, (0, 0, 255), 5)
    cv2.putText(annotation, f"HD Zone (r={r_hd}px)", (et_cx - r_hd, et_cy - r_hd - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    # 余光区（绿色圆形）
    cv2.circle(annotation, (et_cx, et_cy), r_peri, (0, 255, 0), 4)
    # ET 中心
    cv2.drawMarker(annotation, (et_cx, et_cy), (255, 255, 255), cv2.MARKER_CROSS, 40, 4)
    cv2.putText(annotation, "ET Center", (et_cx + 20, et_cy + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    # 缩放保存
    scale = min(1.0, 1200 / max(frame_w, frame_h))
    ann_small = cv2.resize(annotation, (int(frame_w * scale), int(frame_h * scale)))
    ann_path = output_dir / "et_zone_annotation.png"
    cv2.imwrite(str(ann_path), ann_small)
    print(f"[Step 6] 标注图已保存: {ann_path}")

    # ── Step 7: 生成综合图表 ──
    _generate_chart(results, lum_panel, lum_hd, lum_peri,
                    edge_lum_hd, hf_hd, fps, output_dir)

    # ── Step 8: 打印汇总 ──
    print("\n" + "="*100)
    print(f"{'区域':<20} {'综合评分':>8} {'评级':>12} {'PctFlicker':>12} {'边缘PctFlicker':>16} {'边缘评分':>10} {'HF稳定性':>10} {'HF评级':>20}")
    print("-"*100)
    for k, v in results.items():
        print(f"{v['label']:<20} {v['flicker_score']:>8.2f} {v['severity']:>12} "
              f"{v['percent_flicker']:>11.2f}% {v['edge_percent_flicker']:>15.2f}% "
              f"{v['edge_flicker_score']:>10.2f} {v['hf_stability_score']:>10.2f} "
              f"{v['hf_stability_label']:>20}")
    print("="*100)

    return results


def _generate_chart(results, lum_panel, lum_hd, lum_peri, edge_lum_hd, hf_hd, fps, output_dir):
    """生成 6 面板综合图表"""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('VR Panel Flicker Analysis v3.0\n(ET-Center Based)', fontsize=16, fontweight='bold')

    t = np.arange(len(lum_panel)) / fps

    # 1. 亮度时序
    ax = axes[0, 0]
    ax.plot(t, lum_panel, 'b-', alpha=0.5, linewidth=0.8, label='Panel')
    ax.plot(t, lum_hd, 'r-', linewidth=1.2, label='HD Zone')
    ax.plot(t, lum_peri, 'g-', alpha=0.7, linewidth=0.8, label='Peripheral')
    ax.set_title('Luminance Time Series')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Mean Luminance')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. 综合评分对比柱状图
    ax = axes[0, 1]
    zones = ['Panel', 'HD Zone', 'Peripheral']
    scores = [results['panel_full']['flicker_score'],
              results['hd_zone']['flicker_score'],
              results['peripheral']['flicker_score']]
    colors = ['steelblue', 'crimson', 'seagreen']
    bars = ax.bar(zones, scores, color=colors, alpha=0.8)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{score:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_title('Overall Flicker Score (lower is better)')
    ax.set_ylabel('Score (0-100)')
    ax.set_ylim(0, max(scores) * 1.4 + 5)
    ax.grid(True, alpha=0.3, axis='y')

    # 3. 文字边缘闪烁评分对比
    ax = axes[0, 2]
    edge_scores = [results['panel_full']['edge_flicker_score'],
                   results['hd_zone']['edge_flicker_score'],
                   results['peripheral']['edge_flicker_score']]
    edge_pf = [results['panel_full']['edge_percent_flicker'],
               results['hd_zone']['edge_percent_flicker'],
               results['peripheral']['edge_percent_flicker']]
    x = np.arange(len(zones))
    width = 0.35
    bars1 = ax.bar(x - width/2, edge_scores, width, label='Edge Flicker Score', color=['steelblue', 'crimson', 'seagreen'], alpha=0.8)
    ax2_twin = ax.twinx()
    ax2_twin.bar(x + width/2, edge_pf, width, label='Edge Pct Flicker (%)', color=['lightblue', 'lightsalmon', 'lightgreen'], alpha=0.8)
    ax.set_title('Text Edge Flicker Analysis')
    ax.set_ylabel('Edge Flicker Score')
    ax2_twin.set_ylabel('Edge Percent Flicker (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(zones)
    ax.legend(loc='upper left', fontsize=8)
    ax2_twin.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # 4. 文字边缘亮度时序
    ax = axes[1, 0]
    ax.plot(t, edge_lum_hd, 'r-', linewidth=1.0, label='HD Zone Edge Lum')
    ax.set_title('Text Edge Luminance Time Series (HD Zone)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Edge Mean Luminance')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 5. 高频纹理能量时序
    ax = axes[1, 1]
    hf_norm = (hf_hd - np.min(hf_hd)) / (np.max(hf_hd) - np.min(hf_hd) + 1e-6)
    ax.plot(t, hf_norm, 'purple', linewidth=1.0, label='HF Energy (normalized)')
    ax.axhline(y=np.mean(hf_norm), color='orange', linestyle='--', linewidth=1.5, label=f'Mean')
    ax.fill_between(t, hf_norm, np.mean(hf_norm), alpha=0.2, color='purple')
    ax.set_title('High-Freq Texture Energy (HD Zone)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Normalized HF Energy')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 6. 高频纹理稳定性评分对比
    ax = axes[1, 2]
    hf_scores = [results['panel_full']['hf_stability_score'],
                 results['hd_zone']['hf_stability_score'],
                 results['peripheral']['hf_stability_score']]
    bars = ax.bar(zones, hf_scores, color=['steelblue', 'crimson', 'seagreen'], alpha=0.8)
    for bar, score in zip(bars, hf_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{score:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_title('HF Texture Stability Score (lower is better)')
    ax.set_ylabel('Score (0-100)')
    ax.set_ylim(0, max(hf_scores) * 1.4 + 5)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    chart_path = output_dir / "et_flicker_chart_v3.png"
    plt.savefig(str(chart_path), dpi=120, bbox_inches='tight')
    plt.close()
    print(f"[Step 7] 图表已保存: {chart_path}")


# ─────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VR ET-Center Panel Flicker Analyzer v3.0')
    parser.add_argument('video', help='输入视频路径')
    parser.add_argument('--output-dir', default='./et_v3_results', help='输出目录')
    args = parser.parse_args()

    analyze(args.video, args.output_dir)
