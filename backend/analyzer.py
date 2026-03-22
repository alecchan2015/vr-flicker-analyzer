#!/usr/bin/env python3
"""
VR Flicker Analyzer - Core Analysis Engine
==========================================
封装全维度闪烁分析算法：
  - 整体亮度闪烁 (Percent Flicker, Flicker Index, Temporal Contrast)
  - 文字边缘闪烁 (Edge Flicker)
  - 高频纹理稳定性 (HF Texture Stability)
  - 文字边缘锯齿 (Aliasing)
  - 自动面板检测 + ET 中心定位
  - 支持单视频分析 / 双视频对比
"""

import cv2
import numpy as np
import json
import os
import time
import base64
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 中文字体支持
for font in ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'SimHei', 'DejaVu Sans']:
    try:
        rcParams['font.family'] = [font, 'DejaVu Sans']
        break
    except Exception:
        continue
rcParams['axes.unicode_minus'] = False


# ─────────────────────────────────────────────────────────────────────────────
# 基础指标计算
# ─────────────────────────────────────────────────────────────────────────────

def calc_percent_flicker(series):
    mx, mn = np.max(series), np.min(series)
    if mx + mn < 1e-6:
        return 0.0
    return float((mx - mn) / (mx + mn) * 100)

def calc_flicker_index(series):
    mean_val = np.mean(series)
    above = np.sum(np.maximum(series - mean_val, 0))
    total = np.sum(np.abs(series - mean_val))
    return float(above / total) if total > 1e-6 else 0.0

def calc_temporal_contrast(series):
    mu = np.mean(series)
    return float(np.std(series) / mu) if mu > 1e-6 else 0.0

def calc_frame_diff_mean(series):
    diffs = np.abs(np.diff(series))
    return float(np.mean(diffs)) if len(diffs) > 0 else 0.0

def calc_dominant_freq(series, fps):
    n = len(series)
    if n < 4:
        return 0.0
    fft_vals = np.abs(np.fft.rfft(series - np.mean(series)))
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    fft_vals[0] = 0
    return float(freqs[np.argmax(fft_vals)])

def calc_flicker_score(pf, fi, tc, fd_mean):
    s = (pf / 100 * 40) + (fi * 20) + (tc * 1000 * 15) + (min(fd_mean / 50, 1) * 10)
    return float(min(s, 100))

def severity_label(score):
    if score < 10:   return "Excellent"
    elif score < 25: return "Good"
    elif score < 45: return "Moderate"
    elif score < 65: return "Severe"
    else:            return "Critical"


# ─────────────────────────────────────────────────────────────────────────────
# 文字边缘闪烁
# ─────────────────────────────────────────────────────────────────────────────

def extract_edge_lum(frame_gray, canny_low=30, canny_high=100, dilate_px=2):
    edges = cv2.Canny(frame_gray, canny_low, canny_high)
    if dilate_px > 0:
        k = np.ones((dilate_px * 2 + 1, dilate_px * 2 + 1), np.uint8)
        edges = cv2.dilate(edges, k)
    px = frame_gray[edges > 0]
    return float(np.mean(px)) if px.size > 0 else float(np.mean(frame_gray))

def calc_edge_flicker_score(pf, tc, fd_mean):
    s = (min(pf, 100) / 100 * 60) + (min(tc / 0.05, 1) * 30) + (min(fd_mean / 5.0, 1) * 10)
    return float(min(s, 100))


# ─────────────────────────────────────────────────────────────────────────────
# 高频纹理稳定性
# ─────────────────────────────────────────────────────────────────────────────

def extract_hf_energy(frame_gray):
    lap = cv2.Laplacian(frame_gray, cv2.CV_64F)
    return float(np.var(lap))

def calc_hf_stability_score(hf_cv):
    s = min(hf_cv / 50 * 100, 100)
    return float(s)

def hf_stability_label(score):
    if score < 10:   return "极稳定"
    elif score < 25: return "稳定"
    elif score < 45: return "轻微抖动"
    elif score < 65: return "明显抖动"
    else:            return "严重抖动"


# ─────────────────────────────────────────────────────────────────────────────
# 文字边缘锯齿 (Aliasing)
# ─────────────────────────────────────────────────────────────────────────────

def compute_aliasing_metrics(frame_gray):
    """
    文字边缘锯齿评估：
    - Aliasing Ratio: 斜向边缘像素占比（越低越平滑）
    - Edge Transition Ratio: 边缘过渡宽度（越低越锐利）
    - Edge Sharpness CV: 边缘梯度幅值的变异系数
    """
    gx = cv2.Sobel(frame_gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(frame_gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    angle = np.arctan2(np.abs(gy), np.abs(gx)) * 180 / np.pi

    threshold = mag.max() * 0.15
    strong_mask = mag > threshold
    if strong_mask.sum() < 10:
        return 0.0, 0.0, 0.0

    angles = angle[strong_mask]
    diagonal_mask = (angles >= 22.5) & (angles <= 67.5)
    aliasing_ratio = float(diagonal_mask.sum()) / len(angles) * 100

    edges = cv2.Canny(frame_gray, 50, 150)
    k3 = np.ones((3,3), np.uint8)
    k5 = np.ones((5,5), np.uint8)
    edge_px = max(1, (edges > 0).sum())
    dil5_px = (cv2.dilate(edges, k5) > 0).sum()
    transition_ratio = float(dil5_px) / edge_px

    edge_mags = mag[strong_mask]
    sharpness_cv = float(edge_mags.std() / (edge_mags.mean() + 1e-6) * 100)

    return aliasing_ratio, transition_ratio, sharpness_cv


# ─────────────────────────────────────────────────────────────────────────────
# 面板自动检测
# ─────────────────────────────────────────────────────────────────────────────

def detect_panel(frame_gray, min_area_ratio=0.02):
    """自动检测画面中最大的亮色矩形面板"""
    _, bright = cv2.threshold(frame_gray, 200, 255, cv2.THRESH_BINARY)
    kernel = np.ones((30, 30), np.uint8)
    closed = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    h, w = frame_gray.shape
    min_area = h * w * min_area_ratio
    valid = [c for c in contours if cv2.contourArea(c) > min_area]
    if not valid:
        return None
    largest = max(valid, key=cv2.contourArea)
    x, y, pw, ph = cv2.boundingRect(largest)
    return (x, y, pw, ph)


# ─────────────────────────────────────────────────────────────────────────────
# ET 中心定位
# ─────────────────────────────────────────────────────────────────────────────

def find_et_center(frames_gray, panel_rect=None):
    """通过多帧 Laplacian 热力图找到 ET 中心（最高纹理密度区域）"""
    heat = None
    for g in frames_gray[:min(30, len(frames_gray))]:
        if panel_rect:
            x, y, w, h = panel_rect
            roi = g[y:y+h, x:x+w]
        else:
            roi = g
        lap = np.abs(cv2.Laplacian(roi.astype(np.float32), cv2.CV_32F))
        blurred = cv2.GaussianBlur(lap, (51, 51), 0)
        if heat is None:
            heat = blurred.astype(np.float64)
        else:
            heat += blurred.astype(np.float64)

    if heat is None:
        rh, rw = frames_gray[0].shape
        return rw // 2, rh // 2

    _, _, _, max_loc = cv2.minMaxLoc(heat)
    cx, cy = max_loc
    if panel_rect:
        cx += panel_rect[0]
        cy += panel_rect[1]
    return cx, cy


# ─────────────────────────────────────────────────────────────────────────────
# 主分析函数
# ─────────────────────────────────────────────────────────────────────────────

def analyze_video(video_path, output_dir, device_name="Device", max_frames=300,
                  panel_rect=None, progress_callback=None):
    """
    对单个视频进行全维度闪烁分析。

    参数:
        video_path: 视频文件路径
        output_dir: 输出目录
        device_name: 设备名称（用于报告标题）
        max_frames: 最大分析帧数
        panel_rect: 手动指定面板坐标 (x, y, w, h)，None 为自动检测
        progress_callback: 进度回调函数 (0.0~1.0)

    返回:
        dict: 完整的分析结果
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 如果是双目视频（宽度 >> 高度），取右眼
    is_stereo = frame_w > frame_h * 1.5
    eye_offset = frame_w // 2 if is_stereo else 0

    # 采样帧
    step = max(1, total_frames // max_frames)
    frames_gray = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            if is_stereo:
                eye = frame[:, eye_offset:, :]
            else:
                eye = frame
            gray = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
            frames_gray.append(gray)
            if len(frames_gray) >= max_frames:
                break
        frame_idx += 1
        if progress_callback and frame_idx % 30 == 0:
            progress_callback(min(0.4, frame_idx / max(1, total_frames) * 0.4))
    cap.release()

    n = len(frames_gray)
    if n < 3:
        raise ValueError("视频帧数不足（至少需要3帧）")

    # 自动检测面板
    if panel_rect is None:
        panel_rect = detect_panel(frames_gray[n // 2])

    # ET 中心
    et_cx, et_cy = find_et_center(frames_gray, panel_rect)

    # 确定分析 ROI（以 ET 中心为准的圆形区域）
    if panel_rect:
        px, py, pw, ph = panel_rect
        r_focus = min(pw, ph) // 6
        r_surround = min(pw, ph) // 3
    else:
        h, w = frames_gray[0].shape
        r_focus = min(w, h) // 8
        r_surround = min(w, h) // 4

    # 提取各帧指标
    lum_series = []
    edge_series = []
    hf_series = []
    aliasing_series = []
    transition_series = []
    sharpness_series = []

    for i, g in enumerate(frames_gray):
        h, w = g.shape
        # 圆形 ROI
        Y, X = np.ogrid[:h, :w]
        mask = (X - et_cx)**2 + (Y - et_cy)**2 <= r_focus**2
        roi_pixels = g[mask]
        lum = float(np.mean(roi_pixels)) if roi_pixels.size > 0 else float(np.mean(g))
        lum_series.append(lum)

        # 边缘亮度
        x1 = max(0, et_cx - r_focus)
        y1 = max(0, et_cy - r_focus)
        x2 = min(w, et_cx + r_focus)
        y2 = min(h, et_cy + r_focus)
        roi_crop = g[y1:y2, x1:x2]
        edge_series.append(extract_edge_lum(roi_crop))

        # 高频能量
        hf_series.append(extract_hf_energy(roi_crop))

        # 锯齿
        ar, tr, sc = compute_aliasing_metrics(roi_crop)
        aliasing_series.append(ar)
        transition_series.append(tr)
        sharpness_series.append(sc)

        if progress_callback and i % 20 == 0:
            progress_callback(0.4 + i / n * 0.5)

    # 计算指标
    lum_arr = np.array(lum_series)
    edge_arr = np.array(edge_series)
    hf_arr = np.array(hf_series)
    ali_arr = np.array(aliasing_series)

    pf = calc_percent_flicker(lum_arr)
    fi = calc_flicker_index(lum_arr)
    tc = calc_temporal_contrast(lum_arr)
    fd = calc_frame_diff_mean(lum_arr)
    dom_freq = calc_dominant_freq(lum_arr, fps)
    flicker_score = calc_flicker_score(pf, fi, tc, fd)

    e_pf = calc_percent_flicker(edge_arr)
    e_tc = calc_temporal_contrast(edge_arr)
    e_fd = calc_frame_diff_mean(edge_arr)
    edge_score = calc_edge_flicker_score(e_pf, e_tc, e_fd)

    hf_cv = float(hf_arr.std() / (hf_arr.mean() + 1e-6) * 100)
    hf_score = calc_hf_stability_score(hf_cv)

    ali_mean = float(np.mean(ali_arr))
    ali_cv = float(ali_arr.std() / (ali_arr.mean() + 1e-6) * 100)
    tr_mean = float(np.mean(transition_series))
    sc_mean = float(np.mean(sharpness_series))

    result = {
        "device": device_name,
        "video": os.path.basename(video_path),
        "fps": round(fps, 2),
        "n_frames": n,
        "is_stereo": is_stereo,
        "et_center": [et_cx, et_cy],
        "panel_rect": list(panel_rect) if panel_rect else None,
        "focus_radius": r_focus,
        # 整体亮度
        "percent_flicker": round(pf, 4),
        "flicker_index": round(fi, 4),
        "temporal_contrast": round(tc, 4),
        "frame_diff_mean": round(fd, 4),
        "dominant_freq_hz": round(dom_freq, 2),
        "flicker_score": round(flicker_score, 2),
        "flicker_label": severity_label(flicker_score),
        # 边缘闪烁
        "edge_percent_flicker": round(e_pf, 4),
        "edge_temporal_contrast": round(e_tc, 4),
        "edge_frame_diff_mean": round(e_fd, 4),
        "edge_flicker_score": round(edge_score, 2),
        "edge_flicker_label": severity_label(edge_score),
        # 高频纹理
        "hf_stability_cv": round(hf_cv, 4),
        "hf_stability_score": round(hf_score, 2),
        "hf_stability_label": hf_stability_label(hf_score),
        # 锯齿
        "aliasing_ratio_mean": round(ali_mean, 4),
        "aliasing_temporal_cv": round(ali_cv, 4),
        "edge_transition_ratio": round(tr_mean, 4),
        "edge_sharpness_cv": round(sc_mean, 4),
        # 时序数据（用于绘图）
        "_lum_series": lum_arr.tolist(),
        "_edge_series": edge_arr.tolist(),
        "_hf_series": hf_arr.tolist(),
        "_aliasing_series": ali_arr.tolist(),
    }

    # 生成可视化图表
    chart_path = _generate_chart(result, output_dir, device_name)
    result["chart_path"] = chart_path

    # 保存 JSON（不含时序数据）
    json_result = {k: v for k, v in result.items() if not k.startswith('_')}
    json_path = os.path.join(output_dir, f"metrics_{device_name.replace(' ', '_')}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_result, f, ensure_ascii=False, indent=2)
    result["json_path"] = json_path

    if progress_callback:
        progress_callback(1.0)

    return result


def compare_videos(video_a_path, video_b_path, output_dir,
                   name_a="Device A", name_b="Device B",
                   max_frames=300, progress_callback=None):
    """
    对两个视频进行对比分析。

    返回:
        dict: 包含两台设备结果和对比图表路径
    """
    os.makedirs(output_dir, exist_ok=True)

    def cb_a(p):
        if progress_callback:
            progress_callback(p * 0.45)

    def cb_b(p):
        if progress_callback:
            progress_callback(0.45 + p * 0.45)

    result_a = analyze_video(video_a_path, output_dir, name_a, max_frames, progress_callback=cb_a)
    result_b = analyze_video(video_b_path, output_dir, name_b, max_frames, progress_callback=cb_b)

    # 生成对比图表
    compare_chart = _generate_compare_chart(result_a, result_b, output_dir)

    if progress_callback:
        progress_callback(1.0)

    return {
        "device_a": {k: v for k, v in result_a.items() if not k.startswith('_')},
        "device_b": {k: v for k, v in result_b.items() if not k.startswith('_')},
        "compare_chart_path": compare_chart,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 可视化
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    'Excellent': '#4CAF50',
    'Good':      '#8BC34A',
    'Moderate':  '#FFC107',
    'Severe':    '#FF5722',
    'Critical':  '#F44336',
}

def _score_color(score):
    if score < 10:   return COLORS['Excellent']
    elif score < 25: return COLORS['Good']
    elif score < 45: return COLORS['Moderate']
    elif score < 65: return COLORS['Severe']
    else:            return COLORS['Critical']


def _generate_chart(result, output_dir, device_name):
    """生成单设备 6 面板分析图"""
    fps = result['fps']
    lum = np.array(result['_lum_series'])
    edge = np.array(result['_edge_series'])
    hf = np.array(result['_hf_series'])
    ali = np.array(result['_aliasing_series'])
    t = np.arange(len(lum)) / fps

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(f'{device_name} — 全维度闪烁分析报告', fontsize=14, fontweight='bold')
    color = '#2196F3'

    # 1. 亮度时序
    ax = axes[0, 0]
    ax.plot(t, lum, color=color, lw=0.8, alpha=0.9)
    ax.fill_between(t, lum, alpha=0.15, color=color)
    ax.set_title(f'亮度时序  PF={result["percent_flicker"]:.2f}%')
    ax.set_xlabel('时间(s)'); ax.set_ylabel('亮度')
    ax.grid(True, alpha=0.3)

    # 2. 边缘亮度时序
    ax = axes[0, 1]
    ax.plot(t, edge, color='#FF9800', lw=0.8, alpha=0.9)
    ax.set_title(f'边缘亮度时序  EPF={result["edge_percent_flicker"]:.2f}%')
    ax.set_xlabel('时间(s)'); ax.set_ylabel('边缘亮度')
    ax.grid(True, alpha=0.3)

    # 3. 锯齿比例时序
    ax = axes[0, 2]
    ax.plot(t, ali, color='#9C27B0', lw=0.8, alpha=0.9)
    ax.set_title(f'锯齿比例时序  均值={result["aliasing_ratio_mean"]:.2f}%')
    ax.set_xlabel('时间(s)'); ax.set_ylabel('斜向边缘比例(%)')
    ax.grid(True, alpha=0.3)

    # 4. 高频纹理时序
    ax = axes[1, 0]
    ax.plot(t, hf, color='#009688', lw=0.8, alpha=0.9)
    ax.set_title(f'高频纹理能量  CV={result["hf_stability_cv"]:.2f}%')
    ax.set_xlabel('时间(s)'); ax.set_ylabel('Laplacian方差')
    ax.grid(True, alpha=0.3)

    # 5. 综合评分雷达/条形
    ax = axes[1, 1]
    metrics = ['整体闪烁', '边缘闪烁', 'HF稳定性', '锯齿时域CV']
    scores = [
        result['flicker_score'],
        result['edge_flicker_score'],
        result['hf_stability_score'],
        min(result['aliasing_temporal_cv'], 100),
    ]
    bar_colors = [_score_color(s) for s in scores]
    bars = ax.barh(metrics, scores, color=bar_colors, alpha=0.85)
    ax.set_xlim(0, 100)
    ax.set_title('综合评分（越低越好）')
    ax.axvline(x=10, color='green', linestyle='--', alpha=0.5, lw=0.8)
    ax.axvline(x=25, color='orange', linestyle='--', alpha=0.5, lw=0.8)
    ax.axvline(x=50, color='red', linestyle='--', alpha=0.5, lw=0.8)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{score:.1f}', va='center', fontsize=9)
    ax.grid(True, alpha=0.3, axis='x')

    # 6. 数据汇总
    ax = axes[1, 2]
    ax.axis('off')
    table_data = [
        ['指标', '数值', '评级'],
        ['Percent Flicker', f"{result['percent_flicker']:.4f}%", result['flicker_label']],
        ['Edge Flicker', f"{result['edge_percent_flicker']:.4f}%", result['edge_flicker_label']],
        ['Temporal Contrast', f"{result['temporal_contrast']:.4f}", '-'],
        ['HF Stability CV', f"{result['hf_stability_cv']:.2f}%", result['hf_stability_label']],
        ['Aliasing Ratio', f"{result['aliasing_ratio_mean']:.2f}%", '-'],
        ['Edge Transition', f"{result['edge_transition_ratio']:.2f}", '-'],
        ['主频', f"{result['dominant_freq_hz']:.1f}Hz", '-'],
    ]
    table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                     cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#37474F')
            cell.set_text_props(color='white', fontweight='bold')
        elif row % 2 == 0:
            cell.set_facecolor('#F5F5F5')
    ax.set_title('数据汇总', fontsize=10, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    chart_path = os.path.join(output_dir, f"chart_{device_name.replace(' ', '_')}.png")
    plt.savefig(chart_path, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close()
    return chart_path


def _generate_compare_chart(ra, rb, output_dir):
    """生成双设备对比图"""
    fps_a = ra['fps']
    fps_b = rb['fps']
    lum_a = np.array(ra['_lum_series'])
    lum_b = np.array(rb['_lum_series'])
    edge_a = np.array(ra['_edge_series'])
    edge_b = np.array(rb['_edge_series'])
    ali_a = np.array(ra['_aliasing_series'])
    ali_b = np.array(rb['_aliasing_series'])
    t_a = np.arange(len(lum_a)) / fps_a
    t_b = np.arange(len(lum_b)) / fps_b

    color_a = '#2196F3'
    color_b = '#FF5722'
    name_a = ra['device']
    name_b = rb['device']

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(f'VR 闪烁对比分析：{name_a} vs {name_b}', fontsize=14, fontweight='bold')

    # 1. 亮度时序
    ax1 = fig.add_subplot(3, 3, 1)
    ax1.plot(t_a, lum_a, color=color_a, lw=0.8, alpha=0.85, label=name_a)
    ax1.plot(t_b, lum_b, color=color_b, lw=0.8, alpha=0.85, label=name_b)
    ax1.set_title('亮度时序'); ax1.set_xlabel('时间(s)'); ax1.set_ylabel('亮度')
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    # 2. 边缘亮度时序
    ax2 = fig.add_subplot(3, 3, 2)
    ax2.plot(t_a, edge_a, color=color_a, lw=0.8, alpha=0.85, label=name_a)
    ax2.plot(t_b, edge_b, color=color_b, lw=0.8, alpha=0.85, label=name_b)
    ax2.set_title('文字边缘亮度时序'); ax2.set_xlabel('时间(s)'); ax2.set_ylabel('边缘亮度')
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    # 3. 锯齿比例时序
    ax3 = fig.add_subplot(3, 3, 3)
    ax3.plot(t_a, ali_a, color=color_a, lw=0.8, alpha=0.85, label=name_a)
    ax3.plot(t_b, ali_b, color=color_b, lw=0.8, alpha=0.85, label=name_b)
    ax3.set_title('锯齿比例时序'); ax3.set_xlabel('时间(s)'); ax3.set_ylabel('斜向边缘(%)')
    ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)

    # 4. 逐帧亮度差
    ax4 = fig.add_subplot(3, 3, 4)
    ax4.plot(t_a[1:], np.abs(np.diff(lum_a)), color=color_a, lw=0.8, alpha=0.85, label=name_a)
    ax4.plot(t_b[1:], np.abs(np.diff(lum_b)), color=color_b, lw=0.8, alpha=0.85, label=name_b)
    ax4.set_title('逐帧亮度差'); ax4.set_xlabel('时间(s)'); ax4.set_ylabel('帧间差')
    ax4.legend(fontsize=8); ax4.grid(True, alpha=0.3)

    # 5. 亮度分布
    ax5 = fig.add_subplot(3, 3, 5)
    ax5.hist(lum_a, bins=30, color=color_a, alpha=0.6, density=True, label=name_a)
    ax5.hist(lum_b, bins=30, color=color_b, alpha=0.6, density=True, label=name_b)
    ax5.set_title('亮度分布（越集中越稳定）'); ax5.set_xlabel('亮度值'); ax5.set_ylabel('密度')
    ax5.legend(fontsize=8); ax5.grid(True, alpha=0.3)

    # 6. 锯齿比例分布
    ax6 = fig.add_subplot(3, 3, 6)
    ax6.hist(ali_a, bins=20, color=color_a, alpha=0.6, density=True, label=name_a)
    ax6.hist(ali_b, bins=20, color=color_b, alpha=0.6, density=True, label=name_b)
    ax6.set_title('锯齿比例分布'); ax6.set_xlabel('斜向边缘比例(%)'); ax6.set_ylabel('密度')
    ax6.legend(fontsize=8); ax6.grid(True, alpha=0.3)

    # 7. 全维度对比柱状图
    ax7 = fig.add_subplot(3, 3, (7, 8))
    keys = ['percent_flicker', 'edge_percent_flicker', 'temporal_contrast',
            'edge_temporal_contrast', 'hf_stability_cv', 'aliasing_ratio_mean',
            'aliasing_temporal_cv', 'edge_transition_ratio']
    labels = ['PF(%)', 'EdgePF(%)', 'TC(%)', 'EdgeTC(%)',
              'HF CV(%)', 'Ali(%)', 'AliCV(%)', 'EdgeTR']
    x = np.arange(len(keys))
    w = 0.35
    vals_a = [ra.get(k, 0) for k in keys]
    vals_b = [rb.get(k, 0) for k in keys]
    bars_a = ax7.bar(x - w/2, vals_a, w, label=name_a, color=color_a, alpha=0.8)
    bars_b = ax7.bar(x + w/2, vals_b, w, label=name_b, color=color_b, alpha=0.8)
    ax7.set_xticks(x); ax7.set_xticklabels(labels, fontsize=8)
    ax7.set_title('全维度指标对比（越低越好）')
    ax7.legend(fontsize=9); ax7.grid(True, alpha=0.3, axis='y')
    for bar in list(bars_a) + list(bars_b):
        h = bar.get_height()
        ax7.text(bar.get_x()+bar.get_width()/2, h+0.01, f'{h:.2f}',
                 ha='center', va='bottom', fontsize=6, rotation=45)

    # 8. 数据汇总表
    ax8 = fig.add_subplot(3, 3, 9)
    ax8.axis('off')
    metrics_labels = ['Percent Flicker(%)', 'Edge Flicker(%)', 'Temporal Contrast',
                      'HF Stability CV(%)', 'Aliasing Ratio(%)', 'Aliasing CV(%)',
                      'Edge Transition', '综合评分', '边缘评分', 'HF评分']
    metrics_keys_a = ['percent_flicker', 'edge_percent_flicker', 'temporal_contrast',
                      'hf_stability_cv', 'aliasing_ratio_mean', 'aliasing_temporal_cv',
                      'edge_transition_ratio', 'flicker_score', 'edge_flicker_score', 'hf_stability_score']
    table_rows = []
    for lbl, key in zip(metrics_labels, metrics_keys_a):
        va = ra.get(key, 0)
        vb = rb.get(key, 0)
        winner = f'{name_a} ✓' if va < vb else f'{name_b} ✓'
        table_rows.append([lbl, f'{va:.3f}', f'{vb:.3f}', winner])

    table = ax8.table(cellText=table_rows,
                      colLabels=['指标', name_a, name_b, '优胜'],
                      cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#37474F')
            cell.set_text_props(color='white', fontweight='bold')
        elif col == 3 and row > 0:
            text = cell.get_text().get_text()
            if name_a in text:
                cell.set_facecolor('#E3F2FD')
            else:
                cell.set_facecolor('#FFF3E0')
        elif row % 2 == 0:
            cell.set_facecolor('#F5F5F5')
    ax8.set_title('数据汇总', fontsize=9, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    chart_path = os.path.join(output_dir, 'compare_chart.png')
    plt.savefig(chart_path, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close()
    return chart_path


def img_to_base64(path):
    """将图片转为 base64 字符串"""
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')
