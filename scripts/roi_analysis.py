#!/usr/bin/env python3
"""
最终分析：
1. 在透视矫正坐标系中定位 Row5（第4行17号字）的「最看重质量」列坐标
2. 反变换回原始视频坐标系，得到每帧的 ROI 四边形
3. 对每帧做透视矫正后提取 ROI，计算所有闪烁指标
4. 新增文字边缘锯齿（Aliasing）评估
"""
import cv2
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

# ─────────────────────────────────────────────────────────────────────────────
# 锯齿评估函数
# ─────────────────────────────────────────────────────────────────────────────
def compute_aliasing_score(frame_gray):
    """
    文字边缘锯齿评估：
    1. 用 Canny 检测文字边缘
    2. 对边缘像素的梯度方向做统计：
       - 理想的水平/垂直笔画边缘应该是 0°/90°（无锯齿）
       - 锯齿会产生 45°/135° 方向的边缘
    3. 计算锯齿比例 = 斜向边缘数 / 总边缘数
    4. 同时计算边缘亚像素过渡宽度（越窄越锐利，越宽越模糊/锯齿）
    """
    # Sobel 梯度
    gx = cv2.Sobel(frame_gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(frame_gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    angle = np.arctan2(np.abs(gy), np.abs(gx)) * 180 / np.pi  # 0~90°
    
    # 只看强边缘（文字边缘）
    threshold = mag.max() * 0.15
    strong_mask = mag > threshold
    
    if strong_mask.sum() < 10:
        return 0.0, 0.0, 0.0
    
    angles = angle[strong_mask]
    
    # 水平边缘：angle < 22.5° 或 > 67.5°（接近水平/垂直）
    # 斜向边缘：22.5° ~ 67.5°（锯齿）
    diagonal_mask = (angles >= 22.5) & (angles <= 67.5)
    aliasing_ratio = float(diagonal_mask.sum()) / len(angles) * 100
    
    # 边缘过渡宽度（通过梯度幅值的局部宽度估算）
    # 使用 Canny 边缘的膨胀宽度
    edges = cv2.Canny(frame_gray, 50, 150)
    kernel3 = np.ones((3,3), np.uint8)
    kernel5 = np.ones((5,5), np.uint8)
    dilated3 = cv2.dilate(edges, kernel3)
    dilated5 = cv2.dilate(edges, kernel5)
    edge_pixels = edges.sum() / 255
    dilated3_pixels = dilated3.sum() / 255
    dilated5_pixels = dilated5.sum() / 255
    
    # 过渡宽度比 = 膨胀后边缘像素 / 原始边缘像素（越大说明边缘越宽/越模糊）
    transition_ratio = float(dilated5_pixels) / (edge_pixels + 1e-6)
    
    # 边缘清晰度（梯度幅值峰值的集中程度）
    # 用边缘区域梯度幅值的变异系数衡量（越小越均匀/清晰）
    edge_mags = mag[strong_mask]
    sharpness_cv = float(edge_mags.std() / (edge_mags.mean() + 1e-6) * 100)
    
    return aliasing_ratio, transition_ratio, sharpness_cv


# ─────────────────────────────────────────────────────────────────────────────
# 全维度指标计算
# ─────────────────────────────────────────────────────────────────────────────
def compute_all_metrics(frames):
    n = len(frames)
    if n < 2:
        return {}
    
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames]
    means = np.array([g.mean() for g in grays])
    lmax, lmin = means.max(), means.min()
    pf = (lmax - lmin) / (lmax + lmin + 1e-6) * 100
    tc = means.std() / (means.mean() + 1e-6) * 100
    diffs = [np.abs(grays[i+1] - grays[i]).mean() for i in range(n-1)]
    fd = float(np.mean(diffs))
    
    # 文字边缘闪烁
    edge_means = []
    for g in grays:
        g8 = g.astype(np.uint8)
        edges = cv2.Canny(g8, 50, 150)
        mask = edges > 0
        edge_means.append(float(g[mask].mean()) if mask.sum() > 0 else float(g.mean()))
    edge_means = np.array(edge_means)
    e_lmax, e_lmin = edge_means.max(), edge_means.min()
    e_pf = (e_lmax - e_lmin) / (e_lmax + e_lmin + 1e-6) * 100
    e_tc = edge_means.std() / (edge_means.mean() + 1e-6) * 100
    e_diffs = [abs(edge_means[i+1] - edge_means[i]) for i in range(n-1)]
    e_fd = float(np.mean(e_diffs))
    
    # 高频纹理稳定性
    lap_vars = np.array([cv2.Laplacian(g.astype(np.uint8), cv2.CV_64F).var() for g in grays])
    hf_cv = float(lap_vars.std() / (lap_vars.mean() + 1e-6) * 100)
    
    # 锯齿评估（对每帧计算，取均值）
    aliasing_ratios = []
    transition_ratios = []
    sharpness_cvs = []
    for g in grays:
        ar, tr, sc = compute_aliasing_score(g.astype(np.uint8))
        aliasing_ratios.append(ar)
        transition_ratios.append(tr)
        sharpness_cvs.append(sc)
    
    # 锯齿时域稳定性（帧间锯齿比例的变化）
    aliasing_arr = np.array(aliasing_ratios)
    aliasing_temporal_cv = float(aliasing_arr.std() / (aliasing_arr.mean() + 1e-6) * 100)
    
    return {
        'n_frames': n,
        'mean_luminance': round(float(means.mean()), 2),
        'percent_flicker': round(float(pf), 4),
        'temporal_contrast': round(float(tc), 4),
        'frame_diff_mean': round(float(fd), 4),
        'edge_percent_flicker': round(float(e_pf), 4),
        'edge_temporal_contrast': round(float(e_tc), 4),
        'edge_frame_diff_mean': round(float(e_fd), 4),
        'hf_stability_cv': round(float(hf_cv), 4),
        'aliasing_ratio_mean': round(float(np.mean(aliasing_ratios)), 4),
        'aliasing_ratio_std': round(float(np.std(aliasing_ratios)), 4),
        'aliasing_temporal_cv': round(float(aliasing_temporal_cv), 4),
        'edge_transition_ratio': round(float(np.mean(transition_ratios)), 4),
        'edge_sharpness_cv': round(float(np.mean(sharpness_cvs)), 4),
        'luminance_series': means.tolist(),
        'edge_series': edge_means.tolist(),
        'aliasing_series': aliasing_arr.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Galaxy XR：使用透视矫正提取 Row5「最看重质量」
# ─────────────────────────────────────────────────────────────────────────────
print("=== Galaxy XR 分析 ===")

# 加载透视矫正数据
with open('/home/ubuntu/roi_v2/perspective_data.json') as f:
    pdata = json.load(f)
M = np.array(pdata['M'])
dst_w = pdata['dst_w']
dst_h = pdata['dst_h']
rows = pdata['rows']  # 矫正坐标系中的行坐标

# 第4行17号字: y=1075~1210（从可视化确认的精确范围）
r5_ys, r5_ye = 1075, 1210
print(f"第4行17号字: y={r5_ys}~{r5_ye} (h={r5_ye-r5_ys}px)")

# 在矫正图中找「最看重质量」列坐标
warped_sample = cv2.imread('/home/ubuntu/roi_v2/gx_panel_warped.png')
row5_warped = warped_sample[r5_ys:r5_ye, :, :]
row5_gray = cv2.cvtColor(row5_warped, cv2.COLOR_BGR2GRAY)
_, bw5 = cv2.threshold(row5_gray, 180, 255, cv2.THRESH_BINARY_INV)
col_proj = np.sum(bw5 > 0, axis=0).astype(float)
col_smooth = np.convolve(col_proj, np.ones(5)/5, mode='same')
col_norm = col_smooth / (col_smooth.max() + 1e-6)

in_col = col_norm > 0.05
segs = []
start_c = None
for i, v in enumerate(in_col):
    if v and start_c is None:
        start_c = i
    elif not v and start_c is not None:
        if i - start_c >= 5:
            segs.append([start_c, i])
        start_c = None
if start_c is not None:
    segs.append([start_c, dst_w])

seg_merged = []
for s in segs:
    if seg_merged and s[0] - seg_merged[-1][1] < 40:
        seg_merged[-1][1] = s[1]
    else:
        seg_merged.append(s)

print(f"Row5 词组 ({len(seg_merged)} 组):")
for i, (xs, xe) in enumerate(seg_merged):
    print(f"  Group {i+1}: x={xs}~{xe} (w={xe-xs}px)")
    seg_img = row5_warped[:, xs:xe, :]
    scale = max(1, 100 // max(1, r5_ye-r5_ys))
    seg_zoom = cv2.resize(seg_img, (seg_img.shape[1]*scale, (r5_ye-r5_ys)*scale),
                          interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite(f'/home/ubuntu/roi_v2/gx_row5_seg{i+1}.png', seg_zoom)

# 「最看重质量」在 Group1 (x=0~560) 的第3小组：x=221~560
# 已通过可视化确认：Group1内分为 [0~155]》[-]》[221~560]，第3小组就是「最看重质量」
zui_xs, zui_xe = 221, 560  # 矫正坐标系中的列范围
if len(seg_merged) > 0:
    print(f"\n「最看重质量」矫正坐标: x={zui_xs}~{zui_xe}, y={r5_ys}~{r5_ye}")
    
    # 保存验证截图
    roi_warped = warped_sample[r5_ys:r5_ye, zui_xs:zui_xe, :]
    scale = max(1, 200 // max(1, r5_ye-r5_ys))
    roi_zoom = cv2.resize(roi_warped, (roi_warped.shape[1]*scale, (r5_ye-r5_ys)*scale),
                          interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite('/home/ubuntu/roi_v2/gx_zuikanzhi_warped.png', roi_zoom)
    print("Saved: gx_zuikanzhi_warped.png")

# 逐帧提取 ROI（对每帧做透视矫正后裁剪）
gx_video = '/home/ubuntu/upload/GX_完整分辨率录制4.mp4'
cap = cv2.VideoCapture(gx_video)
fps_gx = cap.get(cv2.CAP_PROP_FPS)
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
eye_w = frame_w // 2

gx_frames = []
frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    right = frame[:, eye_w:, :]
    warped_frame = cv2.warpPerspective(right, M, (dst_w, dst_h))
    roi = warped_frame[r5_ys:r5_ye, zui_xs:zui_xe, :]
    if roi.size > 0:
        gx_frames.append(roi.copy())
    frame_idx += 1
cap.release()
print(f"提取 {len(gx_frames)} 帧, {fps_gx:.1f}FPS")

# 保存样本帧
if gx_frames:
    mid = gx_frames[len(gx_frames)//2]
    scale = max(1, 200 // max(1, mid.shape[0]))
    cv2.imwrite('/home/ubuntu/roi_v2/gx_roi_final_sample.png',
                cv2.resize(mid, (mid.shape[1]*scale, mid.shape[0]*scale),
                           interpolation=cv2.INTER_LANCZOS4))

gx_m = compute_all_metrics(gx_frames)
gx_m['device'] = 'Galaxy XR'
gx_m['fps'] = round(fps_gx, 2)
gx_m['roi_note'] = '第4行17号字「最看重质量」（透视矫正后精确ROI）'
print("GX 分析完成")


# ─────────────────────────────────────────────────────────────────────────────
# Swan EVT：精确 ROI（旋转坐标系）
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Swan EVT 分析 ===")
swan_video = '/home/ubuntu/upload/SwanEVT_完整分辨率录制3.mp4'
cap = cv2.VideoCapture(swan_video)
fps_swan = cap.get(cv2.CAP_PROP_FPS)
sx, sy, sw, sh = 516, 966, 1076, 970
roi_rx, roi_ry, roi_rw, roi_rh = 286, 265, 128, 35

swan_frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    text_roi = frame[sy:sy+sh, sx:sx+sw]
    rot = cv2.rotate(text_roi, cv2.ROTATE_90_COUNTERCLOCKWISE)
    roi = rot[roi_ry:roi_ry+roi_rh, roi_rx:roi_rx+roi_rw]
    if roi.size > 0:
        swan_frames.append(roi.copy())
cap.release()
print(f"提取 {len(swan_frames)} 帧, {fps_swan:.1f}FPS")

# 保存样本帧
if swan_frames:
    mid = swan_frames[len(swan_frames)//2]
    scale = max(1, 200 // max(1, mid.shape[0]))
    cv2.imwrite('/home/ubuntu/roi_v2/swan_roi_final_sample.png',
                cv2.resize(mid, (mid.shape[1]*scale, mid.shape[0]*scale),
                           interpolation=cv2.INTER_LANCZOS4))

swan_m = compute_all_metrics(swan_frames)
swan_m['device'] = 'Swan EVT'
swan_m['fps'] = round(fps_swan, 2)
swan_m['roi_note'] = '第4行「最看重质量」（旋转坐标系精确ROI）'
print("Swan 分析完成")


# ─────────────────────────────────────────────────────────────────────────────
# 打印对比表
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*90)
print(f"{'指标':<35} {'Swan EVT':>18} {'Galaxy XR':>18} {'优胜':>10}")
print("-"*90)
rows_table = [
    ('帧数', 'n_frames', False),
    ('FPS', 'fps', False),
    ('平均亮度', 'mean_luminance', False),
    ('── 整体亮度 ──', None, False),
    ('Percent Flicker (%)', 'percent_flicker', True),
    ('Temporal Contrast (%)', 'temporal_contrast', True),
    ('Frame Diff Mean', 'frame_diff_mean', True),
    ('── 文字边缘闪烁 ──', None, False),
    ('Edge Percent Flicker (%)', 'edge_percent_flicker', True),
    ('Edge Temporal Contrast (%)', 'edge_temporal_contrast', True),
    ('Edge Frame Diff Mean', 'edge_frame_diff_mean', True),
    ('── 高频纹理 ──', None, False),
    ('HF Stability CV (%)', 'hf_stability_cv', True),
    ('── 文字边缘锯齿 ──', None, False),
    ('Aliasing Ratio Mean (%)', 'aliasing_ratio_mean', True),
    ('Aliasing Temporal CV (%)', 'aliasing_temporal_cv', True),
    ('Edge Transition Ratio', 'edge_transition_ratio', True),
    ('Edge Sharpness CV (%)', 'edge_sharpness_cv', True),
]
for label, key, lib in rows_table:
    if key is None:
        print(f"\n  {label}")
        continue
    sv = swan_m.get(key, '-')
    gv = gx_m.get(key, '-')
    try:
        w = ('GX ✓' if float(gv) < float(sv) else 'Swan ✓') if lib else '-'
    except:
        w = '-'
    print(f"  {label:<33} {str(sv):>18} {str(gv):>18} {w:>10}")
print("="*90)


# ─────────────────────────────────────────────────────────────────────────────
# 可视化报告
# ─────────────────────────────────────────────────────────────────────────────
print("\n生成可视化报告...")
swan_color = '#2196F3'
gx_color = '#FF5722'

fig = plt.figure(figsize=(18, 16))
fig.suptitle('应用面板第4行「最看重质量」全维度闪烁分析\n（Galaxy XR: 透视矫正精确ROI | Swan EVT: 旋转坐标系精确ROI）',
             fontsize=13, fontweight='bold', y=0.99)

swan_lum = swan_m['luminance_series']
gx_lum = gx_m['luminance_series']
swan_edge = swan_m['edge_series']
gx_edge = gx_m['edge_series']
swan_ali = swan_m['aliasing_series']
gx_ali = gx_m['aliasing_series']
t_swan = np.arange(len(swan_lum)) / fps_swan
t_gx = np.arange(len(gx_lum)) / fps_gx

# 1. 亮度时序
ax1 = fig.add_subplot(4, 3, 1)
ax1.plot(t_swan, swan_lum, color=swan_color, alpha=0.8, lw=0.8, label='Swan EVT')
ax1.plot(t_gx, gx_lum, color=gx_color, alpha=0.8, lw=0.8, label='Galaxy XR')
ax1.set_title('亮度时序（Luminance）')
ax1.set_xlabel('时间(s)'); ax1.set_ylabel('亮度')
ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

# 2. 边缘亮度时序
ax2 = fig.add_subplot(4, 3, 2)
ax2.plot(t_swan, swan_edge, color=swan_color, alpha=0.8, lw=0.8, label='Swan EVT')
ax2.plot(t_gx, gx_edge, color=gx_color, alpha=0.8, lw=0.8, label='Galaxy XR')
ax2.set_title('文字边缘亮度时序（Edge Flicker）')
ax2.set_xlabel('时间(s)'); ax2.set_ylabel('边缘亮度')
ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

# 3. 锯齿比例时序
ax3 = fig.add_subplot(4, 3, 3)
ax3.plot(t_swan, swan_ali, color=swan_color, alpha=0.8, lw=0.8, label='Swan EVT')
ax3.plot(t_gx, gx_ali, color=gx_color, alpha=0.8, lw=0.8, label='Galaxy XR')
ax3.set_title('文字边缘锯齿比例时序（Aliasing Ratio）')
ax3.set_xlabel('时间(s)'); ax3.set_ylabel('斜向边缘比例(%)')
ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)

# 4. 逐帧亮度差
ax4 = fig.add_subplot(4, 3, 4)
swan_diffs = np.abs(np.diff(swan_lum))
gx_diffs = np.abs(np.diff(gx_lum))
ax4.plot(np.arange(len(swan_diffs))/fps_swan, swan_diffs, color=swan_color, alpha=0.8, lw=0.8, label='Swan EVT')
ax4.plot(np.arange(len(gx_diffs))/fps_gx, gx_diffs, color=gx_color, alpha=0.8, lw=0.8, label='Galaxy XR')
ax4.set_title('逐帧亮度差（Frame Diff）')
ax4.set_xlabel('时间(s)'); ax4.set_ylabel('帧间差')
ax4.legend(fontsize=8); ax4.grid(True, alpha=0.3)

# 5. 锯齿比例分布
ax5 = fig.add_subplot(4, 3, 5)
ax5.hist(swan_ali, bins=20, color=swan_color, alpha=0.6, label='Swan EVT', density=True)
ax5.hist(gx_ali, bins=20, color=gx_color, alpha=0.6, label='Galaxy XR', density=True)
ax5.set_title('锯齿比例分布（越集中越稳定）')
ax5.set_xlabel('斜向边缘比例(%)'); ax5.set_ylabel('密度')
ax5.legend(fontsize=8); ax5.grid(True, alpha=0.3)

# 6. 亮度分布
ax6 = fig.add_subplot(4, 3, 6)
ax6.hist(swan_lum, bins=30, color=swan_color, alpha=0.6, label='Swan EVT', density=True)
ax6.hist(gx_lum, bins=30, color=gx_color, alpha=0.6, label='Galaxy XR', density=True)
ax6.set_title('亮度分布（越集中越稳定）')
ax6.set_xlabel('亮度值'); ax6.set_ylabel('密度')
ax6.legend(fontsize=8); ax6.grid(True, alpha=0.3)

# 7. 整体指标柱状图（4维）
ax7 = fig.add_subplot(4, 3, (7, 9))
categories = ['Percent\nFlicker(%)', 'Edge\nFlicker(%)', 'Temporal\nContrast(%)',
              'Edge TC\n(%)', 'HF\nStability\nCV(%)', 'Aliasing\nRatio(%)',
              'Aliasing\nTemporal\nCV(%)', 'Edge\nTransition\nRatio', 'Edge\nSharpness\nCV(%)']
keys = ['percent_flicker', 'edge_percent_flicker', 'temporal_contrast',
        'edge_temporal_contrast', 'hf_stability_cv', 'aliasing_ratio_mean',
        'aliasing_temporal_cv', 'edge_transition_ratio', 'edge_sharpness_cv']
x = np.arange(len(keys))
w_bar = 0.35
swan_vals = [swan_m[k] for k in keys]
gx_vals = [gx_m[k] for k in keys]
bars1 = ax7.bar(x - w_bar/2, swan_vals, w_bar, label='Swan EVT', color=swan_color, alpha=0.8)
bars2 = ax7.bar(x + w_bar/2, gx_vals, w_bar, label='Galaxy XR', color=gx_color, alpha=0.8)
ax7.set_xticks(x); ax7.set_xticklabels(categories, fontsize=8)
ax7.set_ylabel('值（越低越稳定）')
ax7.set_title('全维度闪烁指标对比（越低越好）')
ax7.legend(fontsize=9); ax7.grid(True, alpha=0.3, axis='y')
for bar in bars1:
    h = bar.get_height()
    ax7.text(bar.get_x()+bar.get_width()/2, h+0.01, f'{h:.2f}',
             ha='center', va='bottom', fontsize=6.5, color=swan_color, rotation=45)
for bar in bars2:
    h = bar.get_height()
    ax7.text(bar.get_x()+bar.get_width()/2, h+0.01, f'{h:.2f}',
             ha='center', va='bottom', fontsize=6.5, color=gx_color, rotation=45)

# 8. 数据汇总表
ax8 = fig.add_subplot(4, 3, (10, 12))
ax8.axis('off')
table_data_rows = [
    ['指标', 'Swan EVT', 'Galaxy XR', '优胜'],
    ['帧数/FPS', f"{swan_m['n_frames']}f/{fps_swan:.0f}fps", f"{gx_m['n_frames']}f/{fps_gx:.0f}fps", '-'],
    ['平均亮度', f"{swan_m['mean_luminance']:.1f}", f"{gx_m['mean_luminance']:.1f}", '-'],
    ['Percent Flicker(%)', f"{swan_m['percent_flicker']:.4f}%", f"{gx_m['percent_flicker']:.4f}%",
     'GX ✓' if gx_m['percent_flicker'] < swan_m['percent_flicker'] else 'Swan ✓'],
    ['Edge Flicker(%)', f"{swan_m['edge_percent_flicker']:.4f}%", f"{gx_m['edge_percent_flicker']:.4f}%",
     'GX ✓' if gx_m['edge_percent_flicker'] < swan_m['edge_percent_flicker'] else 'Swan ✓'],
    ['Temporal Contrast(%)', f"{swan_m['temporal_contrast']:.4f}%", f"{gx_m['temporal_contrast']:.4f}%",
     'GX ✓' if gx_m['temporal_contrast'] < swan_m['temporal_contrast'] else 'Swan ✓'],
    ['HF Stability CV(%)', f"{swan_m['hf_stability_cv']:.4f}%", f"{gx_m['hf_stability_cv']:.4f}%",
     'Swan ✓' if swan_m['hf_stability_cv'] < gx_m['hf_stability_cv'] else 'GX ✓'],
    ['Aliasing Ratio(%)', f"{swan_m['aliasing_ratio_mean']:.4f}%", f"{gx_m['aliasing_ratio_mean']:.4f}%",
     'GX ✓' if gx_m['aliasing_ratio_mean'] < swan_m['aliasing_ratio_mean'] else 'Swan ✓'],
    ['Aliasing Temporal CV(%)', f"{swan_m['aliasing_temporal_cv']:.4f}%", f"{gx_m['aliasing_temporal_cv']:.4f}%",
     'GX ✓' if gx_m['aliasing_temporal_cv'] < swan_m['aliasing_temporal_cv'] else 'Swan ✓'],
    ['Edge Transition Ratio', f"{swan_m['edge_transition_ratio']:.4f}", f"{gx_m['edge_transition_ratio']:.4f}",
     'GX ✓' if gx_m['edge_transition_ratio'] < swan_m['edge_transition_ratio'] else 'Swan ✓'],
    ['Edge Sharpness CV(%)', f"{swan_m['edge_sharpness_cv']:.4f}%", f"{gx_m['edge_sharpness_cv']:.4f}%",
     'GX ✓' if gx_m['edge_sharpness_cv'] < swan_m['edge_sharpness_cv'] else 'Swan ✓'],
]
table = ax8.table(cellText=table_data_rows[1:], colLabels=table_data_rows[0],
                  cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(8.5)
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor('#37474F')
        cell.set_text_props(color='white', fontweight='bold')
    elif col == 3 and row > 0:
        text = cell.get_text().get_text()
        if 'GX' in text:
            cell.set_facecolor('#FFE0D0')
        elif 'Swan' in text:
            cell.set_facecolor('#E3F2FD')
    elif row % 2 == 0:
        cell.set_facecolor('#F5F5F5')
ax8.set_title('数据汇总', fontsize=11, fontweight='bold', pad=8)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('/home/ubuntu/roi_v2/zuikanzhi_v2_report.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: zuikanzhi_v2_report.png")

# 保存 JSON
for m in [swan_m, gx_m]:
    del m['luminance_series']
    del m['edge_series']
    del m['aliasing_series']
with open('/home/ubuntu/roi_v2/zuikanzhi_v2_metrics.json', 'w', encoding='utf-8') as f:
    json.dump({'swan': swan_m, 'gx': gx_m}, f, ensure_ascii=False, indent=2)
print("Saved: zuikanzhi_v2_metrics.json")
print("\n分析完成！")
