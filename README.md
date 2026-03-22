# VR Flicker Analyzer

VR Flicker Analyzer 是一个用于客观评估 VR/XR 设备屏幕显示质量的自动化工具。它通过分析 scrcpy 录制的设备屏幕视频，量化评估文字闪烁、边缘锯齿以及高频纹理的稳定性。

本项目包含一个完整的 Web 应用（React + Flask），支持单设备分析和双设备横向对比。

## 核心评估指标

1. **整体亮度闪烁 (Global Flicker)**
   - **Percent Flicker (%)**: 宏观亮度波动幅度（基于 IEEE 1789 标准）。
   - **Flicker Index**: 亮度波动的能量占比。
   - **主频 (Hz)**: 闪烁的主要频率。

2. **文字边缘闪烁 (Edge Flicker)**
   - **Edge Percent Flicker (%)**: 专门针对高对比度文字边缘像素的亮度波动幅度。
   - 评估亚像素级别的渲染抖动（如 TAA 鬼影、重投影误差）。

3. **高频纹理稳定性 (HF Texture Stability)**
   - **HF Stability CV (%)**: Laplacian 方差的时域变异系数。
   - 量化高频细节（如细小文字、复杂纹理）在帧间的抖动程度。

4. **文字边缘锯齿 (Text Aliasing)**
   - **Aliasing Ratio (%)**: 边缘中斜向像素的比例，反映锯齿感。
   - **Edge Transition Ratio**: 边缘过渡带宽度，反映文字的锐利度。

## 快速开始

### 1. 启动后端服务 (Flask)

后端服务负责处理视频分析任务。

```bash
cd backend
pip install -r requirements.txt
python app.py
```
后端服务默认运行在 `http://localhost:5050`。

### 2. 启动前端服务 (React + Vite)

前端服务提供用户交互界面。

```bash
cd frontend
pnpm install
pnpm dev
```
前端服务默认运行在 `http://localhost:5173`。

## 使用说明

1. 打开浏览器访问前端页面。
2. **单平台分析**：上传一段 VR 设备的 scrcpy 录制视频（建议 3~30 秒，包含静态文字 UI 面板）。
3. **双平台对比**：上传两段不同设备的视频，系统将自动生成横向对比报告。
4. **双目视频支持**：如果上传的是双目拼接视频（宽度 > 高度 1.5 倍），系统会自动提取右眼视角进行分析。

## 适用场景

- VR 渲染管线优化（TAA、FSR、重投影算法调优）
- 屏幕硬件选型与对比（Fast-LCD vs OLEDoS）
- 文本阅读体验客观评估

## 许可证

本项目采用 MIT 许可证。
