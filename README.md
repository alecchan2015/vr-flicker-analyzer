# VR Flicker Analyzer

VR Flicker Analyzer 是一个用于客观评估 VR/XR 设备屏幕显示质量的自动化工具。它通过分析 scrcpy 录制的设备屏幕视频，量化评估文字闪烁、边缘锯齿以及高频纹理的稳定性。

本项目包含一个完整的 Web 应用（React + Flask），支持单设备分析和双设备横向对比，前后端一体化打包，**一条命令即可启动**。

## 核心评估指标

| 维度 | 指标 | 说明 |
|---|---|---|
| 整体亮度闪烁 | Percent Flicker (%) | 宏观亮度波动幅度，基于 IEEE 1789 标准 |
| 整体亮度闪烁 | Flicker Index | 亮度波动的能量占比，基于 IES 标准 |
| 文字边缘闪烁 | Edge Percent Flicker (%) | 文字边缘像素的亮度波动，评估 TAA 鬼影和重投影误差 |
| 高频纹理稳定性 | HF Stability CV (%) | Laplacian 方差时域变异系数，量化帧间细节抖动 |
| 文字边缘锯齿 | Aliasing Ratio (%) | 斜向边缘比例，反映锯齿感 |
| 文字边缘锯齿 | Edge Transition Ratio | 边缘过渡带宽度，反映文字锐利度 |

## 快速部署（一体化模式）

前后端合并为单一服务，无需分别启动。

### 1. 克隆仓库

```bash
git clone https://github.com/alecchan2015/vr-flicker-analyzer.git
cd vr-flicker-analyzer
```

### 2. 安装依赖

```bash
# Python 后端依赖
pip install -r backend/requirements.txt

# 前端依赖（构建产物已包含在 frontend/dist，通常无需重新构建）
# 如需重新构建前端：
# cd frontend && pnpm install && pnpm build && cd ..
```

### 3. 启动服务

```bash
# 默认端口 10010
python server.py

# 或指定端口
PORT=8080 python server.py
```

打开浏览器访问 `http://localhost:10010` 即可使用。

## 分开部署（开发模式）

如需分别启动前后端进行开发调试：

```bash
# 后端（端口 5050）
cd backend && python app.py

# 前端（端口 5173，新终端）
cd frontend && pnpm install && pnpm dev
```

## 使用说明

1. 打开浏览器访问 Web 界面。
2. **单平台分析**：上传一段 VR 设备的 scrcpy 录制视频（建议 3~30 秒，包含静态文字 UI 面板）。
3. **双平台对比**：上传两段不同设备的视频，系统将自动生成横向对比报告。
4. **双目视频支持**：如果上传的是双目拼接视频（宽度 > 高度 1.5 倍），系统会自动提取右眼视角进行分析。

## 适用场景

- VR 渲染管线优化（TAA、FSR、重投影算法调优）
- 屏幕硬件选型与对比（Fast-LCD vs OLEDoS）
- 文本阅读体验客观评估

## 许可证

本项目采用 MIT 许可证。
