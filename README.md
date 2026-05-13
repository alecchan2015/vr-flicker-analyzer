# VR Flicker Analyzer

VR Flicker Analyzer 是一个用于客观评估 VR/XR 设备屏幕显示质量的自动化工具。它通过分析设备屏幕视频，量化评估文字闪烁、边缘锯齿、高频纹理稳定性和画面撕裂。

本项目包含三种使用形态：
1. **Web 应用** (React + Flask) — 上传视频做单设备分析或双设备对比
2. **本地 Agent (CLI)** — 直接连接 Android/VR 设备录屏并自动跑分析
3. **REST API** — 程序化集成

## 核心评估指标

| 维度 | 指标 | 说明 |
|---|---|---|
| 整体亮度闪烁 | Percent Flicker (%) | 宏观亮度波动幅度，基于 IEEE 1789 标准 |
| 整体亮度闪烁 | Flicker Index | 亮度波动的能量占比，基于 IES 标准 |
| 文字边缘闪烁 | Edge Percent Flicker (%) | 文字边缘像素的亮度波动，评估 TAA 鬼影和重投影误差 |
| 高频纹理稳定性 | HF Stability CV (%) | Laplacian 方差时域变异系数，量化帧间细节抖动 |
| 文字边缘锯齿 | Aliasing Ratio (%) | 斜向边缘比例，反映锯齿感 |
| 文字边缘锯齿 | Edge Transition Ratio | 边缘过渡带宽度，反映文字锐利度 |
| **画面撕裂** | **Tearing Score** | **行级稠密光流检测水平接缝，量化 vsync 失效程度** |

### 严重程度分级（统一 5 档）

每项指标和综合评分都会映射到 `Excellent / Good / Moderate / Severe / Critical`。阈值参考：

| 指标 | Excellent | Good | Moderate | Severe | Critical |
|---|---|---|---|---|---|
| Percent Flicker (%)       | <8   | <20  | <40  | <70  | ≥70 |
| Edge Flicker (%)          | <5   | <12  | <25  | <45  | ≥45 |
| HF Stability CV (%)       | <2   | <5   | <10  | <20  | ≥20 |
| Aliasing Ratio (%)        | <8   | <15  | <25  | <40  | ≥40 |
| Tearing Score             | <0.5 | <2   | <5   | <10  | ≥10 |

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

## 本地 Agent（直接从设备录屏）

不想手动录视频再上传？把设备 USB 接上电脑，让 Agent 自动完成「录屏 → 分析 → 出报告」。

### macOS 环境准备

```bash
# 1. 安装 adb + scrcpy
brew install android-platform-tools scrcpy

# 2. Python 依赖
pip install -r backend/requirements.txt

# 3. 设备侧
#   - Android 手机：设置 → 关于本机 → 连点 7 次版本号 → 开发者选项 → 开启 USB 调试
#   - PICO / Quest：开发者模式开启后，USB 调试授权
#   - 首次插上电脑时设备会弹「允许调试」对话框，勾选「始终允许」
```

### CLI 用法

```bash
# 列出已连接的设备
python -m agent list

# 录制 10 秒并自动分析，报告输出到 ./report
python -m agent capture --duration 10 --out ./report

# 多设备时指定 serial、自定义参数
python -m agent capture --device <serial> --duration 15 \
    --max-size 1920 --bit-rate 12M --fps 60 \
    --name "PICO4-Ultra" --out ./report

# 跑已有视频（不录屏）
python -m agent analyze /path/to/clip.mp4 --name "Quest3" --out ./report
```

CLI 会在终端打印一份带颜色的报告（5 项指标 + 综合评级），同时把 PNG 图表和 JSON 数据落到 `--out` 目录。

### Web UI 直采

Web 服务启动后还会暴露以下 REST 端点：

| 方法 | 路径 | 说明 |
|---|---|---|
| GET  | `/api/devices`       | 列出 ADB 设备 |
| POST | `/api/capture/start` | 触发录制+分析，返回 `task_id`（body: `serial`, `duration`, `max_size`, `bit_rate`, `fps`, `name`） |
| GET  | `/api/task/<id>`     | 轮询任务进度与结果（含 `phase`: `capturing`/`analyzing`/`done`） |

## 适用场景

- VR 渲染管线优化（TAA、FSR、重投影算法调优）
- 屏幕硬件选型与对比（Fast-LCD vs OLEDoS）
- 文本阅读体验客观评估
- vsync / 帧调度异常排查（撕裂检测）

## 许可证

本项目采用 MIT 许可证。
