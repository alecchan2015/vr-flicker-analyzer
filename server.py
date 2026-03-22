#!/usr/bin/env python3
"""
VR Flicker Analyzer - 一体化生产服务器
Flask 同时提供：
  - 前端静态文件 (React 构建产物)
  - 后端 REST API (/api/*)
端口：10010
"""

import os
import sys
import uuid
import threading
import time
from pathlib import Path
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# 将 backend 目录加入 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))
from analyzer import analyze_video, compare_videos, img_to_base64

# ─── 路径配置 ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / 'frontend' / 'dist'
UPLOAD_DIR = Path('/tmp/vr_uploads')
OUTPUT_DIR = Path('/tmp/vr_outputs')
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXT = {'mp4', 'mkv', 'avi', 'mov', 'webm'}
MAX_CONTENT_LENGTH = 2 * 1024 * 1024 * 1024  # 2 GB

# ─── Flask 应用 ───────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path='')
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
CORS(app)

# ─── 任务状态 ─────────────────────────────────────────────────────────────────
tasks: dict = {}
tasks_lock = threading.Lock()


def allowed(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT


def set_task(task_id: str, **kw):
    with tasks_lock:
        if task_id in tasks:
            tasks[task_id].update(kw)


# ─── 后台分析线程 ──────────────────────────────────────────────────────────────

def _run_single(task_id, video_path, device_name, out_dir):
    try:
        def prog(p):
            set_task(task_id, progress=round(p * 100, 1))

        result = analyze_video(
            video_path=video_path,
            output_dir=str(out_dir),
            device_name=device_name,
            max_frames=300,
            progress_callback=prog,
        )
        chart_b64 = img_to_base64(result['chart_path']) if os.path.exists(result.get('chart_path', '')) else None
        clean = {k: v for k, v in result.items() if k not in ('chart_path', 'json_path')}
        set_task(task_id, status='done', progress=100, mode='single',
                 result=clean, chart_b64=chart_b64)
    except Exception as e:
        set_task(task_id, status='error', error=str(e))
    finally:
        try:
            os.remove(video_path)
        except Exception:
            pass


def _run_compare(task_id, path_a, path_b, name_a, name_b, out_dir):
    try:
        def prog(p):
            set_task(task_id, progress=round(p * 100, 1))

        result = compare_videos(
            video_a_path=path_a,
            video_b_path=path_b,
            output_dir=str(out_dir),
            name_a=name_a,
            name_b=name_b,
            max_frames=300,
            progress_callback=prog,
        )
        cmp_b64 = img_to_base64(result['compare_chart_path']) if os.path.exists(result.get('compare_chart_path', '')) else None
        ca = img_to_base64(result['device_a'].get('chart_path', '')) if os.path.exists(result['device_a'].get('chart_path', '')) else None
        cb = img_to_base64(result['device_b'].get('chart_path', '')) if os.path.exists(result['device_b'].get('chart_path', '')) else None
        set_task(task_id, status='done', progress=100, mode='compare',
                 result=result, compare_chart_b64=cmp_b64,
                 chart_a_b64=ca, chart_b_b64=cb)
    except Exception as e:
        set_task(task_id, status='error', error=str(e))
    finally:
        for p in [path_a, path_b]:
            try:
                os.remove(p)
            except Exception:
                pass


# ─── API 路由 ─────────────────────────────────────────────────────────────────

@app.route('/api/health')
def health():
    return jsonify({'status': 'ok', 'version': '1.0.0'})


@app.route('/api/analyze/single', methods=['POST'])
def analyze_single():
    if 'video' not in request.files:
        return jsonify({'error': '未上传视频文件'}), 400
    f = request.files['video']
    device_name = request.form.get('device_name', 'Device').strip() or 'Device'
    if not f.filename or not allowed(f.filename):
        return jsonify({'error': f'不支持的格式，请上传: {", ".join(ALLOWED_EXT)}'}), 400

    task_id = str(uuid.uuid4())
    save_path = UPLOAD_DIR / f'{task_id}_{secure_filename(f.filename)}'
    f.save(str(save_path))
    out_dir = OUTPUT_DIR / task_id
    out_dir.mkdir(parents=True, exist_ok=True)

    with tasks_lock:
        tasks[task_id] = {'status': 'running', 'progress': 0, 'mode': 'single', 'created_at': time.time()}

    threading.Thread(target=_run_single, args=(task_id, str(save_path), device_name, out_dir), daemon=True).start()
    return jsonify({'task_id': task_id})


@app.route('/api/analyze/compare', methods=['POST'])
def analyze_compare():
    if 'video_a' not in request.files or 'video_b' not in request.files:
        return jsonify({'error': '需要上传两个视频 (video_a, video_b)'}), 400
    fa, fb = request.files['video_a'], request.files['video_b']
    name_a = request.form.get('name_a', 'Device A').strip() or 'Device A'
    name_b = request.form.get('name_b', 'Device B').strip() or 'Device B'
    for f in [fa, fb]:
        if not f.filename or not allowed(f.filename):
            return jsonify({'error': f'不支持的格式: {f.filename}'}), 400

    task_id = str(uuid.uuid4())
    pa = UPLOAD_DIR / f'{task_id}_a_{secure_filename(fa.filename)}'
    pb = UPLOAD_DIR / f'{task_id}_b_{secure_filename(fb.filename)}'
    fa.save(str(pa))
    fb.save(str(pb))
    out_dir = OUTPUT_DIR / task_id
    out_dir.mkdir(parents=True, exist_ok=True)

    with tasks_lock:
        tasks[task_id] = {'status': 'running', 'progress': 0, 'mode': 'compare', 'created_at': time.time()}

    threading.Thread(target=_run_compare, args=(task_id, str(pa), str(pb), name_a, name_b, out_dir), daemon=True).start()
    return jsonify({'task_id': task_id})


@app.route('/api/task/<task_id>')
def get_task(task_id):
    with tasks_lock:
        task = tasks.get(task_id)
    if not task:
        return jsonify({'error': '任务不存在'}), 404
    return jsonify(task)


# ─── 前端静态文件托管 ──────────────────────────────────────────────────────────

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    """所有非 API 请求都返回 React SPA 的 index.html"""
    if path and (STATIC_DIR / path).exists():
        return send_from_directory(str(STATIC_DIR), path)
    return send_from_directory(str(STATIC_DIR), 'index.html')


# ─── 启动 ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10010))
    print(f'[VR Flicker Analyzer] 启动于 http://0.0.0.0:{port}')
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
