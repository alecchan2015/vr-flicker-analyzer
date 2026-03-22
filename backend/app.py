#!/usr/bin/env python3
"""
VR Flicker Analyzer - Flask API Server
"""

import os
import uuid
import threading
import json
import time
from pathlib import Path
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

# 确保 analyzer 可以被导入
import sys
sys.path.insert(0, os.path.dirname(__file__))
from analyzer import analyze_video, compare_videos, img_to_base64

app = Flask(__name__)
CORS(app)

# 配置
UPLOAD_FOLDER = '/tmp/vr_flicker_uploads'
OUTPUT_FOLDER = '/tmp/vr_flicker_outputs'
MAX_FILE_SIZE = 2 * 1024 * 1024 * 1024  # 2GB
ALLOWED_EXTENSIONS = {'mp4', 'mkv', 'avi', 'mov', 'webm'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# 任务状态存储
tasks = {}
tasks_lock = threading.Lock()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def update_task(task_id, **kwargs):
    with tasks_lock:
        if task_id in tasks:
            tasks[task_id].update(kwargs)


def run_single_analysis(task_id, video_path, device_name, output_dir):
    """后台线程：单视频分析"""
    try:
        def progress(p):
            update_task(task_id, progress=round(p * 100, 1))

        result = analyze_video(
            video_path=video_path,
            output_dir=output_dir,
            device_name=device_name,
            max_frames=300,
            progress_callback=progress
        )

        # 读取图表为 base64
        chart_b64 = img_to_base64(result['chart_path']) if os.path.exists(result['chart_path']) else None

        # 清理时序数据（不需要返回给前端）
        clean_result = {k: v for k, v in result.items() if not k.startswith('_') and k != 'chart_path' and k != 'json_path'}

        update_task(task_id,
                    status='done',
                    progress=100,
                    mode='single',
                    result=clean_result,
                    chart_b64=chart_b64)
    except Exception as e:
        update_task(task_id, status='error', error=str(e))
    finally:
        # 清理上传文件
        try:
            os.remove(video_path)
        except Exception:
            pass


def run_compare_analysis(task_id, video_a_path, video_b_path, name_a, name_b, output_dir):
    """后台线程：双视频对比分析"""
    try:
        def progress(p):
            update_task(task_id, progress=round(p * 100, 1))

        result = compare_videos(
            video_a_path=video_a_path,
            video_b_path=video_b_path,
            output_dir=output_dir,
            name_a=name_a,
            name_b=name_b,
            max_frames=300,
            progress_callback=progress
        )

        compare_b64 = img_to_base64(result['compare_chart_path']) if os.path.exists(result['compare_chart_path']) else None
        chart_a_path = result['device_a'].get('chart_path', '')
        chart_b_path = result['device_b'].get('chart_path', '')
        chart_a_b64 = img_to_base64(chart_a_path) if chart_a_path and os.path.exists(chart_a_path) else None
        chart_b_b64 = img_to_base64(chart_b_path) if chart_b_path and os.path.exists(chart_b_path) else None

        update_task(task_id,
                    status='done',
                    progress=100,
                    mode='compare',
                    result=result,
                    compare_chart_b64=compare_b64,
                    chart_a_b64=chart_a_b64,
                    chart_b_b64=chart_b_b64)
    except Exception as e:
        update_task(task_id, status='error', error=str(e))
    finally:
        for p in [video_a_path, video_b_path]:
            try:
                os.remove(p)
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# API 路由
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'version': '1.0.0'})


@app.route('/api/analyze/single', methods=['POST'])
def analyze_single():
    """单视频分析"""
    if 'video' not in request.files:
        return jsonify({'error': '未上传视频文件'}), 400

    video_file = request.files['video']
    device_name = request.form.get('device_name', 'Device').strip() or 'Device'

    if not video_file.filename or not allowed_file(video_file.filename):
        return jsonify({'error': f'不支持的文件格式，请上传: {", ".join(ALLOWED_EXTENSIONS)}'}), 400

    task_id = str(uuid.uuid4())
    filename = f"{task_id}_{secure_filename(video_file.filename)}"
    video_path = os.path.join(UPLOAD_FOLDER, filename)
    video_file.save(video_path)

    output_dir = os.path.join(OUTPUT_FOLDER, task_id)
    os.makedirs(output_dir, exist_ok=True)

    with tasks_lock:
        tasks[task_id] = {
            'status': 'running',
            'progress': 0,
            'mode': 'single',
            'created_at': time.time()
        }

    thread = threading.Thread(
        target=run_single_analysis,
        args=(task_id, video_path, device_name, output_dir),
        daemon=True
    )
    thread.start()

    return jsonify({'task_id': task_id})


@app.route('/api/analyze/compare', methods=['POST'])
def analyze_compare():
    """双视频对比分析"""
    if 'video_a' not in request.files or 'video_b' not in request.files:
        return jsonify({'error': '需要上传两个视频文件 (video_a, video_b)'}), 400

    video_a = request.files['video_a']
    video_b = request.files['video_b']
    name_a = request.form.get('name_a', 'Device A').strip() or 'Device A'
    name_b = request.form.get('name_b', 'Device B').strip() or 'Device B'

    for vf in [video_a, video_b]:
        if not vf.filename or not allowed_file(vf.filename):
            return jsonify({'error': f'不支持的文件格式: {vf.filename}'}), 400

    task_id = str(uuid.uuid4())
    path_a = os.path.join(UPLOAD_FOLDER, f"{task_id}_a_{secure_filename(video_a.filename)}")
    path_b = os.path.join(UPLOAD_FOLDER, f"{task_id}_b_{secure_filename(video_b.filename)}")
    video_a.save(path_a)
    video_b.save(path_b)

    output_dir = os.path.join(OUTPUT_FOLDER, task_id)
    os.makedirs(output_dir, exist_ok=True)

    with tasks_lock:
        tasks[task_id] = {
            'status': 'running',
            'progress': 0,
            'mode': 'compare',
            'created_at': time.time()
        }

    thread = threading.Thread(
        target=run_compare_analysis,
        args=(task_id, path_a, path_b, name_a, name_b, output_dir),
        daemon=True
    )
    thread.start()

    return jsonify({'task_id': task_id})


@app.route('/api/task/<task_id>', methods=['GET'])
def get_task(task_id):
    """查询任务状态和结果"""
    with tasks_lock:
        task = tasks.get(task_id)
    if not task:
        return jsonify({'error': '任务不存在'}), 404
    return jsonify(task)


@app.route('/api/task/<task_id>/chart', methods=['GET'])
def get_chart(task_id):
    """下载分析图表"""
    output_dir = os.path.join(OUTPUT_FOLDER, task_id)
    chart_path = os.path.join(output_dir, 'compare_chart.png')
    if not os.path.exists(chart_path):
        # 尝试单设备图表
        for f in os.listdir(output_dir):
            if f.startswith('chart_') and f.endswith('.png'):
                chart_path = os.path.join(output_dir, f)
                break
    if not os.path.exists(chart_path):
        return jsonify({'error': '图表不存在'}), 404
    return send_file(chart_path, mimetype='image/png')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=False)
