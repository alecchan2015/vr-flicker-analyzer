import { useState, useRef } from 'react'
import type { DragEvent, ChangeEvent } from 'react'

interface Props {
  onSubmit: (video: File, deviceName: string) => void
}

export function SingleAnalysis({ onSubmit }: Props) {
  const [video, setVideo] = useState<File | null>(null)
  const [deviceName, setDeviceName] = useState('')
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const handleFile = (file: File) => {
    if (file && /\.(mp4|mkv|avi|mov|webm)$/i.test(file.name)) {
      setVideo(file)
    }
  }

  const onDrop = (e: DragEvent) => {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files[0]
    if (file) handleFile(file)
  }

  const onFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) handleFile(file)
  }

  const formatSize = (bytes: number) => {
    if (bytes > 1024 * 1024 * 1024) return `${(bytes / 1024 / 1024 / 1024).toFixed(1)} GB`
    if (bytes > 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`
    return `${(bytes / 1024).toFixed(0)} KB`
  }

  return (
    <div className="max-w-2xl mx-auto">
      <div
        className={`border-2 border-dashed rounded-2xl p-10 text-center cursor-pointer transition-all ${
          dragging
            ? 'border-violet-500 bg-violet-500/10'
            : video
            ? 'border-green-500 bg-green-500/10'
            : 'border-slate-600 hover:border-slate-500 bg-slate-800/50'
        }`}
        onDragOver={e => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".mp4,.mkv,.avi,.mov,.webm"
          className="hidden"
          onChange={onFileChange}
        />
        {video ? (
          <div>
            <div className="text-4xl mb-3">🎬</div>
            <p className="font-semibold text-green-400 text-lg">{video.name}</p>
            <p className="text-slate-400 text-sm mt-1">{formatSize(video.size)}</p>
            <p className="text-slate-500 text-xs mt-2">点击重新选择</p>
          </div>
        ) : (
          <div>
            <div className="text-5xl mb-4">📹</div>
            <p className="text-slate-300 font-medium text-lg mb-1">拖拽视频文件到此处</p>
            <p className="text-slate-500 text-sm">或点击选择文件</p>
            <p className="text-slate-600 text-xs mt-3">支持 MP4 / MKV / AVI / MOV / WebM</p>
          </div>
        )}
      </div>

      <div className="mt-5 space-y-4">
        <div>
          <label className="block text-sm font-medium text-slate-300 mb-1.5">设备名称（可选）</label>
          <input
            type="text"
            value={deviceName}
            onChange={e => setDeviceName(e.target.value)}
            placeholder="例如：Galaxy XR、Pico 4 Ultra..."
            className="w-full bg-slate-800 border border-slate-600 rounded-xl px-4 py-2.5 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-violet-500 transition-colors"
          />
        </div>

        <button
          disabled={!video}
          onClick={() => video && onSubmit(video, deviceName || '设备')}
          className="w-full py-3 rounded-xl font-semibold text-sm transition-all disabled:opacity-40 disabled:cursor-not-allowed bg-violet-600 hover:bg-violet-500 text-white"
        >
          开始分析
        </button>
      </div>

      <div className="mt-6 bg-slate-800/50 rounded-xl p-4 border border-slate-700">
        <p className="text-xs font-semibold text-slate-400 mb-2">💡 使用说明</p>
        <ul className="text-xs text-slate-500 space-y-1">
          <li>• 支持 scrcpy 录制的 VR 设备屏幕视频（单目或双目拼接格式）</li>
          <li>• 双目拼接视频（宽度 &gt; 高度 1.5 倍）将自动提取右眼视角分析</li>
          <li>• 建议视频时长 3~30 秒，包含静态文字 UI 面板</li>
          <li>• 分析时间约为视频时长的 2~5 倍</li>
        </ul>
      </div>
    </div>
  )
}
