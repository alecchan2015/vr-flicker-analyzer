import { useState, useRef } from 'react'
import type { DragEvent, ChangeEvent } from 'react'

interface Props {
  onSubmit: (videoA: File, videoB: File, nameA: string, nameB: string) => void
}

interface VideoSlot {
  file: File | null
  name: string
}

function VideoDropZone({
  slot, label, color, onChange
}: {
  slot: VideoSlot
  label: string
  color: 'blue' | 'orange'
  onChange: (file: File, name: string) => void
}) {
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const handleFile = (file: File) => {
    if (/\.(mp4|mkv|avi|mov|webm)$/i.test(file.name)) {
      onChange(file, slot.name)
    }
  }

  const onDrop = (e: DragEvent) => {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files[0]
    if (file) handleFile(file)
  }

  const formatSize = (bytes: number) => {
    if (bytes > 1024 * 1024 * 1024) return `${(bytes / 1024 / 1024 / 1024).toFixed(1)} GB`
    if (bytes > 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`
    return `${(bytes / 1024).toFixed(0)} KB`
  }

  const borderColor = color === 'blue'
    ? (dragging ? 'border-blue-500 bg-blue-500/10' : slot.file ? 'border-blue-400 bg-blue-500/10' : 'border-slate-600 hover:border-blue-500/50')
    : (dragging ? 'border-orange-500 bg-orange-500/10' : slot.file ? 'border-orange-400 bg-orange-500/10' : 'border-slate-600 hover:border-orange-500/50')

  return (
    <div className="flex-1">
      <div
        className={`border-2 border-dashed rounded-2xl p-6 text-center cursor-pointer transition-all bg-slate-800/50 ${borderColor}`}
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
          onChange={(e: ChangeEvent<HTMLInputElement>) => {
            const file = e.target.files?.[0]
            if (file) handleFile(file)
          }}
        />
        <div className={`text-xs font-bold mb-2 px-2 py-0.5 rounded-full w-fit mx-auto ${
          color === 'blue' ? 'bg-blue-600/30 text-blue-400' : 'bg-orange-600/30 text-orange-400'
        }`}>{label}</div>
        {slot.file ? (
          <div>
            <div className="text-2xl mb-1">🎬</div>
            <p className={`font-medium text-sm truncate max-w-full ${color === 'blue' ? 'text-blue-300' : 'text-orange-300'}`}>
              {slot.file.name}
            </p>
            <p className="text-slate-500 text-xs mt-0.5">{formatSize(slot.file.size)}</p>
          </div>
        ) : (
          <div>
            <div className="text-3xl mb-2">📹</div>
            <p className="text-slate-400 text-sm">拖拽或点击上传</p>
            <p className="text-slate-600 text-xs mt-1">MP4 / MKV / AVI / MOV</p>
          </div>
        )}
      </div>
      <input
        type="text"
        value={slot.name}
        onChange={e => onChange(slot.file!, e.target.value)}
        placeholder={`设备名称（如 ${label}）`}
        className="mt-2 w-full bg-slate-800 border border-slate-600 rounded-xl px-3 py-2 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-violet-500 transition-colors"
      />
    </div>
  )
}

export function CompareAnalysis({ onSubmit }: Props) {
  const [slotA, setSlotA] = useState<VideoSlot>({ file: null, name: 'Device A' })
  const [slotB, setSlotB] = useState<VideoSlot>({ file: null, name: 'Device B' })

  const canSubmit = slotA.file && slotB.file

  return (
    <div className="max-w-3xl mx-auto">
      <div className="flex gap-4">
        <VideoDropZone
          slot={slotA}
          label="设备 A"
          color="blue"
          onChange={(file, name) => setSlotA({ file: file || slotA.file, name })}
        />
        <div className="flex items-center justify-center">
          <div className="w-8 h-8 rounded-full bg-slate-700 flex items-center justify-center text-slate-400 text-sm font-bold flex-shrink-0">
            VS
          </div>
        </div>
        <VideoDropZone
          slot={slotB}
          label="设备 B"
          color="orange"
          onChange={(file, name) => setSlotB({ file: file || slotB.file, name })}
        />
      </div>

      <button
        disabled={!canSubmit}
        onClick={() => canSubmit && onSubmit(slotA.file!, slotB.file!, slotA.name, slotB.name)}
        className="mt-6 w-full py-3 rounded-xl font-semibold text-sm transition-all disabled:opacity-40 disabled:cursor-not-allowed bg-violet-600 hover:bg-violet-500 text-white"
      >
        开始对比分析
      </button>

      <div className="mt-5 bg-slate-800/50 rounded-xl p-4 border border-slate-700">
        <p className="text-xs font-semibold text-slate-400 mb-2">💡 对比分析说明</p>
        <ul className="text-xs text-slate-500 space-y-1">
          <li>• 两台设备的视频将分别进行全维度分析，然后生成横向对比报告</li>
          <li>• 建议两段视频内容相同（同一应用面板的录制），以确保对比的公平性</li>
          <li>• 系统将自动识别双目拼接格式并提取右眼视角</li>
          <li>• 对比分析时间约为单设备分析的 2 倍</li>
        </ul>
      </div>
    </div>
  )
}
