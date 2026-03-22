import { useState } from 'react'
import { SingleAnalysis } from './components/SingleAnalysis'
import { CompareAnalysis } from './components/CompareAnalysis'
import { ResultPanel } from './components/ResultPanel'

type Mode = 'single' | 'compare'
type AppState = 'idle' | 'analyzing' | 'done' | 'error'

export interface TaskResult {
  mode: 'single' | 'compare'
  result?: Record<string, unknown>
  chart_b64?: string
  device_a?: Record<string, unknown>
  device_b?: Record<string, unknown>
  compare_chart_b64?: string
  chart_a_b64?: string
  chart_b_b64?: string
  error?: string
}

const API_BASE = (import.meta.env.VITE_API_BASE as string) ?? ''

export default function App() {
  const [mode, setMode] = useState<Mode>('single')
  const [appState, setAppState] = useState<AppState>('idle')
  const [progress, setProgress] = useState(0)
  const [taskResult, setTaskResult] = useState<TaskResult | null>(null)

  const pollTask = (taskId: string, taskMode: Mode) => {
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE}/api/task/${taskId}`)
        const data = await res.json()
        setProgress(data.progress || 0)
        if (data.status === 'done') {
          clearInterval(interval)
          setTaskResult({ ...data, mode: taskMode } as TaskResult)
          setAppState('done')
        } else if (data.status === 'error') {
          clearInterval(interval)
          setTaskResult({ mode: taskMode, error: data.error })
          setAppState('error')
        }
      } catch {
        clearInterval(interval)
        setTaskResult({ mode: taskMode, error: '无法连接到分析服务器，请确认后端已启动' })
        setAppState('error')
      }
    }, 1500)
  }

  const handleSingleSubmit = async (video: File, deviceName: string) => {
    setAppState('analyzing')
    setProgress(0)
    setTaskResult(null)
    const form = new FormData()
    form.append('video', video)
    form.append('device_name', deviceName)
    try {
      const res = await fetch(`${API_BASE}/api/analyze/single`, { method: 'POST', body: form })
      const data = await res.json()
      if (data.task_id) pollTask(data.task_id, 'single')
      else throw new Error(data.error || '提交失败')
    } catch (e: unknown) {
      setTaskResult({ mode: 'single', error: String(e) })
      setAppState('error')
    }
  }

  const handleCompareSubmit = async (videoA: File, videoB: File, nameA: string, nameB: string) => {
    setAppState('analyzing')
    setProgress(0)
    setTaskResult(null)
    const form = new FormData()
    form.append('video_a', videoA)
    form.append('video_b', videoB)
    form.append('name_a', nameA)
    form.append('name_b', nameB)
    try {
      const res = await fetch(`${API_BASE}/api/analyze/compare`, { method: 'POST', body: form })
      const data = await res.json()
      if (data.task_id) pollTask(data.task_id, 'compare')
      else throw new Error(data.error || '提交失败')
    } catch (e: unknown) {
      setTaskResult({ mode: 'compare', error: String(e) })
      setAppState('error')
    }
  }

  const handleReset = () => {
    setAppState('idle')
    setProgress(0)
    setTaskResult(null)
  }

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100">
      {/* Header */}
      <header className="border-b border-slate-700 bg-slate-900/80 backdrop-blur sticky top-0 z-10">
        <div className="max-w-5xl mx-auto px-4 py-4 flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-violet-600 flex items-center justify-center text-white font-bold text-sm flex-shrink-0">VR</div>
          <div>
            <h1 className="text-lg font-bold text-white leading-tight">VR Flicker Analyzer</h1>
            <p className="text-xs text-slate-400">文字闪烁 · 边缘锯齿 · 高频纹理 全维度客观评估</p>
          </div>
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-4 py-8">
        {appState === 'idle' && (
          <>
            {/* Hero */}
            <div className="text-center mb-8">
              <h2 className="text-3xl font-bold text-white mb-3">VR 设备文字显示质量评估</h2>
              <p className="text-slate-400 max-w-2xl mx-auto">
                上传 scrcpy 录制的 VR 设备视频，自动分析文字区域的闪烁、锯齿和纹理稳定性，
                输出可量化的客观指标报告。支持单设备分析和双设备横向对比。
              </p>
            </div>

            {/* Mode Selector */}
            <div className="flex gap-2 mb-8 p-1 bg-slate-800 rounded-xl w-fit mx-auto">
              <button
                onClick={() => setMode('single')}
                className={`px-6 py-2.5 rounded-lg text-sm font-medium transition-all ${
                  mode === 'single'
                    ? 'bg-violet-600 text-white shadow'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                单平台分析
              </button>
              <button
                onClick={() => setMode('compare')}
                className={`px-6 py-2.5 rounded-lg text-sm font-medium transition-all ${
                  mode === 'compare'
                    ? 'bg-violet-600 text-white shadow'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                双平台对比
              </button>
            </div>

            {mode === 'single' ? (
              <SingleAnalysis onSubmit={handleSingleSubmit} />
            ) : (
              <CompareAnalysis onSubmit={handleCompareSubmit} />
            )}

            {/* Feature Cards */}
            <div className="mt-12 grid grid-cols-2 md:grid-cols-4 gap-4">
              {[
                { icon: '📊', title: '整体亮度闪烁', desc: 'Percent Flicker / Flicker Index / 主频分析' },
                { icon: '✏️', title: '文字边缘闪烁', desc: 'Canny 边缘提取 + 时域亮度波动评估' },
                { icon: '🔬', title: '高频纹理稳定性', desc: 'Laplacian 方差时序 / TAA 鬼影检测' },
                { icon: '🔲', title: '文字边缘锯齿', desc: '斜向边缘比例 / 过渡宽度 / 时域稳定性' },
              ].map(card => (
                <div key={card.title} className="bg-slate-800 rounded-xl p-4 border border-slate-700">
                  <div className="text-2xl mb-2">{card.icon}</div>
                  <div className="font-semibold text-sm text-white mb-1">{card.title}</div>
                  <div className="text-xs text-slate-400">{card.desc}</div>
                </div>
              ))}
            </div>
          </>
        )}

        {appState === 'analyzing' && (
          <div className="flex flex-col items-center justify-center py-24 gap-6">
            <div className="w-16 h-16 border-4 border-violet-600 border-t-transparent rounded-full animate-spin" />
            <div className="text-center">
              <p className="text-xl font-semibold text-white mb-2">正在分析中...</p>
              <p className="text-sm text-slate-400">正在提取帧数据并计算全维度闪烁指标，请耐心等待</p>
            </div>
            <div className="w-80 bg-slate-700 rounded-full h-2.5">
              <div
                className="bg-violet-600 h-2.5 rounded-full transition-all duration-500"
                style={{ width: `${progress}%` }}
              />
            </div>
            <p className="text-sm text-slate-400">{progress.toFixed(0)}%</p>
          </div>
        )}

        {(appState === 'done' || appState === 'error') && taskResult && (
          <ResultPanel result={taskResult} onReset={handleReset} />
        )}
      </main>

      <footer className="border-t border-slate-800 mt-16 py-6 text-center text-xs text-slate-500">
        VR Flicker Analyzer · 基于 IEEE 1789 / IES 闪烁标准 · 支持 MP4 / MKV / AVI / MOV / WebM
      </footer>
    </div>
  )
}
