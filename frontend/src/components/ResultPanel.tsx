import type { TaskResult } from '../App'

interface Props {
  result: TaskResult
  onReset: () => void
}

function scoreColor(score: number) {
  if (score < 10) return 'text-green-400'
  if (score < 25) return 'text-lime-400'
  if (score < 45) return 'text-yellow-400'
  if (score < 65) return 'text-orange-400'
  return 'text-red-400'
}

function scoreBg(score: number) {
  if (score < 10) return 'bg-green-500/20 border-green-500/40'
  if (score < 25) return 'bg-lime-500/20 border-lime-500/40'
  if (score < 45) return 'bg-yellow-500/20 border-yellow-500/40'
  if (score < 65) return 'bg-orange-500/20 border-orange-500/40'
  return 'bg-red-500/20 border-red-500/40'
}

function MetricRow({ label, value, unit = '', highlight = false }: {
  label: string; value: string | number; unit?: string; highlight?: boolean
}) {
  return (
    <div className={`flex justify-between items-center py-2 px-3 rounded-lg ${highlight ? 'bg-slate-700/50' : ''}`}>
      <span className="text-xs text-slate-400">{label}</span>
      <span className="text-sm font-mono font-medium text-slate-200">{value}{unit}</span>
    </div>
  )
}

function ScoreCard({ title, score, label }: { title: string; score: number; label: string }) {
  return (
    <div className={`rounded-xl p-4 border ${scoreBg(score)} text-center`}>
      <p className="text-xs text-slate-400 mb-1">{title}</p>
      <p className={`text-3xl font-bold ${scoreColor(score)}`}>{score.toFixed(1)}</p>
      <p className={`text-xs mt-1 font-medium ${scoreColor(score)}`}>{label}</p>
    </div>
  )
}

function SingleResult({ result, chartB64 }: { result: Record<string, unknown>; chartB64?: string }) {
  const r = result as Record<string, number & string>
  return (
    <div>
      {/* Score Cards */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        <ScoreCard title="整体闪烁评分" score={r.flicker_score} label={r.flicker_label} />
        <ScoreCard title="边缘闪烁评分" score={r.edge_flicker_score} label={r.edge_flicker_label} />
        <ScoreCard title="HF 稳定性评分" score={r.hf_stability_score} label={r.hf_stability_label} />
      </div>

      {/* Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        <div className="bg-slate-800 rounded-xl p-4 border border-slate-700">
          <h4 className="text-sm font-semibold text-slate-300 mb-3">📊 整体亮度闪烁</h4>
          <MetricRow label="Percent Flicker" value={(r.percent_flicker as unknown as number).toFixed(4)} unit="%" highlight />
          <MetricRow label="Flicker Index" value={(r.flicker_index as unknown as number).toFixed(4)} />
          <MetricRow label="Temporal Contrast" value={(r.temporal_contrast as unknown as number).toFixed(4)} highlight />
          <MetricRow label="Frame Diff Mean" value={(r.frame_diff_mean as unknown as number).toFixed(4)} />
          <MetricRow label="主频" value={(r.dominant_freq_hz as unknown as number).toFixed(1)} unit=" Hz" highlight />
        </div>

        <div className="bg-slate-800 rounded-xl p-4 border border-slate-700">
          <h4 className="text-sm font-semibold text-slate-300 mb-3">✏️ 文字边缘闪烁</h4>
          <MetricRow label="Edge Percent Flicker" value={(r.edge_percent_flicker as unknown as number).toFixed(4)} unit="%" highlight />
          <MetricRow label="Edge Temporal Contrast" value={(r.edge_temporal_contrast as unknown as number).toFixed(4)} />
          <MetricRow label="Edge Frame Diff Mean" value={(r.edge_frame_diff_mean as unknown as number).toFixed(4)} highlight />
        </div>

        <div className="bg-slate-800 rounded-xl p-4 border border-slate-700">
          <h4 className="text-sm font-semibold text-slate-300 mb-3">🔬 高频纹理稳定性</h4>
          <MetricRow label="HF Stability CV" value={(r.hf_stability_cv as unknown as number).toFixed(2)} unit="%" highlight />
        </div>

        <div className="bg-slate-800 rounded-xl p-4 border border-slate-700">
          <h4 className="text-sm font-semibold text-slate-300 mb-3">🔲 文字边缘锯齿</h4>
          <MetricRow label="Aliasing Ratio Mean" value={(r.aliasing_ratio_mean as unknown as number).toFixed(2)} unit="%" highlight />
          <MetricRow label="Aliasing Temporal CV" value={(r.aliasing_temporal_cv as unknown as number).toFixed(2)} unit="%" />
          <MetricRow label="Edge Transition Ratio" value={(r.edge_transition_ratio as unknown as number).toFixed(2)} highlight />
          <MetricRow label="Edge Sharpness CV" value={(r.edge_sharpness_cv as unknown as number).toFixed(2)} unit="%" />
        </div>
      </div>

      {/* Chart */}
      {chartB64 && (
        <div className="bg-slate-800 rounded-xl p-4 border border-slate-700">
          <h4 className="text-sm font-semibold text-slate-300 mb-3">📈 可视化报告</h4>
          <img src={`data:image/png;base64,${chartB64}`} alt="分析图表" className="w-full rounded-lg" />
        </div>
      )}
    </div>
  )
}

function CompareResult({ result }: { result: TaskResult }) {
  const ra = result.device_a as Record<string, unknown>
  const rb = result.device_b as Record<string, unknown>
  if (!ra || !rb) return null

  const nameA = ra.device as string
  const nameB = rb.device as string

  const metrics = [
    { key: 'percent_flicker', label: 'Percent Flicker', unit: '%', lower_better: true },
    { key: 'edge_percent_flicker', label: 'Edge Flicker', unit: '%', lower_better: true },
    { key: 'temporal_contrast', label: 'Temporal Contrast', unit: '', lower_better: true },
    { key: 'edge_temporal_contrast', label: 'Edge TC', unit: '', lower_better: true },
    { key: 'hf_stability_cv', label: 'HF Stability CV', unit: '%', lower_better: true },
    { key: 'aliasing_ratio_mean', label: 'Aliasing Ratio', unit: '%', lower_better: true },
    { key: 'aliasing_temporal_cv', label: 'Aliasing CV', unit: '%', lower_better: true },
    { key: 'edge_transition_ratio', label: 'Edge Transition', unit: '', lower_better: true },
    { key: 'flicker_score', label: '综合闪烁评分', unit: '', lower_better: true },
    { key: 'edge_flicker_score', label: '边缘闪烁评分', unit: '', lower_better: true },
    { key: 'hf_stability_score', label: 'HF 稳定性评分', unit: '', lower_better: true },
  ]

  return (
    <div>
      {/* Score comparison */}
      <div className="grid grid-cols-2 gap-6 mb-6">
        <div className="bg-slate-800 rounded-xl p-5 border border-blue-500/30">
          <div className="flex items-center gap-2 mb-4">
            <div className="w-3 h-3 rounded-full bg-blue-500" />
            <h3 className="font-bold text-blue-300">{nameA}</h3>
          </div>
          <div className="grid grid-cols-3 gap-3">
            <ScoreCard title="整体闪烁" score={ra.flicker_score as number} label={ra.flicker_label as string} />
            <ScoreCard title="边缘闪烁" score={ra.edge_flicker_score as number} label={ra.edge_flicker_label as string} />
            <ScoreCard title="HF稳定性" score={ra.hf_stability_score as number} label={ra.hf_stability_label as string} />
          </div>
        </div>
        <div className="bg-slate-800 rounded-xl p-5 border border-orange-500/30">
          <div className="flex items-center gap-2 mb-4">
            <div className="w-3 h-3 rounded-full bg-orange-500" />
            <h3 className="font-bold text-orange-300">{nameB}</h3>
          </div>
          <div className="grid grid-cols-3 gap-3">
            <ScoreCard title="整体闪烁" score={rb.flicker_score as number} label={rb.flicker_label as string} />
            <ScoreCard title="边缘闪烁" score={rb.edge_flicker_score as number} label={rb.edge_flicker_label as string} />
            <ScoreCard title="HF稳定性" score={rb.hf_stability_score as number} label={rb.hf_stability_label as string} />
          </div>
        </div>
      </div>

      {/* Comparison Table */}
      <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden mb-6">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-slate-700/80">
              <th className="text-left px-4 py-3 text-slate-300 font-semibold">指标</th>
              <th className="text-center px-4 py-3 text-blue-300 font-semibold">{nameA}</th>
              <th className="text-center px-4 py-3 text-orange-300 font-semibold">{nameB}</th>
              <th className="text-center px-4 py-3 text-slate-300 font-semibold">优胜</th>
            </tr>
          </thead>
          <tbody>
            {metrics.map((m, i) => {
              const va = ra[m.key] as number
              const vb = rb[m.key] as number
              const aWins = m.lower_better ? va < vb : va > vb
              return (
                <tr key={m.key} className={i % 2 === 0 ? 'bg-slate-800' : 'bg-slate-800/50'}>
                  <td className="px-4 py-2.5 text-slate-400">{m.label}</td>
                  <td className={`px-4 py-2.5 text-center font-mono ${aWins ? 'text-blue-300 font-semibold' : 'text-slate-400'}`}>
                    {va?.toFixed(4)}{m.unit}
                  </td>
                  <td className={`px-4 py-2.5 text-center font-mono ${!aWins ? 'text-orange-300 font-semibold' : 'text-slate-400'}`}>
                    {vb?.toFixed(4)}{m.unit}
                  </td>
                  <td className="px-4 py-2.5 text-center">
                    <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${
                      aWins ? 'bg-blue-500/20 text-blue-300' : 'bg-orange-500/20 text-orange-300'
                    }`}>
                      {aWins ? nameA : nameB} ✓
                    </span>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      {/* Compare Chart */}
      {result.compare_chart_b64 && (
        <div className="bg-slate-800 rounded-xl p-4 border border-slate-700 mb-6">
          <h4 className="text-sm font-semibold text-slate-300 mb-3">📈 对比可视化报告</h4>
          <img src={`data:image/png;base64,${result.compare_chart_b64}`} alt="对比图表" className="w-full rounded-lg" />
        </div>
      )}

      {/* Individual Charts */}
      {(result.chart_a_b64 || result.chart_b_b64) && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {result.chart_a_b64 && (
            <div className="bg-slate-800 rounded-xl p-4 border border-blue-500/30">
              <h4 className="text-sm font-semibold text-blue-300 mb-3">{nameA} 详细报告</h4>
              <img src={`data:image/png;base64,${result.chart_a_b64}`} alt={`${nameA} 图表`} className="w-full rounded-lg" />
            </div>
          )}
          {result.chart_b_b64 && (
            <div className="bg-slate-800 rounded-xl p-4 border border-orange-500/30">
              <h4 className="text-sm font-semibold text-orange-300 mb-3">{nameB} 详细报告</h4>
              <img src={`data:image/png;base64,${result.chart_b_b64}`} alt={`${nameB} 图表`} className="w-full rounded-lg" />
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export function ResultPanel({ result, onReset }: Props) {
  if (result.error) {
    return (
      <div className="text-center py-16">
        <div className="text-5xl mb-4">❌</div>
        <h3 className="text-xl font-bold text-red-400 mb-2">分析失败</h3>
        <p className="text-slate-400 text-sm mb-6 max-w-md mx-auto">{result.error}</p>
        <button
          onClick={onReset}
          className="px-6 py-2.5 bg-slate-700 hover:bg-slate-600 rounded-xl text-sm font-medium transition-colors"
        >
          重新开始
        </button>
      </div>
    )
  }

  return (
    <div>
      {/* Result Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-xl font-bold text-white">
            {result.mode === 'compare' ? '双平台对比分析结果' : '单平台分析结果'}
          </h2>
          <p className="text-sm text-slate-400 mt-0.5">
            {result.mode === 'compare'
              ? `${(result.device_a as Record<string, unknown>)?.device} vs ${(result.device_b as Record<string, unknown>)?.device}`
              : (result.result as Record<string, unknown>)?.device as string}
          </p>
        </div>
        <button
          onClick={onReset}
          className="px-4 py-2 bg-slate-700 hover:bg-slate-600 rounded-xl text-sm font-medium transition-colors"
        >
          重新分析
        </button>
      </div>

      {result.mode === 'single' && result.result && (
        <SingleResult result={result.result} chartB64={result.chart_b64} />
      )}
      {result.mode === 'compare' && (
        <CompareResult result={result} />
      )}
    </div>
  )
}
