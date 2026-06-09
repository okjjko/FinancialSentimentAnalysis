import ResultLabel from "./ResultLabel"

export default function DetailPanel({ entry, onBack }) {
  if (!entry) return null

  return (
    <div className="detail-panel card">
      <button className="btn btn-secondary btn-sm" onClick={onBack}>← 返回</button>
      <div className="detail-top" style={{ marginTop: 16 }}>
        <div className="detail-text-wrap">
          <pre>{entry.text}</pre>
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <ResultLabel results={entry.results} />
          <div className="detail-meta">
            <div>🕐 {entry.timestamp}</div>
            <div>📁 来源: {entry.source}</div>
          </div>
        </div>
      </div>
    </div>
  )
}
