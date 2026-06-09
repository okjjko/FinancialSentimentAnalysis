import { useEffect, useState, useRef } from "react"
import { getExamples, predictSingle } from "../api"
import ResultLabel from "./ResultLabel"

export default function SingleTab({ onPredictionDone }) {
  const [examples, setExamples] = useState([])
  const [text, setText] = useState("")
  const [predicting, setPredicting] = useState(false)
  const [result, setResult] = useState(null)
  const textareaRef = useRef(null)

  useEffect(() => {
    getExamples().then((data) => setExamples(data.examples || []))
  }, [])

  const handlePredict = async () => {
    if (!text.trim()) return
    setPredicting(true)
    try {
      const data = await predictSingle(text)
      setResult(data.results)
      onPredictionDone()
    } catch (err) {
      setResult([{ class: `错误: ${err.message}`, confidence: 1 }])
    }
    setPredicting(false)
  }

  const handleClear = () => {
    setText("")
    setResult(null)
  }

  const handleExampleClick = (exText) => {
    setText(exText)
    setResult(null)
    textareaRef.current?.focus()
  }

  return (
    <div className="row">
      <div className="col col-4">
        <div className="card">
          <h3>输入文本</h3>
          <div className="text-input-area">
            <textarea ref={textareaRef}
              value={text}
              onChange={(e) => { setText(e.target.value); setResult(null) }}
              placeholder="输入英文新闻或评论..." />
          </div>

          <div className="btn-group">
            <button className="btn btn-primary" disabled={!text.trim() || predicting} onClick={handlePredict}>
              {predicting ? <span className="loading" /> : "开始分析"}
            </button>
            <button className="btn btn-secondary" onClick={handleClear}>清除</button>
          </div>

          {examples.length > 0 && (
            <>
              <div className="hint" style={{ marginTop: 16, marginBottom: 4 }}>快速测试 — 点击下方示例文本</div>
              <div className="example-list">
                {examples.map((ex, i) => (
                  <button key={i} className="example-btn" onClick={() => handleExampleClick(ex.text)}>
                    {ex.text}
                  </button>
                ))}
              </div>
            </>
          )}
        </div>
      </div>

      <div className="col col-5">
        <div className="card">
          <h3>分析结果</h3>
          {result ? <ResultLabel results={result} /> : <div className="result-empty">等待输入...</div>}
        </div>
      </div>
    </div>
  )
}
