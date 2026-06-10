import { useState, useRef } from "react"
import { predictBatch } from "../api"

export default function BatchTab({ onPredictionDone }) {
  const [fileName, setFileName] = useState(null)
  const [results, setResults] = useState(null)
  const [predicting, setPredicting] = useState(false)
  const fileRef = useRef(null)

  const handleFile = (file) => {
    if (!file) return
    setFileName(file.name)
    setResults(null)
  }

  const handlePredict = async () => {
    const file = fileRef.current?.files?.[0]
    if (!file) return
    setPredicting(true)
    try {
      const data = await predictBatch(file)
      setResults(data.results)
      onPredictionDone()
    } catch (err) {
      setResults([{ text: "错误", error: err.message }])
    }
    setPredicting(false)
  }

  const downloadCSV = () => {
    if (!results) return
    const emotionOrder = ["joy", "sadness", "anger", "fear", "love", "surprise"]
    const headers = ["文本", ...emotionOrder, "Top-1", "Top-1 置信度"]
    const rows = results.map((r) => {
      if (r.error) return [r.text, ...emotionOrder.map(() => ""), `错误: ${r.error}`, ""]
      const probs = emotionOrder.map((e) => {
        const found = r.results.find((p) => p["class"] === e)
        return found ? found.confidence.toFixed(4) : ""
      })
      const top = r.results[0]
      return [r.text, ...probs, top ? top["class"] : "", top ? top.confidence.toFixed(4) : ""]
    })
    const csv = [headers.join(","), ...rows.map((r) => `"${r.join('","')}"`)].join("\n")
    const blob = new Blob(["\ufeff" + csv], { type: "text/csv;charset=utf-8" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url; a.download = "batch_results.csv"; a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <>
      <div className="card">
        <h3>上传文本文件</h3>
        <div className="row">
          <div className="col" style={{ flex: 2, minWidth: 260 }}>
            <div className="upload-area" onClick={() => fileRef.current?.click()}>
              <div className="icon">📄</div>
              <div className="text">{fileName || "点击选择 .txt 文件"}</div>
              <div className="hint">每行一条文本，UTF-8 编码</div>
            </div>
            <input ref={fileRef} type="file" accept=".txt"
              style={{ display: "none" }}
              onChange={(e) => handleFile(e.target.files[0])} />
            <button className="btn btn-primary" disabled={!fileName || predicting}
              onClick={handlePredict} style={{ marginTop: 12, width: "100%" }}>
              {predicting ? <span className="loading" /> : "开始批量分析"}
            </button>
          </div>
        </div>
      </div>

      {results && (
        <div className="card" style={{ marginTop: 14 }}>
          <h3>分析结果（{results.length} 条）</h3>
          <button className="btn btn-secondary btn-sm" onClick={downloadCSV} style={{ marginBottom: 8 }}>
            下载 CSV
          </button>
          <div style={{ overflowX: "auto" }}>
            <table className="batch-table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>文本</th>
                  <th>Top-1</th>
                  <th>Top-2</th>
                  <th>Top-3</th>
                  <th>Top-1置信度</th>
                  <th>Top-2置信度</th>
                  <th>Top-3置信度</th>
                </tr>
              </thead>
              <tbody>
                {results.map((r, i) => {
                  if (r.error) {
                    return (
                      <tr key={i}>
                        <td>{i + 1}</td>
                        <td colSpan={7} style={{ color: "var(--terracotta)" }}>错误: {r.error}</td>
                      </tr>
                    )
                  }
                  return (
                    <tr key={i}>
                      <td>{i + 1}</td>
                      <td style={{ maxWidth: 300, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                        {r.text}
                      </td>
                      {[0, 1, 2].map((j) => (
                        <td key={`c${j}`}>{r.results[j] ? r.results[j]["class"] : ""}</td>
                      ))}
                      {[0, 1, 2].map((j) => (
                        <td key={`p${j}`}>{r.results[j] ? r.results[j].confidence.toFixed(4) : ""}</td>
                      ))}
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </>
  )
}
