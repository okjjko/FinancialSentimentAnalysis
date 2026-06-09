const API = ""

export async function getStatus() {
  const r = await fetch(`${API}/api/status`)
  return r.json()
}

export async function getExamples() {
  const r = await fetch(`${API}/api/examples`)
  return r.json()
}

export async function predictSingle(text) {
  const r = await fetch(`${API}/api/predict/single`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  })
  if (!r.ok) throw new Error(await r.text())
  return r.json()
}

export async function predictBatch(file) {
  const fd = new FormData()
  fd.append("file", file)
  const r = await fetch(`${API}/api/predict/batch`, {
    method: "POST",
    body: fd,
  })
  if (!r.ok) throw new Error(await r.text())
  return r.json()
}

export async function getHistory() {
  const r = await fetch(`${API}/api/history`)
  return r.json()
}

export async function getHistoryDetail(id) {
  const r = await fetch(`${API}/api/history/${id}`)
  if (!r.ok) throw new Error("记录不存在")
  return r.json()
}

export async function clearHistory() {
  const r = await fetch(`${API}/api/history`, { method: "DELETE" })
  return r.json()
}
