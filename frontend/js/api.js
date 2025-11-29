const apiBase =
  (typeof window !== "undefined" && window.API_BASE) ||
  document.querySelector('meta[name="api-base"]')?.content ||
  `${window.location.protocol}//${window.location.hostname}:8000`;

export const API_BASE = apiBase;
const DEFAULT_TIMEOUT_MS = 180000;

async function fetchWithTimeout(url, options = {}, timeoutMs = DEFAULT_TIMEOUT_MS) {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const resp = await fetch(url, { ...options, signal: controller.signal });
    return resp;
  } catch (e) {
    if (e.name === "AbortError") {
      throw new Error("Таймаут запроса к API");
    }
    throw e;
  } finally {
    clearTimeout(id);
  }
}

async function handleResponse(resp) {
  let data;
  try {
    data = await resp.json();
  } catch (e) {
    throw new Error("Не удалось разобрать ответ");
  }
  if (!resp.ok) {
    const detail = data?.detail || resp.statusText;
    throw new Error(detail);
  }
  return data;
}

export async function health() {
  const resp = await fetch(`${apiBase}/health`);
  return handleResponse(resp);
}

export async function predictText(text) {
  const resp = await fetch(`${apiBase}/predict_text`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  return handleResponse(resp);
}

export async function predictCsv(file, opts = {}) {
  const fd = new FormData();
  fd.append("file", file);
  const params = new URLSearchParams();
  if (opts.returnFile) params.set("return_file", "true");
  if (opts.stream) params.set("stream", "true");
  const resp = await fetch(`${apiBase}/predict_csv?${params.toString()}`, {
    method: "POST",
    body: fd,
  });
  if (opts.returnFile) {
    const blob = await resp.blob();
    return { blob, filename: file.name?.replace(".csv", "_predicted.csv") || "predicted.csv" };
  }
  return handleResponse(resp);
}

export async function analyzeCsv(file) {
  const fd = new FormData();
  fd.append("file", file);
  const resp = await fetchWithTimeout(`${apiBase}/analyze_csv`, {
    method: "POST",
    body: fd,
  });
  return handleResponse(resp);
}

export async function startAnalyzeCsv(file) {
  const fd = new FormData();
  fd.append("file", file);
  const resp = await fetchWithTimeout(`${apiBase}/analyze_csv_async`, {
    method: "POST",
    body: fd,
  });
  return handleResponse(resp);
}

export async function fetchAnalyzeStatus(taskId) {
  const resp = await fetchWithTimeout(`${apiBase}/analyze_csv_async/${taskId}`, {}, 60000);
  return handleResponse(resp);
}

export async function scoreCsv(file) {
  const fd = new FormData();
  fd.append("file", file);
  const resp = await fetch(`${apiBase}/score`, {
    method: "POST",
    body: fd,
  });
  return handleResponse(resp);
}

export function resolveApiUrl(pathOrUrl) {
  if (!pathOrUrl) return null;
  try {
    return new URL(pathOrUrl, apiBase).href;
  } catch (e) {
    return pathOrUrl;
  }
}
