import { fetchAnalyzeStatus, health, resolveApiUrl, startAnalyzeCsv } from "./api.js";

const batchFile = document.getElementById("batch-file");
const batchFileLabel = document.getElementById("batch-file-label");
const dropzone = document.getElementById("dropzone");
const batchResult = document.getElementById("batch-result");
const batchSample = document.getElementById("batch-sample");
const batchStatus = document.getElementById("batch-status");
const downloadLink = document.getElementById("download-link");
const processingTime = document.getElementById("processing-time");
const analyzeProgress = document.getElementById("analyze-progress");
const fileButton = batchFile?.closest(".file-btn");
const classChartEl = document.getElementById("class-chart");
const sourceChartEl = document.getElementById("source-chart");
const sourceChartWrap = document.getElementById("source-chart-wrap");
const DEFAULT_FILE_LABEL = "Выбрать файл";
const MODE_PREDICT = "predict";
const MODE_SCORE = "score";
const POLL_INTERVAL_MS = 2000;
const MAX_POLL_ATTEMPTS = 150; // ~5 минут ожидания
const MAX_TIMEOUT_RETRIES = 1;
const STORAGE_KEYS = {
  taskId: "analyze_task_id",
  result: "analyze_result",
};

const CLASS_NAMES = {
  "0": "Нейтрально",
  "1": "Позитив",
  "2": "Негатив",
};

let isBusy = false;
let currentTaskId = null;
let pollTimer = null;
let pollAttempt = 0;
let classChart = null;
let sourceChart = null;
let lastFile = null;
let timeoutRetryCount = 0;

function canUseStorage() {
  try {
    const testKey = "__test__";
    window.localStorage.setItem(testKey, "1");
    window.localStorage.removeItem(testKey);
    return true;
  } catch (e) {
    return false;
  }
}

function persistTaskId(taskId) {
  if (!canUseStorage()) return;
  try {
    localStorage.setItem(STORAGE_KEYS.taskId, taskId);
    localStorage.removeItem(STORAGE_KEYS.result);
  } catch (e) {
    // ignore
  }
}

function persistResult(result) {
  if (!canUseStorage()) return;
  try {
    localStorage.setItem(STORAGE_KEYS.result, JSON.stringify(result));
    localStorage.removeItem(STORAGE_KEYS.taskId);
  } catch (e) {
    // ignore
  }
}

function loadTaskId() {
  if (!canUseStorage()) return null;
  try {
    return localStorage.getItem(STORAGE_KEYS.taskId);
  } catch (e) {
    return null;
  }
}

function loadResult() {
  if (!canUseStorage()) return null;
  try {
    const raw = localStorage.getItem(STORAGE_KEYS.result);
    return raw ? JSON.parse(raw) : null;
  } catch (e) {
    return null;
  }
}

function clearStoredTask() {
  if (!canUseStorage()) return;
  try {
    localStorage.removeItem(STORAGE_KEYS.taskId);
  } catch (e) {
    // ignore
  }
}

function clearStoredResult() {
  if (!canUseStorage()) return;
  try {
    localStorage.removeItem(STORAGE_KEYS.result);
  } catch (e) {
    // ignore
  }
}

function setStatus(text, tone = "muted") {
  if (!batchStatus) return;
  batchStatus.textContent = text;
  batchStatus.className = `pill ${tone}`;
}

function setProcessingTime(ms) {
  if (!processingTime) return;
  if (ms === undefined || ms === null) {
    processingTime.classList.add("hidden");
    processingTime.textContent = "";
    return;
  }
  processingTime.textContent = `~${ms} мс`;
  processingTime.classList.remove("hidden");
}

function setDownloadLink(url) {
  if (!downloadLink) return;
  if (url) {
    downloadLink.href = url;
    downloadLink.classList.remove("is-disabled");
  } else {
    downloadLink.href = "#";
    downloadLink.classList.add("is-disabled");
  }
}

function setProgressVisible(visible) {
  if (!analyzeProgress) return;
  if (visible) {
    analyzeProgress.classList.remove("hidden");
  } else {
    analyzeProgress.classList.add("hidden");
  }
}

function escapeHtml(text = "") {
  return text.replace(/[&<>"]/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[ch]));
}

function labelFromValue(val) {
  if (val === undefined || val === null) return null;
  return CLASS_NAMES[String(val)] || String(val);
}

function labelName(row) {
  const label = row?.predicted_label;
  if (label !== undefined && CLASS_NAMES[String(label)]) {
    return CLASS_NAMES[String(label)];
  }
  if (row?.predicted_class_name) return row.predicted_class_name;
  return "Класс не распознан";
}

function cardsToHtml(cards = []) {
  return cards
    .map(
      (card) => `
        <div class="stat-card">
          <div class="stat-card__label">${card.label}</div>
          <div class="stat-card__value">${card.value}</div>
        </div>
      `
    )
    .join("");
}

function renderSummary({ summary = { total_rows: 0, class_counts: {} }, mode = MODE_PREDICT, macro_f1, f1_per_class } = {}) {
  if (!batchResult) return;
  batchResult.classList.remove("hidden");
  const counts = summary.class_counts || {};
  const countCards = [
    { label: "Всего строк", value: summary.total_rows ?? 0 },
    { label: CLASS_NAMES["1"], value: counts["1"] ?? 0 },
    { label: CLASS_NAMES["0"], value: counts["0"] ?? 0 },
    { label: CLASS_NAMES["2"], value: counts["2"] ?? 0 },
  ];
  if (mode === MODE_SCORE) {
    const formatScore = (val) => (val === undefined || val === null ? "—" : Number(val).toFixed(3));
    const f1Cards = [
      { label: "Macro-F1", value: formatScore(macro_f1) },
      { label: `F1 ${CLASS_NAMES["1"]}`, value: formatScore(f1_per_class?.["1"]) },
      { label: `F1 ${CLASS_NAMES["0"]}`, value: formatScore(f1_per_class?.["0"]) },
      { label: `F1 ${CLASS_NAMES["2"]}`, value: formatScore(f1_per_class?.["2"]) },
    ];
    batchResult.innerHTML = `
      <div class="stat-cards">${cardsToHtml(f1Cards)}</div>
      <div class="stat-cards">${cardsToHtml(countCards)}</div>
    `;
    return;
  }
  batchResult.innerHTML = cardsToHtml(countCards);
}

function renderSample(sample = [], mode = MODE_PREDICT) {
  if (!batchSample) return;
  if (!sample.length) {
    batchSample.textContent = "Нет данных — загрузите файл.";
    batchSample.classList.add("muted");
    batchSample.classList.remove("hidden");
    return;
  }
  batchSample.classList.remove("muted");
  batchSample.classList.remove("hidden");
  const limited = sample.slice(0, 10);
  const hasTrueLabel = mode === MODE_SCORE || limited.some((row) => row.label !== undefined);
  const rows = limited
    .map((row) => {
      const trueLabel = hasTrueLabel ? `<td><span class="pill muted">${labelFromValue(row.label) || "—"}</span></td>` : "";
      return `
        <tr>
          <td>${escapeHtml(row.text || "")}</td>
          ${trueLabel}
          <td><span class="pill">${labelName(row)}</span></td>
        </tr>
      `;
    })
    .join("");
  const headerTrueLabel = hasTrueLabel ? "<th>Истина</th>" : "";
  batchSample.innerHTML = `
    <div class="sample-table__title">Фрагмент результата (первые ${Math.min(sample.length, 10)} строк)</div>
    <table>
      <thead><tr><th>Текст</th>${headerTrueLabel}<th>Предсказание</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

function destroyCharts() {
  if (classChart) {
    classChart.destroy();
    classChart = null;
  }
  if (sourceChart) {
    sourceChart.destroy();
    sourceChart = null;
  }
  sourceChartWrap?.classList.add("hidden");
}

function renderCharts(summary = { class_counts: {} }, sourceBreakdown = null) {
  if (typeof Chart === "undefined" || !classChartEl) return;
  const counts = summary?.class_counts || {};
  const classOrder = ["1", "0", "2"];
  const labels = classOrder.map((k) => CLASS_NAMES[k] || k);
  const data = classOrder.map((k) => counts[k] || 0);
  destroyCharts();
  classChart = new Chart(classChartEl, {
    type: "doughnut",
    data: {
      labels,
      datasets: [
        {
          data,
          backgroundColor: ["#22c55e", "#9ca3af", "#ef4444"],
          borderWidth: 0,
        },
      ],
    },
    options: {
      plugins: {
        legend: { position: "bottom" },
      },
      cutout: "55%",
    },
  });

  const hasSources = sourceBreakdown && Object.keys(sourceBreakdown || {}).length > 0;
  if (!hasSources || !sourceChartEl) {
    sourceChartWrap?.classList.add("hidden");
    return;
  }
  const srcEntries = Object.entries(sourceBreakdown || {}).sort((a, b) => {
    const sum = (obj) => Object.values(obj || {}).reduce((acc, v) => acc + v, 0);
    return sum(b[1]) - sum(a[1]);
  });
  const srcLabels = srcEntries.map(([src]) => src);
  const colors = { "1": "#22c55e", "0": "#9ca3af", "2": "#ef4444" };
  const datasets = classOrder.map((cls) => ({
    label: CLASS_NAMES[cls] || cls,
    data: srcEntries.map(([, c]) => c?.[cls] || 0),
    backgroundColor: colors[cls],
  }));
  sourceChart = new Chart(sourceChartEl, {
    type: "bar",
    data: { labels: srcLabels, datasets },
    options: {
      responsive: true,
      plugins: {
        legend: { position: "bottom" },
      },
      scales: {
        x: { stacked: true },
        y: { stacked: true, beginAtZero: true, ticks: { precision: 0 } },
      },
    },
  });
  sourceChartWrap?.classList.remove("hidden");
}

function formatError(err) {
  if (!err) return "Неизвестная ошибка";
  return err.message || String(err);
}

function attachDragAndDrop(area, input, labelEl) {
  if (!area || !input) return;
  ["dragenter", "dragover"].forEach((ev) =>
    area.addEventListener(ev, (e) => {
      e.preventDefault();
      e.stopPropagation();
      area.classList.add("dragover");
    })
  );
  ["dragleave", "drop"].forEach((ev) =>
    area.addEventListener(ev, (e) => {
      e.preventDefault();
      e.stopPropagation();
      area.classList.remove("dragover");
    })
  );
  area.addEventListener("drop", (e) => {
    const files = e.dataTransfer?.files;
    if (files && files.length) {
      input.files = files;
      if (labelEl) labelEl.textContent = files[0].name;
      input.dispatchEvent(new Event("change", { bubbles: true }));
    }
  });
  input.addEventListener("change", () => {
    if (input.files.length && labelEl) {
      labelEl.textContent = input.files[0].name;
    }
  });
}

function clearReport() {
  if (batchResult) {
    batchResult.innerHTML = "";
    batchResult.classList.add("hidden");
  }
  if (batchSample) {
    batchSample.textContent = "Нет данных — загрузите файл.";
    batchSample.classList.add("hidden");
    batchSample.classList.add("muted");
  }
  setProgressVisible(false);
  destroyCharts();
}

function resetFileInput() {
  if (batchFile) {
    batchFile.value = "";
    batchFile.disabled = false;
  }
  if (batchFileLabel) {
    batchFileLabel.textContent = DEFAULT_FILE_LABEL;
  }
  fileButton?.classList.remove("disabled");
}

function stopPolling() {
  if (pollTimer) {
    clearTimeout(pollTimer);
    pollTimer = null;
  }
  currentTaskId = null;
  pollAttempt = 0;
  if (batchFile) {
    batchFile.disabled = false;
  }
}

function handleResult(resp, { fromRestore = false } = {}) {
  const mode = resp?.mode === MODE_SCORE ? MODE_SCORE : MODE_PREDICT;
  setStatus(mode === MODE_SCORE ? "Оценка завершена (валидация)" : "Анализ завершен", "success");
  renderSummary({
    summary: resp.summary,
    mode,
    macro_f1: resp.macro_f1,
    f1_per_class: resp.f1_per_class,
  });
  renderSample(resp.sample || [], mode);
  const fileUrl = resp.file_url ? resolveApiUrl(resp.file_url) : null;
  setDownloadLink(fileUrl);
  setProcessingTime(resp.processing_time_ms);
  renderCharts(resp.summary, resp.source_breakdown);
  timeoutRetryCount = 0;
  isBusy = false;
  persistResult(resp);
  if (!fromRestore) {
    stopPolling();
    resetFileInput();
  } else {
    currentTaskId = null;
  }
  setProgressVisible(false);
}

function isTimeoutError(message) {
  if (!message) return false;
  const lower = message.toLowerCase();
  return lower.includes("таймаут") || lower.includes("timeout");
}

function handleError(message) {
  const timeout = isTimeoutError(message);
  setStatus(message || "Ошибка", timeout ? "muted" : "error");
  clearReport();
  setDownloadLink(null);
  setProcessingTime();
  isBusy = false;
  stopPolling();
  resetFileInput();
  clearStoredTask();
  clearStoredResult();
  setProgressVisible(false);
  renderCharts();

  const canRetry = timeout && lastFile && timeoutRetryCount < MAX_TIMEOUT_RETRIES;
  if (canRetry) {
    timeoutRetryCount += 1;
    setStatus("Таймаут, повторяем отправку файла...", "muted");
    setProgressVisible(true);
    setTimeout(() => runPredict(lastFile, { isRetry: true }), 600);
  }
}

async function pollAnalyze(taskId) {
  if (currentTaskId && taskId !== currentTaskId) return;
  try {
    const resp = await fetchAnalyzeStatus(taskId);
    const status = resp?.status;
    if (status === "completed" && resp.result) {
      handleResult(resp.result);
      return;
    }
    if (status === "error") {
      handleError(resp.error || "Ошибка анализа файла");
      return;
    }
    if (pollAttempt >= MAX_POLL_ATTEMPTS) {
      handleError("Таймаут ожидания анализа");
      return;
    }
    pollAttempt += 1;
    setStatus("Файл загружен, идет анализ...", "muted");
  } catch (e) {
    if (pollAttempt >= 3) {
      handleError(formatError(e));
      return;
    }
    pollAttempt += 1;
    setStatus("Пробуем ещё раз получить статус...", "muted");
  }
  pollTimer = setTimeout(() => pollAnalyze(taskId), POLL_INTERVAL_MS);
}

async function initHealth() {
  try {
    const resp = await health();
    if (!isBusy && resp?.model_loaded) {
      setStatus("Модель готова, загрузите CSV", "success");
    }
  } catch (e) {
    if (!isBusy) {
      setStatus("API недоступен, попробуйте позже", "error");
    }
  }
}

async function runPredict(fileArg = null, opts = {}) {
  if (isBusy) return;
  const file = fileArg || batchFile?.files?.[0] || lastFile;
  if (!file) {
    setStatus("Выберите CSV с колонкой text", "error");
    return;
  }
  if (!opts.isRetry) {
    lastFile = file;
    timeoutRetryCount = 0;
  }
  stopPolling();
  clearStoredTask();
  isBusy = true;
  if (batchFile) {
    batchFile.disabled = true;
  }
  fileButton?.classList.add("disabled");
  setStatus("Отправляем файл в модель...", "muted");
  setProcessingTime();
  setDownloadLink(null);
  clearReport();
  setProgressVisible(true);
  clearStoredResult();
  try {
    const resp = await startAnalyzeCsv(file);
    if (!resp?.task_id) {
      throw new Error("Не получили идентификатор задачи");
    }
    persistTaskId(resp.task_id);
    resetFileInput();
    currentTaskId = resp.task_id;
    setStatus("Файл загружен, идет анализ...", "muted");
    pollAttempt = 0;
    pollTimer = setTimeout(() => pollAnalyze(resp.task_id), POLL_INTERVAL_MS);
  } catch (e) {
    handleError(formatError(e));
  }
}

attachDragAndDrop(dropzone || batchFile?.parentElement, batchFile, batchFileLabel);
dropzone?.addEventListener("click", (e) => {
  if (e.target.closest("label")) return;
  if (batchFile) batchFile.value = "";
  batchFile?.click();
});
batchFile?.addEventListener("change", runPredict);
batchFile?.addEventListener("click", () => {
  // Позволяем выбрать тот же файл повторно.
  batchFile.value = "";
});
batchFile?.addEventListener("input", () => {
  // Safari может триггерить input вместо change.
  if (!isBusy) runPredict();
});

function restoreState() {
  const savedResult = loadResult();
  if (savedResult) {
    handleResult(savedResult, { fromRestore: true });
  }
  const savedTaskId = loadTaskId();
  if (savedTaskId) {
    currentTaskId = savedTaskId;
    isBusy = true;
    setStatus("Продолжаем анализ вашего файла...", "muted");
    setProgressVisible(true);
    pollAttempt = 0;
    pollTimer = setTimeout(() => pollAnalyze(savedTaskId), POLL_INTERVAL_MS);
  }
}

restoreState();
initHealth();
