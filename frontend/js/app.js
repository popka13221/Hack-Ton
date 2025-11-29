import { fetchAnalyzeStatus, health, resolveApiUrl, startAnalyzeCsv } from "./api.js";

const batchFile = document.getElementById("batch-file");
const batchFileLabel = document.getElementById("batch-file-label");
const dropzone = document.getElementById("dropzone");
const batchResult = document.getElementById("batch-result");
const batchSample = document.getElementById("batch-sample");
const batchStatus = document.getElementById("batch-status");
const downloadLink = document.getElementById("download-link");
const processingTime = document.getElementById("processing-time");
const DEFAULT_FILE_LABEL = "Выбрать файл";
const MODE_PREDICT = "predict";
const MODE_SCORE = "score";
const POLL_INTERVAL_MS = 2000;
const MAX_POLL_ATTEMPTS = 150; // ~5 минут ожидания

const CLASS_NAMES = {
  "0": "Нейтрально",
  "1": "Позитив",
  "2": "Негатив",
};

let isBusy = false;
let currentTaskId = null;
let pollTimer = null;
let pollAttempt = 0;

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
}

function resetFileInput() {
  if (batchFile) {
    batchFile.value = "";
  }
  if (batchFileLabel) {
    batchFileLabel.textContent = DEFAULT_FILE_LABEL;
  }
}

function stopPolling() {
  if (pollTimer) {
    clearTimeout(pollTimer);
    pollTimer = null;
  }
  currentTaskId = null;
  pollAttempt = 0;
}

function handleResult(resp) {
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
  isBusy = false;
  stopPolling();
  resetFileInput();
}

function handleError(message) {
  setStatus(message || "Ошибка", "error");
  clearReport();
  setDownloadLink(null);
  setProcessingTime();
  isBusy = false;
  stopPolling();
  resetFileInput();
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

async function runPredict() {
  if (isBusy) return;
  const file = batchFile?.files?.[0];
  if (!file) {
    setStatus("Выберите CSV с колонкой text", "error");
    return;
  }
  stopPolling();
  isBusy = true;
  setStatus("Отправляем файл в модель...", "muted");
  setProcessingTime();
  setDownloadLink(null);
  clearReport();
  try {
    const resp = await startAnalyzeCsv(file);
    if (!resp?.task_id) {
      throw new Error("Не получили идентификатор задачи");
    }
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

initHealth();
