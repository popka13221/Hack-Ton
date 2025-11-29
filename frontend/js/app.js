import { predictText, predictCsv, scoreCsv, health } from "./api.js";

const singleText = document.getElementById("single-text");
const singleBtn = document.getElementById("single-btn");
const singleResult = document.getElementById("single-result");

const batchFile = document.getElementById("batch-file");
const batchBtn = document.getElementById("batch-btn");
const batchResult = document.getElementById("batch-result");
const batchFileLabel = document.getElementById("batch-file-label");

const scoreFile = document.getElementById("score-file");
const scoreBtn = document.getElementById("score-btn");
const scoreResult = document.getElementById("score-result");
const scoreFileLabel = document.getElementById("score-file-label");

const apiStatus = document.getElementById("api-status");

function setResult(el, html) {
  el.innerHTML = html;
}

function formatCounts(counts = {}) {
  return Object.entries(counts)
    .map(([k, v]) => `<span class="pill">${k}: ${v}</span>`)
    .join(" ");
}

function attachDragAndDrop(area, input, labelEl) {
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
    }
  });
  input.addEventListener("change", () => {
    if (input.files.length && labelEl) {
      labelEl.textContent = input.files[0].name;
    }
  });
}

async function initHealth() {
  try {
    apiStatus.textContent = "API: проверка...";
    const h = await health();
    const model = h.model_loaded ? "модель загружена" : "модель не загружена";
    const db = h.db_connected ? "БД: ok" : "БД: нет";
    apiStatus.textContent = `API: ok · ${model} · ${db}`;
    apiStatus.classList.remove("error");
  } catch (e) {
    apiStatus.textContent = "API: не доступен";
    apiStatus.classList.add("error");
  }
}

singleBtn?.addEventListener("click", async () => {
  if (!singleText.value.trim()) {
    setResult(singleResult, `<span class="pill error">Введите текст</span>`);
    return;
  }
  setResult(singleResult, `<span class="pill muted">Обработка...</span>`);
  try {
    const resp = await predictText(singleText.value);
    setResult(
      singleResult,
      `<div class="result-block">
        <span class="pill">predicted_label: ${resp.predicted_label}</span>
        <span class="pill">${resp.predicted_class_name}</span>
      </div>`
    );
  } catch (e) {
    setResult(singleResult, `<span class="pill error">${e.message}</span>`);
  }
});

batchFile && attachDragAndDrop(batchFile.parentElement, batchFile, batchFileLabel);

batchBtn?.addEventListener("click", async () => {
  if (!batchFile.files.length) {
    setResult(batchResult, `<span class="pill error">Выберите CSV</span>`);
    return;
  }
  setResult(batchResult, `<span class="pill muted">Загружаем...</span>`);
  try {
    const resp = await predictCsv(batchFile.files[0], { returnFile: false });
    const counts = formatCounts(resp.summary?.class_counts);
    const download = resp.file_url
      ? `<a class="download-link" href="${resp.file_url}">Скачать результат</a>`
      : "";
    const time = resp.processing_time_ms ? `<span class="pill">~${resp.processing_time_ms} мс</span>` : "";
    setResult(
      batchResult,
      `<div class="result-block">
         <div>Строк: ${resp.summary?.total_rows ?? 0}</div>
         <div class="stat-grid">${counts}</div>
         <div>${download} ${time}</div>
       </div>`
    );
  } catch (e) {
    setResult(batchResult, `<span class="pill error">${e.message}</span>`);
  }
});

scoreFile && attachDragAndDrop(scoreFile.parentElement, scoreFile, scoreFileLabel);

scoreBtn?.addEventListener("click", async () => {
  if (!scoreFile.files.length) {
    setResult(scoreResult, `<span class="pill error">Выберите CSV</span>`);
    return;
  }
  setResult(scoreResult, `<span class="pill muted">Загружаем...</span>`);
  try {
    const resp = await scoreCsv(scoreFile.files[0]);
    const perClass = Object.entries(resp.f1_per_class || {})
      .map(([k, v]) => `<span class="pill">${k}: ${v.toFixed(3)}</span>`)
      .join(" ");
    const support = formatCounts(resp.support);
    const download = resp.file_url
      ? `<a class="download-link" href="${resp.file_url}">Скачать файл</a>`
      : "";
    const time = resp.processing_time_ms ? `<span class="pill">~${resp.processing_time_ms} мс</span>` : "";
    setResult(
      scoreResult,
      `<div class="result-block">
         <div class="pill success">macro-F1: ${resp.macro_f1.toFixed(3)}</div>
         <div class="stat-grid">${perClass}</div>
         <div class="stat-grid">${support}</div>
         <div>${download} ${time}</div>
      </div>`
    );
  } catch (e) {
    // Если нет label — fallback в predict_csv.
    if (e.message && e.message.includes("label")) {
      try {
        const resp = await predictCsv(scoreFile.files[0], { returnFile: false });
        const counts = formatCounts(resp.summary?.class_counts);
        const download = resp.file_url
          ? `<a class="download-link" href="${resp.file_url}">Скачать результат</a>`
          : "";
        const time = resp.processing_time_ms ? `<span class="pill">~${resp.processing_time_ms} мс</span>` : "";
        setResult(
          scoreResult,
          `<div class="result-block">
             <div class="pill success">Файл без label, выполнено predict.</div>
             <div>Строк: ${resp.summary?.total_rows ?? 0}</div>
             <div class="stat-grid">${counts}</div>
             <div>${download} ${time}</div>
           </div>`
        );
        return;
      } catch (err2) {
        setResult(scoreResult, `<span class="pill error">${err2.message}</span>`);
        return;
      }
    }
    setResult(scoreResult, `<span class="pill error">${e.message || "Ошибка запроса"}</span>`);
  }
});

initHealth();
