import { predictCsv, health } from "./api.js";

const batchFile = document.getElementById("batch-file");
const batchResult = document.getElementById("batch-result");
const batchFileLabel = document.getElementById("batch-file-label");
const dropzone = document.getElementById("dropzone");
function setResult(el, html) {
  el.innerHTML = html;
}

function formatCounts(counts = {}) {
  return Object.entries(counts)
    .map(([k, v]) => `<span class="pill">${k}: ${v}</span>`)
    .join(" ");
}

function formatError(err) {
  if (!err) return "Неизвестная ошибка";
  return err.message || String(err);
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
      runPredict();
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
    await health();
  } catch (e) {
    // no-op
  }
}

async function runPredict() {
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
    setResult(batchResult, `<span class="pill error">${formatError(e)}</span>`);
  }
}

// Hooks
batchFile && attachDragAndDrop(dropzone || batchFile.parentElement, batchFile, batchFileLabel);
dropzone?.addEventListener("click", () => batchFile?.click());
batchFile?.addEventListener("change", runPredict);

initHealth();
