from collections import Counter
from io import StringIO
from typing import Dict, Tuple, Optional
import logging
import time
from pathlib import Path
import csv

import pandas as pd
from fastapi import HTTPException, UploadFile
from starlette.responses import StreamingResponse
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np

from .ml_model import get_model
from .config import (
    STORAGE_DIR,
    DOWNLOAD_BASE_URL,
    MAX_REVIEWS_TO_SAVE,
    MAX_CSV_BYTES,
    MAX_CSV_ROWS,
    STORAGE_TTL_HOURS,
    STORAGE_MAX_FILES,
    PREDICT_SCORES_MAX_ROWS,
)
from . import db

logger = logging.getLogger(__name__)


def _read_csv_bytes(data: bytes, require_label: bool = False) -> pd.DataFrame:
    if not data:
        raise HTTPException(status_code=400, detail="Пустой файл")
    if MAX_CSV_BYTES > 0 and len(data) > MAX_CSV_BYTES:
        raise HTTPException(status_code=413, detail="Слишком большой файл")
    try:
        df = pd.read_csv(StringIO(data.decode("utf-8-sig")))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Не удалось прочитать CSV: {exc}")

    df = df.rename(columns=lambda c: c.lstrip("\ufeff"))
    if "text" not in df.columns:
        raise HTTPException(status_code=400, detail="Отсутствует колонка 'text'")
    if require_label and "label" not in df.columns:
        raise HTTPException(status_code=400, detail="Отсутствует колонка 'label'")

    df["text"] = df["text"].fillna("").astype(str)
    if MAX_CSV_ROWS > 0 and len(df) > MAX_CSV_ROWS:
        raise HTTPException(status_code=413, detail="Слишком много строк в CSV")
    logger.info("Файл: %d байт, %d строк", len(data), len(df))
    return df


def read_upload(file: UploadFile, require_label: bool = False) -> pd.DataFrame:
    data = file.file.read()
    return _read_csv_bytes(data, require_label=require_label)


def add_predictions(df: pd.DataFrame) -> pd.DataFrame:
    model = get_model()
    if not model.is_loaded:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    t0 = time.perf_counter()
    labels, class_names = model.predict_many(df["text"].tolist())
    logger.info("Предсказание %d строк заняло %.2f c", len(df), time.perf_counter() - t0)
    df = df.copy()
    df["predicted_label"] = labels
    df["predicted_class_name"] = class_names
    if PREDICT_SCORES_MAX_ROWS > 0 and len(df) <= PREDICT_SCORES_MAX_ROWS:
        # Попробуем получить decision_function или predict_proba.
        try:
            pipeline = model.pipeline
            clf = pipeline.named_steps["clf"]
            features = pipeline.named_steps["features"].transform(df["text"].tolist())
            if hasattr(clf, "predict_proba"):
                scores = clf.predict_proba(features)
            elif hasattr(clf, "decision_function"):
                scores = clf.decision_function(features)
                # decision_function может вернуть 1D для бинарки, но у нас 3 класса — ожидаем 2D.
            else:
                scores = None
            if scores is not None:
                scores_arr = np.asarray(scores)
                df["predicted_scores"] = scores_arr.tolist()
        except Exception as exc:
            logger.warning("Не удалось рассчитать predicted_scores: %s", exc)
    return df


def summarize_predictions(df: pd.DataFrame) -> Dict:
    class_counts = (
        df["predicted_label"].value_counts().to_dict()
        if "predicted_label" in df.columns
        else {}
    )
    class_counts = {str(int(k)): int(v) for k, v in class_counts.items()}
    return {"total_rows": int(len(df)), "class_counts": class_counts}


def dataframe_to_streaming_csv(df: pd.DataFrame, filename: str) -> StreamingResponse:
    def iter_rows():
        buf = StringIO()
        writer = csv.writer(buf)
        writer.writerow(df.columns.tolist())
        yield buf.getvalue()
        buf.seek(0)
        buf.truncate(0)
        for row in df.itertuples(index=False, name=None):
            writer.writerow(row)
            if buf.tell() >= 8192:
                yield buf.getvalue()
                buf.seek(0)
                buf.truncate(0)
        if buf.tell():
            yield buf.getvalue()

    return StreamingResponse(
        iter_rows(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def save_dataframe_csv(df: pd.DataFrame, filename: str) -> Path:
    cleanup_storage()
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    path = STORAGE_DIR / filename
    df.to_csv(path, index=False)
    return path


def build_download_url(path: Path) -> str:
    if DOWNLOAD_BASE_URL:
        return f"{DOWNLOAD_BASE_URL.rstrip('/')}/{path.name}"
    return f"/download/{path.name}"


def cleanup_storage():
    if STORAGE_TTL_HOURS <= 0 and STORAGE_MAX_FILES <= 0:
        return
    try:
        files = sorted(STORAGE_DIR.glob("*"), key=lambda p: p.stat().st_mtime)
        now = time.time()
        removed = 0
        if STORAGE_TTL_HOURS > 0:
            threshold = now - STORAGE_TTL_HOURS * 3600
            for f in files:
                if f.stat().st_mtime < threshold:
                    f.unlink(missing_ok=True)
                    removed += 1
            files = [f for f in files if f.exists()]
        if STORAGE_MAX_FILES > 0 and len(files) > STORAGE_MAX_FILES:
            extra = len(files) - STORAGE_MAX_FILES
            for f in files[:extra]:
                f.unlink(missing_ok=True)
                removed += 1
        if removed:
            logger.info("Очистка storage: удалено %d файлов", removed)
    except Exception as exc:
        logger.warning("Не удалось очистить storage: %s", exc)


def score_predictions(df: pd.DataFrame) -> Tuple[float, Dict[str, float], Dict[str, int]]:
    if "label" not in df.columns or "predicted_label" not in df.columns:
        raise ValueError("Нет label или predicted_label для оценки")
    y_true = df["label"].astype(int).tolist()
    y_pred = df["predicted_label"].astype(int).tolist()
    macro = float(f1_score(y_true, y_pred, average="macro"))
    per_class_arr = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2])
    f1_per_class = {str(i): float(score) for i, score in enumerate(per_class_arr)}
    support = Counter(y_true)
    support = {str(k): int(v) for k, v in support.items()}
    return macro, f1_per_class, support


def confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    return cm.tolist()


def persist_batch(
    file_name: str,
    purpose: str,
    total_rows: int,
    class_counts: Dict[str, int],
    *,
    macro_f1: Optional[float] = None,
    f1_per_class: Optional[Dict[str, float]] = None,
    support: Optional[Dict[str, int]] = None,
    output_path: Optional[str] = None,
    df_with_preds: Optional[pd.DataFrame] = None,
):
    conn = db.get_connection()
    if not conn:
        return None
    try:
        batch_id = db.insert_batch(
            conn,
            file_name=file_name,
            purpose=purpose,
            status="completed",
            total_rows=total_rows,
            class_counts=class_counts,
            macro_f1=macro_f1,
            f1_per_class=f1_per_class,
            support=support,
            output_path=output_path,
        )
        logger.info("Сохранил batch %s в БД", batch_id)
        if df_with_preds is not None and MAX_REVIEWS_TO_SAVE > 0:
            saved = db.insert_reviews(conn, batch_id=batch_id, df=df_with_preds, max_rows=MAX_REVIEWS_TO_SAVE)
            logger.info("Сохранил %d строк в reviews (ограничение %d)", saved, MAX_REVIEWS_TO_SAVE)
        return batch_id
    except Exception as exc:
        logger.warning("Не удалось сохранить batch в БД: %s", exc)
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass
