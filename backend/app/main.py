import logging
import time
import uuid
import json
from pathlib import Path
import threading

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.responses import PlainTextResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .models import (
    HealthResponse,
    PredictTextRequest,
    PredictTextResponse,
    PredictManyRequest,
    PredictManyResponse,
    PredictCsvResponse,
    ScoreResponse,
    AnalyzeCsvResponse,
)
from .ml_model import get_model
from . import csv_utils, db
from .config import (
    STORAGE_DIR,
    CORS_ALLOW_ORIGINS,
    MAX_CONCURRENT_TASKS,
    TASK_ACQUIRE_TIMEOUT,
    RATE_LIMIT_PER_MIN,
    RATE_LIMIT_WINDOW_SEC,
)


app = FastAPI(title="Sentiment Service", version="0.1.0")
logger = logging.getLogger(__name__)
if MAX_CONCURRENT_TASKS > 0:
    TASK_SEM = threading.Semaphore(MAX_CONCURRENT_TASKS)
else:
    TASK_SEM = None

# Простая in-memory rate-limit таблица.
_rate_lock = threading.Lock()
_rate_map = {}  # key -> (window_start_ts, count)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)

# Enable permissive CORS for early development; tighten in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    try:
        response = await call_next(request)
        status = response.status_code
        error = None
    except Exception as exc:
        status = 500
        error = str(exc)
        response = JSONResponse({"detail": "Internal server error"}, status_code=500)
    duration_ms = int((time.perf_counter() - start) * 1000)
    logger.info(
        json.dumps(
            {
                "method": request.method,
                "path": request.url.path,
                "status": status,
                "duration_ms": duration_ms,
                "error": error,
                "client": request.client.host if request.client else None,
            },
            ensure_ascii=False,
        ),
        exc_info=bool(error),
    )
    return response


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    model = get_model()
    db_ok = db.is_available()
    return HealthResponse(model_loaded=model.is_loaded, db_connected=db_ok)


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    logger.warning(
        json.dumps(
            {
                "method": request.method,
                "path": request.url.path,
                "status": exc.status_code,
                "detail": exc.detail,
                "client": request.client.host if request.client else None,
            },
            ensure_ascii=False,
        )
    )
    return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(
        json.dumps(
            {
                "method": request.method,
                "path": request.url.path,
                "status": 422,
                "detail": exc.errors(),
                "client": request.client.host if request.client else None,
            },
            ensure_ascii=False,
        )
    )
    return PlainTextResponse(str(exc), status_code=422)


@app.post("/predict_text", response_model=PredictTextResponse)
def predict_text(payload: PredictTextRequest) -> PredictTextResponse:
    model = get_model()
    if not model.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    label, class_name = model.predict_text(payload.text)
    return PredictTextResponse(predicted_label=label, predicted_class_name=class_name)


@app.post("/predict_many", response_model=PredictManyResponse)
def predict_many(payload: PredictManyRequest) -> PredictManyResponse:
    model = get_model()
    if not model.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    labels, class_names = model.predict_many(payload.texts)
    return PredictManyResponse(predicted_labels=labels, predicted_class_names=class_names)


@app.post("/predict_csv", response_model=PredictCsvResponse)
def predict_csv(
    request: Request,
    file: UploadFile = File(...),
    return_file: bool = Query(False, description="Вернуть CSV файлом, а не JSON"),
    stream: bool = Query(False, description="Стримить CSV без сохранения файла"),
):
    enforce_rate_limit(request)
    if TASK_SEM and not TASK_SEM.acquire(timeout=TASK_ACQUIRE_TIMEOUT):
        raise HTTPException(status_code=429, detail="Слишком много одновременных задач, попробуйте позже")
    try:
        t0 = time.perf_counter()
        df = csv_utils.read_upload(file, require_label=False)
        df_pred = csv_utils.add_predictions(df)
        summary = csv_utils.summarize_predictions(df_pred)
        sample = df_pred.head(20).to_dict(orient="records")
        duration_ms = int((time.perf_counter() - t0) * 1000)
        filename = f"predicted_{uuid.uuid4().hex}.csv"
        saved_path = None
        file_url = None
        output_path = None
        if not stream:
            saved_path = csv_utils.save_dataframe_csv(df_pred, filename)
            file_url = csv_utils.build_download_url(saved_path)
            output_path = str(saved_path)
        csv_utils.persist_batch(
            file_name=file.filename or "upload.csv",
            purpose="predict",
            total_rows=summary["total_rows"],
            class_counts=summary["class_counts"],
            output_path=output_path,
            df_with_preds=df_pred,
        )
    finally:
        if TASK_SEM:
            TASK_SEM.release()
    if return_file:
        if stream:
            return csv_utils.dataframe_to_streaming_csv(df_pred, filename=file.filename or "predicted.csv")
        return FileResponse(saved_path, media_type="text/csv", filename=file.filename or "predicted.csv")
    return PredictCsvResponse(summary=summary, sample=sample, file_url=file_url, processing_time_ms=duration_ms)


@app.post("/score", response_model=ScoreResponse)
def score(request: Request, file: UploadFile = File(...)):
    enforce_rate_limit(request)
    if TASK_SEM and not TASK_SEM.acquire(timeout=TASK_ACQUIRE_TIMEOUT):
        raise HTTPException(status_code=429, detail="Слишком много одновременных задач, попробуйте позже")
    try:
        t0 = time.perf_counter()
        df = csv_utils.read_upload(file, require_label=True)
        df_pred = csv_utils.add_predictions(df)
        macro, per_class, support = csv_utils.score_predictions(df_pred)
        cm = csv_utils.confusion(df_pred["label"].astype(int).tolist(), df_pred["predicted_label"].astype(int).tolist())
        duration_ms = int((time.perf_counter() - t0) * 1000)
        class_counts = csv_utils.summarize_predictions(df_pred)["class_counts"]

        filename = f"scored_{uuid.uuid4().hex}.csv"
        saved_path = csv_utils.save_dataframe_csv(df_pred, filename)
        file_url = csv_utils.build_download_url(saved_path)
        csv_utils.persist_batch(
            file_name=file.filename or "upload.csv",
            purpose="score",
            total_rows=len(df_pred),
            class_counts=class_counts,
            macro_f1=macro,
            f1_per_class=per_class,
            support=support,
            output_path=str(saved_path),
            df_with_preds=df_pred,
        )
    finally:
        if TASK_SEM:
            TASK_SEM.release()

    return ScoreResponse(
        macro_f1=macro,
        f1_per_class=per_class,
        support=support,
        confusion_matrix=cm,
        file_url=file_url,
        processing_time_ms=duration_ms,
    )


@app.post("/analyze_csv", response_model=AnalyzeCsvResponse)
def analyze_csv(request: Request, file: UploadFile = File(...)):
    enforce_rate_limit(request)
    if TASK_SEM and not TASK_SEM.acquire(timeout=TASK_ACQUIRE_TIMEOUT):
        raise HTTPException(status_code=429, detail="Слишком много одновременных задач, попробуйте позже")
    try:
        t0 = time.perf_counter()
        df = csv_utils.read_upload(file, require_label=False)
        has_label = "label" in df.columns
        df_pred = csv_utils.add_predictions(df)
        summary = csv_utils.summarize_predictions(df_pred)
        sample = df_pred.head(20).to_dict(orient="records")
        duration_ms = int((time.perf_counter() - t0) * 1000)
        filename_prefix = "scored" if has_label else "predicted"
        filename = f"{filename_prefix}_{uuid.uuid4().hex}.csv"
        saved_path = csv_utils.save_dataframe_csv(df_pred, filename)
        file_url = csv_utils.build_download_url(saved_path)

        if has_label:
            macro, per_class, support = csv_utils.score_predictions(df_pred)
            cm = csv_utils.confusion(
                df_pred["label"].astype(int).tolist(),
                df_pred["predicted_label"].astype(int).tolist(),
            )
            csv_utils.persist_batch(
                file_name=file.filename or "upload.csv",
                purpose="score",
                total_rows=len(df_pred),
                class_counts=summary["class_counts"],
                macro_f1=macro,
                f1_per_class=per_class,
                support=support,
                output_path=str(saved_path),
                df_with_preds=df_pred,
            )
            return AnalyzeCsvResponse(
                mode="score",
                summary=summary,
                sample=sample,
                file_url=file_url,
                processing_time_ms=duration_ms,
                macro_f1=macro,
                f1_per_class=per_class,
                support=support,
                confusion_matrix=cm,
            )

        csv_utils.persist_batch(
            file_name=file.filename or "upload.csv",
            purpose="predict",
            total_rows=summary["total_rows"],
            class_counts=summary["class_counts"],
            output_path=str(saved_path),
            df_with_preds=df_pred,
        )
        return AnalyzeCsvResponse(
            mode="predict",
            summary=summary,
            sample=sample,
            file_url=file_url,
            processing_time_ms=duration_ms,
        )
    finally:
        if TASK_SEM:
            TASK_SEM.release()


@app.get("/download/{filename}")
def download_file(filename: str):
    # Простейшая безопасная раздача файлов из STORAGE_DIR.
    target = (STORAGE_DIR / filename).resolve()
    if STORAGE_DIR not in target.parents:
        raise HTTPException(status_code=400, detail="Invalid path")
    if not target.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(target, media_type="text/csv", filename=filename)


def enforce_rate_limit(request: Request):
    if RATE_LIMIT_PER_MIN <= 0:
        return
    client = request.client.host if request.client else "unknown"
    now = time.time()
    with _rate_lock:
        window_start, count = _rate_map.get(client, (now, 0))
        if now - window_start > RATE_LIMIT_WINDOW_SEC:
            window_start = now
            count = 0
        count += 1
        _rate_map[client] = (window_start, count)
        if count > RATE_LIMIT_PER_MIN:
            raise HTTPException(status_code=429, detail="Превышен лимит запросов, попробуйте позже")
