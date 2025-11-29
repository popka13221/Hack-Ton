from pathlib import Path
import os


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model_artifacts"
MODEL_PATH = MODEL_DIR / "model.joblib"
VECTORIZER_PATH = MODEL_DIR / "vectorizer.joblib"

# Simple mapping for class names; keep in sync with ML pipeline.
CLASS_NAMES = {
    0: "negative",
    1: "neutral",
    2: "positive",
}

# Статистика/выгрузки CSV будут складываться сюда.
STORAGE_DIR = BASE_DIR / "storage"
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

# База (опционально).
DATABASE_URL = os.getenv("DATABASE_URL")
MAX_REVIEWS_TO_SAVE = int(os.getenv("MAX_REVIEWS_TO_SAVE", "0"))  # 0 — не сохранять

# Базовый URL для скачивания файлов (если фронт крутится отдельно). Если не задан,
# возвращаем относительный путь /download/<file>.
DOWNLOAD_BASE_URL = os.getenv("DOWNLOAD_BASE_URL")

# Ограничения для CSV (0 — без лимитов).
MAX_CSV_BYTES = int(os.getenv("MAX_CSV_BYTES", "0"))
MAX_CSV_ROWS = int(os.getenv("MAX_CSV_ROWS", "0"))

# CORS origins (через запятую). По умолчанию — все.
_origins_env = os.getenv("CORS_ALLOW_ORIGINS", "*")
CORS_ALLOW_ORIGINS = (
    ["*"]
    if _origins_env.strip() == "*"
    else [o.strip() for o in _origins_env.split(",") if o.strip()]
)

# Очистка storage.
STORAGE_TTL_HOURS = int(os.getenv("STORAGE_TTL_HOURS", "0"))  # 0 — не чистить по времени
STORAGE_MAX_FILES = int(os.getenv("STORAGE_MAX_FILES", "0"))  # 0 — не ограничивать количество

# Параллельные задачи (0 — без ограничения).
MAX_CONCURRENT_TASKS = int(os.getenv("MAX_CONCURRENT_TASKS", "0"))
TASK_ACQUIRE_TIMEOUT = float(os.getenv("TASK_ACQUIRE_TIMEOUT", "1.0"))

# Предсказанные вероятности/скоринг (0 — не считать).
PREDICT_SCORES_MAX_ROWS = int(os.getenv("PREDICT_SCORES_MAX_ROWS", "0"))

# Простой rate-limit (внутрипроцессный) для тяжёлых эндпоинтов, запросов в минуту на IP.
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "0"))  # 0 — выкл
RATE_LIMIT_WINDOW_SEC = int(os.getenv("RATE_LIMIT_WINDOW_SEC", "60"))
