# Веб‑сервис анализа тональности отзывов

Минимальный сервис: FastAPI + scikit‑learn модель (TF‑IDF word+char + LinearSVC/LogReg), фронтенд на чистом HTML/CSS/JS. Принимает текст или CSV, предсказывает тональность (0/1/2), отдаёт размеченный CSV, умеет считать macro-F1 по загруженному датасету. Есть заглушка для БД (PostgreSQL) и сохранения batch‑результатов.  
Классы: 0 — нейтрально, 1 — позитив, 2 — негатив (учтите при интерпретации).

## Что даёт сервис
- Единичный текст: `/predict_text`.
- CSV без меток: `/predict_csv` — добавляет `predicted_label`, возвращает файл/summary, умеет stream.
- CSV с метками: `/score` — считает macro-F1, F1 по классам, confusion matrix, отдаёт файл с предсказаниями.
- Фронт: `frontend/index.html` — формы для всех сценариев, drag&drop CSV, отображение статуса API.

## Быстрый старт
```
python -m venv venv && source venv/bin/activate
pip install -r backend/requirements.txt
python ml/train_model.py          # обучает модель и кладёт в backend/model_artifacts/model.joblib
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
# или ./run_local.sh (поднимет API на 8000 и статику фронта на 8080)
```
Открыть фронт: http://localhost:8080/index.html (или прямо файл в браузере). Swagger: http://localhost:8000/docs, health: http://localhost:8000/health.

## Данные и обучение
- Ожидаемые файлы: `ml/data/train.csv` (ID, text, src, label), `ml/data/test.csv` (ID, text, src).
- Обучение: `python ml/train_model.py --model svm|logreg ...` (CLI-параметры для C, max_features и т.д.).
- Оценка: `python ml/evaluate_model.py --data ml/data/train.csv`.
- Сабмит: `python ml/make_submission.py --output ml/data/submission.csv`.

## Бэкенд и конфиг
- Эндпоинты: `/health`, `/predict_text`, `/predict_many`, `/predict_csv`, `/score`, `/download/{file}`.
- ENV (главные): `DATABASE_URL` (опц., PostgreSQL), `CORS_ALLOW_ORIGINS`, `MAX_CONCURRENT_TASKS`, `RATE_LIMIT_PER_MIN`, `PREDICT_SCORES_MAX_ROWS`, `STORAGE_TTL_HOURS/STORAGE_MAX_FILES`, `DOWNLOAD_BASE_URL`, `MAX_CSV_BYTES/MAX_CSV_ROWS` (0 — без лимитов).
- БД‑схема: `database/schema.sql` (batches, reviews, model_metrics, sources). Если `DATABASE_URL` не задан, записи в БД пропускаются.

## Docker
- Билд API: `docker compose -f docker/docker-compose.yml build`
- Запуск: `docker compose -f docker/docker-compose.yml up`
Фронт — статичный, можно раздавать любым HTTP‑сервером или с того же хоста (Nginx).

## Структура
- `backend/app` — FastAPI, утилиты CSV, конфиг.
- `backend/model_artifacts` — обученная модель (`model.joblib`).
- `frontend` — UI.
- `ml` — скрипты обучения/оценки/сабмита.
- `database/schema.sql` — DDL для PostgreSQL.
