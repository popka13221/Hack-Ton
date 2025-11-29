"""
Обучение модели: TF-IDF word (1-2) + char (3-5) с LinearSVC или LogisticRegression.
Параметры задаются через CLI (см. --help).
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Tuple, Dict

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "ml" / "data"
TRAIN_PATH = DATA_DIR / "train.csv"
MODEL_DIR = ROOT / "backend" / "model_artifacts"
MODEL_PATH = MODEL_DIR / "model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

sys.path.append(str(ROOT))
from backend.app.preprocessing import normalize_text  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Не найден файл {path}")
    logger.info("Читаю датасет: %s", path)
    df = pd.read_csv(path)
    df = df.rename(columns=lambda c: c.lstrip("\ufeff"))
    missing = {"text", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"Отсутствуют колонки {missing} в {path}")
    df["text"] = df["text"].fillna("").astype(str).apply(normalize_text)
    df["label"] = df["label"].astype(int)
    logger.info("Датасет загружен: %d строк", len(df))
    logger.info("Распределение меток: %s", df["label"].value_counts().to_dict())
    return df


def build_pipeline(args) -> Pipeline:
    # Объединяем word и char n-граммы; размеры задаются через CLI.
    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=args.word_max_features,
        min_df=args.word_min_df,
        sublinear_tf=True,
    )
    char_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        max_features=args.char_max_features,
        min_df=args.char_min_df,
        sublinear_tf=True,
    )
    features = FeatureUnion(
        [("word", word_vectorizer), ("char", char_vectorizer)],
    )
    if args.model == "svm":
        clf = LinearSVC(
            C=args.C,
            class_weight="balanced",
        )
    else:
        clf = LogisticRegression(
            C=args.C,
            max_iter=400,
            n_jobs=-1,
            class_weight="balanced",
            solver="saga",
        )
    return Pipeline([("features", features), ("clf", clf)])


def evaluate(model: Pipeline, x_texts, y_true) -> Tuple[float, Dict[str, float]]:
    preds = model.predict(x_texts)
    macro = float(f1_score(y_true, preds, average="macro"))
    per_class_arr = f1_score(y_true, preds, average=None, labels=[0, 1, 2])
    per_class = {str(i): float(score) for i, score in enumerate(per_class_arr)}
    return macro, per_class


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["svm", "logreg"], default="svm")
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--word-max-features", type=int, default=100_000)
    parser.add_argument("--char-max-features", type=int, default=120_000)
    parser.add_argument("--word-min-df", type=int, default=3)
    parser.add_argument("--char-min-df", type=int, default=5)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info(
        "Config: model=%s, C=%.3f, word_max=%d, char_max=%d, word_min_df=%d, char_min_df=%d",
        args.model,
        args.C,
        args.word_max_features,
        args.char_max_features,
        args.word_min_df,
        args.char_min_df,
    )
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset(TRAIN_PATH)
    train_df, val_df = train_test_split(
        df, test_size=args.val_size, stratify=df["label"], random_state=args.random_state
    )
    logger.info("Split train/val: %d / %d", len(train_df), len(val_df))

    pipeline = build_pipeline(args)
    start = time.perf_counter()
    logger.info("Начинаю обучение...")
    pipeline.fit(train_df["text"], train_df["label"])
    logger.info("Обучение завершено за %.2f c", time.perf_counter() - start)

    # Лог размеров словарей.
    features = pipeline.named_steps["features"]
    word_vect = dict(features.transformer_list)["word"]
    char_vect = dict(features.transformer_list)["char"]
    logger.info(
        "Размер словарей: word=%d, char=%d",
        len(word_vect.vocabulary_),
        len(char_vect.vocabulary_),
    )

    macro, per_class = evaluate(pipeline, val_df["text"], val_df["label"])
    logger.info("Val macro-F1: %.4f", macro)
    logger.info("Val F1 per class: %s", per_class)

    joblib.dump(pipeline, MODEL_PATH)
    logger.info("Модель сохранена: %s", MODEL_PATH)

    METRICS_PATH.write_text(
        json.dumps(
            {
                "macro_f1": macro,
                "f1_per_class": per_class,
                "n_train": len(train_df),
                "n_val": len(val_df),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    logger.info("Метрики сохранены: %s", METRICS_PATH)


if __name__ == "__main__":
    main()
