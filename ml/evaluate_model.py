"""
Оценка сохранённой модели на произвольном CSV с метками.

Запуск из корня проекта:
    python ml/evaluate_model.py --data ml/data/train.csv
"""

import argparse
from pathlib import Path
from typing import Dict

import joblib
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "backend" / "model_artifacts" / "model.joblib"

import sys

sys.path.append(str(ROOT))
from backend.app.preprocessing import normalize_text  # noqa: E402


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns=lambda c: c.lstrip("\ufeff"))
    missing = {"text", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"Отсутствуют колонки {missing} в {path}")
    df["text"] = df["text"].fillna("").astype(str).apply(normalize_text)
    df["label"] = df["label"].astype(int)
    return df


def evaluate(model, df: pd.DataFrame) -> Dict[str, float]:
    preds = model.predict(df["text"])
    macro = float(f1_score(df["label"], preds, average="macro"))
    per_class_arr = f1_score(df["label"], preds, average=None, labels=[0, 1, 2])
    per_class = {str(i): float(score) for i, score in enumerate(per_class_arr)}
    return {"macro_f1": macro, "f1_per_class": per_class}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Путь к CSV с колонками text,label")
    args = parser.parse_args()

    data_path = Path(args.data)
    df = load_dataset(data_path)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Модель не найдена: {MODEL_PATH}. Обучите её через ml/train_model.py")

    model = joblib.load(MODEL_PATH)
    metrics = evaluate(model, df)
    print(f"Macro-F1: {metrics['macro_f1']:.4f}")
    print("Per-class F1:", metrics["f1_per_class"])


if __name__ == "__main__":
    main()
