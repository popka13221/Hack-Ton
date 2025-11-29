"""
Генерация submission-файла по test.csv с помощью обученной модели.

Запуск из корня проекта:
    python ml/make_submission.py --output ml/data/submission.csv
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd

import sys

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "backend" / "model_artifacts" / "model.joblib"
TEST_PATH = ROOT / "ml" / "data" / "test.csv"

sys.path.append(str(ROOT))
from backend.app.preprocessing import normalize_text  # noqa: E402


def load_test(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns=lambda c: c.lstrip("\ufeff"))
    missing = {"ID", "text"} - set(df.columns)
    if missing:
        raise ValueError(f"Отсутствуют колонки {missing} в {path}")
    df["text"] = df["text"].fillna("").astype(str).apply(normalize_text)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=str(ROOT / "ml" / "data" / "submission.csv"),
        help="Куда сохранить сабмит",
    )
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise FileNotFoundError("Сначала обучите модель: python ml/train_model.py")

    model = joblib.load(MODEL_PATH)
    test_df = load_test(TEST_PATH)
    preds = model.predict(test_df["text"])
    out_df = pd.DataFrame({"ID": test_df["ID"], "label": preds})
    out_path = Path(args.output)
    out_df.to_csv(out_path, index=False)
    print(f"Submission сохранён в {out_path}")


if __name__ == "__main__":
    main()
