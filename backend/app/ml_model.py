from pathlib import Path
from typing import List, Optional

import joblib

from . import preprocessing
from .config import MODEL_PATH, CLASS_NAMES


class SentimentModel:
    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        self.model_path = model_path
        self.pipeline = None

    def load(self) -> None:
        self.pipeline = joblib.load(self.model_path)

    @property
    def is_loaded(self) -> bool:
        return self.pipeline is not None

    def predict_text(self, text: str):
        if not self.pipeline:
            raise RuntimeError("Model is not loaded")
        normalized = preprocessing.normalize_text(text)
        label = int(self.pipeline.predict([normalized])[0])
        class_name = CLASS_NAMES.get(label, str(label))
        return label, class_name

    def predict_many(self, texts: List[str]):
        if not self.pipeline:
            raise RuntimeError("Model is not loaded")
        normalized = [preprocessing.normalize_text(t) for t in texts]
        labels = [int(lbl) for lbl in self.pipeline.predict(normalized)]
        class_names = [CLASS_NAMES.get(lbl, str(lbl)) for lbl in labels]
        return labels, class_names


model: Optional[SentimentModel] = None


def get_model() -> SentimentModel:
    global model
    if model is None:
        model = SentimentModel()
        if MODEL_PATH.exists():
            model.load()
    return model
