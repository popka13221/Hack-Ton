from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool
    db_connected: bool = False


class PredictTextRequest(BaseModel):
    text: str = Field(..., description="Текст отзыва для классификации")


class PredictTextResponse(BaseModel):
    predicted_label: int
    predicted_class_name: str
    predicted_scores: Optional[Dict[str, float]] = None


class PredictManyRequest(BaseModel):
    texts: List[str] = Field(..., description="Список текстов для классификации")


class PredictManyResponse(BaseModel):
    predicted_labels: List[int]
    predicted_class_names: List[str]


class PredictCsvSummary(BaseModel):
    total_rows: int
    class_counts: Dict[str, int]


class PredictCsvResponse(BaseModel):
    summary: PredictCsvSummary
    sample: List[Dict[str, Any]]
    file_url: Optional[str] = None
    processing_time_ms: Optional[int] = None


class ScoreResponse(BaseModel):
    macro_f1: float
    f1_per_class: Dict[str, float]
    support: Dict[str, int]
    confusion_matrix: List[List[int]]
    file_url: Optional[str] = None
    processing_time_ms: Optional[int] = None
