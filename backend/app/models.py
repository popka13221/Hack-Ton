from typing import List, Optional, Dict, Any, Literal
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


class AnalyzeCsvResponse(BaseModel):
    mode: Literal["predict", "score"]
    summary: Optional[PredictCsvSummary] = None
    sample: Optional[List[Dict[str, Any]]] = None
    file_url: Optional[str] = None
    processing_time_ms: Optional[int] = None
    macro_f1: Optional[float] = None
    f1_per_class: Optional[Dict[str, float]] = None
    support: Optional[Dict[str, int]] = None
    confusion_matrix: Optional[List[List[int]]] = None


class AnalyzeTaskStatus(BaseModel):
    task_id: str
    status: Literal["processing", "completed", "error"]
    result: Optional[AnalyzeCsvResponse] = None
    error: Optional[str] = None
