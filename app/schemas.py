from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class MushroomFeatures(BaseModel):
    cap_diameter: float = Field(..., description="Диаметр шляпки")
    cap_shape: str = Field(..., description="Форма шляпки")
    cap_surface: str = Field(..., description="Поверхность шляпки")
    cap_color: str = Field(..., description="Цвет шляпки")
    does_bruise_or_bleed: str = Field(..., description="Синяки или кровотечение")
    gill_attachment: str = Field(..., description="Прикрепление жабр")
    gill_spacing: str = Field(..., description="Расстояние между жабрами")
    gill_color: str = Field(..., description="Цвет жабр")
    stem_height: float = Field(..., description="Высота ножки")
    stem_width: float = Field(..., description="Ширина ножки")
    stem_root: str = Field(..., description="Корень ножки")
    stem_surface: str = Field(..., description="Поверхность ножки")
    stem_color: str = Field(..., description="Цвет ножки")
    veil_type: str = Field(..., description="Тип покрывала")
    veil_color: str = Field(..., description="Цвет покрывала")
    has_ring: str = Field(..., description="Наличие кольца")
    ring_type: str = Field(..., description="Тип кольца")
    spore_print_color: str = Field(..., description="Цвет спорового отпечатка")
    habitat: str = Field(..., description="Среда обитания")
    season: str = Field(..., description="Сезон")

class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="Предсказание (0 - съедобный, 1 - ядовитый)")
    probabilities: Optional[Dict[str, float]] = Field(None, description="Вероятности классов")

class BatchPredictionRequest(BaseModel):
    mushrooms: List[MushroomFeatures] = Field(..., description="Список грибов для классификации")

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse] = Field(..., description="Предсказания для всех грибов")

class ModelStatus(BaseModel):
    model_loaded: bool = Field(..., description="Модель загружена")
    train_date: Optional[datetime] = Field(None, description="Дата обучения модели")
    feature_count: Optional[int] = Field(None, description="Количество признаков")

class FitRequest(BaseModel):
    data: List[Dict[str, Any]] = Field(..., description="Данные для переобучения модели")
    target_column: str = Field(..., description="Название целевой колонки")

class FitResponse(BaseModel):
    success: bool = Field(..., description="Успех переобучения")
    message: str = Field(..., description="Сообщение о результате")
    train_date: Optional[datetime] = Field(None, description="Дата переобучения")
