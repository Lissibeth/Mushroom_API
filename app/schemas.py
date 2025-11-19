from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator

class MushroomFeatures(BaseModel):
    cap_diameter: float = Field(..., ge=0, le=50, description="Диаметр шляпки (0-50 см)")
    cap_shape: str = Field(..., description="Форма шляпки")
    cap_surface: str = Field(..., description="Поверхность шляпки")
    cap_color: str = Field(..., description="Цвет шляпки")
    does_bruise_or_bleed: str = Field(..., description="Синяки или кровотечение")
    gill_attachment: str = Field(..., description="Прикрепление жабр")
    gill_spacing: str = Field(..., description="Расстояние между жабрами")
    gill_color: str = Field(..., description="Цвет жабр")
    stem_height: float = Field(..., ge=0, le=30, description="Высота ножки (0-30 см)")
    stem_width: float = Field(..., ge=0, le=10, description="Ширина ножки (0-10 см)")
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

    @validator('cap_shape', 'cap_surface', 'cap_color', 'does_bruise_or_bleed', 
                'gill_attachment', 'gill_spacing', 'gill_color', 'stem_root',
                'stem_surface', 'stem_color', 'veil_type', 'veil_color', 
                'has_ring', 'ring_type', 'spore_print_color', 'habitat', 'season')
    def validate_single_letter(cls, v):
        """Проверяет, что значение - одна буква"""
        if not isinstance(v, str) or len(v) != 1 or not v.isalpha():
            raise ValueError(f"Value must be a single letter, got '{v}'")
        return v
    @validator('does_bruise_or_bleed', 'has_ring')
    def validate_boolean_flags(cls, v):
        """Проверяет флаги t/f"""
        if v not in ['t', 'f']:
            raise ValueError(f"Value must be 't' or 'f', got '{v}'")
        return v

    @validator('season')
    def validate_season(cls, v):
        """Проверяет сезон"""
        allowed_seasons = ['a', 's', 'u', 'w']
        if v not in allowed_seasons:
            raise ValueError(f"Season must be one of {allowed_seasons}, got '{v}'")
        return v
    
class CategoriesResponse(BaseModel):
    categories: Dict[str, List[str]] = Field(..., description="Допустимые значения для категориальных признаков")

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
