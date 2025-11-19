
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from typing import Dict, List

from .ml_model import mushroom_model
from .schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    FitRequest,
    FitResponse,
    ModelStatus,
    PredictionResponse,
)

app = FastAPI(
    title="Mushroom Classification API",
    description="API для классификации съедобных и ядовитых грибов",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    """Загрузка модели при старте приложения"""
    mushroom_model.load_model()

@app.get("/")
async def root():
    """Корневая страница"""
    return {
        "message": "Mushroom Classification API",
        "endpoints": {
            "predict": "/predict",
            "predict_batch": "/predict_batch",
            "categories": "/categories",
            "status": "/status",
            "fit": "/fit",
            "docs": "/docs"
        }
    }

@app.get("/categories")
async def get_categories():
    """Получение допустимых значений для категориальных признаков"""
    try:
        if mushroom_model.model is None:
            raise HTTPException(status_code=500, detail="Модель не загружена")
        
        categories = mushroom_model.get_allowed_categories()
        return categories
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения категорий: {str(e)}") from e

def _create_features_dict(
    cap_diameter: float,
    cap_shape: str,
    cap_surface: str,
    cap_color: str,
    does_bruise_or_bleed: str,
    gill_attachment: str,
    gill_spacing: str,
    gill_color: str,
    stem_height: float,
    stem_width: float,
    stem_root: str,
    stem_surface: str,
    stem_color: str,
    veil_type: str,
    veil_color: str,
    has_ring: str,
    ring_type: str,
    spore_print_color: str,
    habitat: str,
    season: str
) -> Dict[str, any]:
    """Создает словарь признаков из параметров"""
    return {
        "cap-diameter": cap_diameter,
        "cap-shape": cap_shape,
        "cap-surface": cap_surface,
        "cap-color": cap_color,
        "does-bruise-or-bleed": does_bruise_or_bleed,
        "gill-attachment": gill_attachment,
        "gill-spacing": gill_spacing,
        "gill-color": gill_color,
        "stem-height": stem_height,
        "stem-width": stem_width,
        "stem-root": stem_root,
        "stem-surface": stem_surface,
        "stem-color": stem_color,
        "veil-type": veil_type,
        "veil-color": veil_color,
        "has-ring": has_ring,
        "ring-type": ring_type,
        "spore-print-color": spore_print_color,
        "habitat": habitat,
        "season": season
    }

@app.get("/predict", response_model=PredictionResponse)
async def predict(
    cap_diameter: float = Query(..., description="Диаметр шляпки"),
    cap_shape: str = Query(..., description="Форма шляпки"),
    cap_surface: str = Query(..., description="Поверхность шляпки"),
    cap_color: str = Query(..., description="Цвет шляпки"),
    does_bruise_or_bleed: str = Query(..., description="Синяки или кровотечение"),
    gill_attachment: str = Query(..., description="Прикрепление жабр"),
    gill_spacing: str = Query(..., description="Расстояние между жабрами"),
    gill_color: str = Query(..., description="Цвет жабр"),
    stem_height: float = Query(..., description="Высота ножки"),
    stem_width: float = Query(..., description="Ширина ножки"),
    stem_root: str = Query(..., description="Корень ножки"),
    stem_surface: str = Query(..., description="Поверхность ножки"),
    stem_color: str = Query(..., description="Цвет ножки"),
    veil_type: str = Query(..., description="Тип покрывала"),
    veil_color: str = Query(..., description="Цвет покрывала"),
    has_ring: str = Query(..., description="Наличие кольца"),
    ring_type: str = Query(..., description="Тип кольца"),
    spore_print_color: str = Query(..., description="Цвет спорового отпечатка"),
    habitat: str = Query(..., description="Среда обитания"),
    season: str = Query(..., description="Сезон")
):
    """Предсказание для одного гриба"""
    try:
        features = _create_features_dict(
            cap_diameter, cap_shape, cap_surface, cap_color,
            does_bruise_or_bleed, gill_attachment, gill_spacing, gill_color,
            stem_height, stem_width, stem_root, stem_surface, stem_color,
            veil_type, veil_color, has_ring, ring_type,
            spore_print_color, habitat, season
        )

        prediction, probabilities = mushroom_model.predict(features)
        return PredictionResponse(
            prediction=prediction,
            probabilities=probabilities
        )
    except ValueError as e:
            raise HTTPException(status_code=422, detail=f"Ошибка валидации данных: {str(e)}")
    except Exception as e:
            raise HTTPException(status_code=500, detail=f"Ошибка предсказания: {str(e)}") from e

@app.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_batch(
        request: BatchPredictionRequest,
        return_probabilities: bool = Query(True, description="Возвращать вероятности")
):
    """Предсказание для нескольких грибов"""
    try:
        features_list = []
        for mushroom in request.mushrooms:
            features = _create_features_dict(
                mushroom.cap_diameter, mushroom.cap_shape, mushroom.cap_surface, mushroom.cap_color,
                mushroom.does_bruise_or_bleed, mushroom.gill_attachment, mushroom.gill_spacing, mushroom.gill_color,
                mushroom.stem_height, mushroom.stem_width, mushroom.stem_root, mushroom.stem_surface, mushroom.stem_color,
                mushroom.veil_type, mushroom.veil_color, mushroom.has_ring, mushroom.ring_type,
                mushroom.spore_print_color, mushroom.habitat, mushroom.season
            )
            features_list.append(features)

        results = mushroom_model.predict_batch(features_list)

        predictions = []
        for prediction, probabilities in results:
            predictions.append(PredictionResponse(
                prediction=prediction,
                probabilities=probabilities if return_probabilities else None
            ))

        return BatchPredictionResponse(predictions=predictions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка предсказания: {str(e)}") from e


@app.get("/status", response_model=ModelStatus)
async def status():
    """Статус модели"""
    return ModelStatus(
        model_loaded=mushroom_model.model is not None,
        train_date=mushroom_model.train_date,
        feature_count=len(mushroom_model.features) if mushroom_model.features else None
    )

@app.post("/fit", response_model=FitResponse)
async def fit_model(request: FitRequest):
    """Переобучение модели на новых данных"""
    try:
        success = mushroom_model.retrain_model(request.data, request.target_column)
        if success:
            return FitResponse(
                success=True,
                message="Модель успешно переобучена",
                train_date=mushroom_model.train_date
            )
        else:
            return FitResponse(
                success=False,
                message="Ошибка переобучения модели"
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка переобучения: {str(e)}") from e

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8008, reload=True)
