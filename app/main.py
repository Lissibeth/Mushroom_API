
import uvicorn
from fastapi import FastAPI, HTTPException, Query

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
            "predict_proba": "/predict_proba",
            "predict_batch": "/predict_batch",
            "predict_proba_batch": "/predict_proba_batch",
            "status": "/status",
            "docs": "/docs"
        }
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
        features = {
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

        prediction, probabilities = mushroom_model.predict(features)
        return PredictionResponse(
            prediction=prediction,
            probabilities=probabilities
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка предсказания: {str(e)}") from e

@app.get("/predict_proba", response_model=PredictionResponse)
async def predict_proba(
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
    """Вероятность для одного гриба"""
    try:
        features = {
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

        prediction, probabilities = mushroom_model.predict(features)
        return PredictionResponse(
            prediction=prediction,
            probabilities=probabilities
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка предсказания: {str(e)}") from e

@app.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Предсказание для нескольких грибов"""
    try:
        features_list = []
        for mushroom in request.mushrooms:
            features = {
                "cap-diameter": mushroom.cap_diameter,
                "cap-shape": mushroom.cap_shape,
                "cap-surface": mushroom.cap_surface,
                "cap-color": mushroom.cap_color,
                "does-bruise-or-bleed": mushroom.does_bruise_or_bleed,
                "gill-attachment": mushroom.gill_attachment,
                "gill-spacing": mushroom.gill_spacing,
                "gill-color": mushroom.gill_color,
                "stem-height": mushroom.stem_height,
                "stem-width": mushroom.stem_width,
                "stem-root": mushroom.stem_root,
                "stem-surface": mushroom.stem_surface,
                "stem-color": mushroom.stem_color,
                "veil-type": mushroom.veil_type,
                "veil-color": mushroom.veil_color,
                "has-ring": mushroom.has_ring,
                "ring-type": mushroom.ring_type,
                "spore-print-color": mushroom.spore_print_color,
                "habitat": mushroom.habitat,
                "season": mushroom.season
            }
            features_list.append(features)

        results = mushroom_model.predict_batch(features_list)

        predictions = []
        for prediction, probabilities in results:
            predictions.append(PredictionResponse(
                prediction=prediction,
                probabilities=probabilities
            ))

        return BatchPredictionResponse(predictions=predictions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка предсказания: {str(e)}") from e

@app.post("/predict_proba_batch", response_model=BatchPredictionResponse)
async def predict_proba_batch(request: BatchPredictionRequest):
    """Вероятности для нескольких грибов"""
    return await predict_batch(request)

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
