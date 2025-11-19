import os
import pickle
from typing import Any, Dict, List, Tuple
from pydantic import ValidationError

import numpy as np
import pandas as pd

from .schemas import MushroomFeatures

class MushroomModel:
    """Модель для классификации съедобных и ядовитых грибов."""
    def __init__(self):
        """Инициализация модели."""
        self.model = None
        self.label_encoders = None
        self.features = None
        self.numeric_features = None
        self.categorical_features = None
        self.train_date = None
        self.feature_names = None
        self.preprocessors = None
        self.model_path = "mushroom_model.pkl"
        self.allowed_categories = {}

    def load_model(self) -> bool:
        """Загружает модель из файла."""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)

                self.model = model_data['model']
                self.label_encoders = model_data['label_encoders']
                self.features = model_data['features']
                self.numeric_features = model_data['numeric_features']
                self.categorical_features = model_data['categorical_features']
                self.train_date = model_data['train_date']
                self.feature_names = model_data.get('feature_names', self.features)
                self.preprocessors = model_data.get('preprocessors', {})
                self._build_allowed_categories()
                print(f"Модель загружена. Дата обучения: {self.train_date}")
                return True
            else:
                print("Файл модели не найден")
                return False
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            return False
        
    def _build_allowed_categories(self):
        """Строит словарь допустимых категориальных значений."""
        if self.label_encoders:
            for feature, encoder in self.label_encoders.items():
                self.allowed_categories[feature] = set(encoder.classes_)

    def _validate_categorical_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Проверяет валидность категориальных признаков."""
        validated_features = features.copy()
        
        for feature in self.categorical_features:
            if feature in validated_features:
                value = str(validated_features[feature])
                if (feature in self.allowed_categories and 
                    value not in self.allowed_categories[feature]):
                    print(f"Предупреждение: недопустимое значение '{value}' для признака '{feature}'. "
                          f"Допустимые значения: {list(self.allowed_categories[feature])}")
                    validated_features[feature] = 'unknown'
        
        return validated_features

    def preprocess_features(self, features: Dict[str, Any]) -> np.ndarray:
        """Предобрабатывает признаки для одного гриба."""
        try:
            validated_features = self._validate_categorical_features(features)
            
            feature_dict = {}
            for feature in self.features:
                feature_key = feature.replace('_', '-')
                if feature_key in validated_features:
                    feature_dict[feature] = [validated_features[feature_key]]
                else:
                    if feature in self.numeric_features:
                        feature_dict[feature] = [0.0]
                    else:
                        feature_dict[feature] = ['unknown']

            df = pd.DataFrame(feature_dict)

            if 'numeric_imputer' in self.preprocessors and self.numeric_features:
                df[self.numeric_features] = self.preprocessors['numeric_imputer'].transform(df[self.numeric_features])

            for col in self.categorical_features:
                if col in df.columns and col in self.label_encoders:
                    encoder = self.label_encoders[col]
                    df[col] = df[col].apply(lambda x: x if str(x) in encoder.classes_ else 'unknown')
                    df[col] = encoder.transform(df[col].astype(str))

            if 'scaler' in self.preprocessors and self.numeric_features:
                df[self.numeric_features] = self.preprocessors['scaler'].transform(df[self.numeric_features])

            return df.values

        except Exception as e:
            print(f"Ошибка препроцессинга: {e}")
            raise


    def predict(self, features: Dict[str, Any]) -> Tuple[int, Dict[str, float]]:
        """Предсказывает класс гриба и вероятности."""
        if self.model is None:
            raise ValueError("Модель не загружена")
        try:
            validated_features = MushroomFeatures(**features).dict()
        except ValidationError as e:
            raise ValueError(f"Невалидные данные: {e}")

        X = self.preprocess_features(features)
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]

        prob_dict = {
            'edible': float(probabilities[0]),
            'poisonous': float(probabilities[1])
        }

        return int(prediction), prob_dict
    
    def preprocess_batch(self, features_list: List[Dict[str, Any]]) -> np.ndarray:
        """Предобработка всего батча данных"""
        try:
            processed_data = []
            
            for features in features_list:
                validated_features = self._validate_categorical_features(features)
                
                feature_dict = {}
                for feature in self.features:
                    feature_key = feature.replace('_', '-')
                    if feature_key in validated_features:
                        feature_dict[feature] = [validated_features[feature_key]]
                    else:
                        if feature in self.numeric_features:
                            feature_dict[feature] = [0.0]
                        else:
                            feature_dict[feature] = ['unknown']
                
                processed_data.append(feature_dict)
            
            df_list = []
            for feature_dict in processed_data:
                df_list.append(pd.DataFrame(feature_dict))
            
            df_batch = pd.concat(df_list, ignore_index=True)
            
            if 'numeric_imputer' in self.preprocessors and self.numeric_features:
                df_batch[self.numeric_features] = self.preprocessors['numeric_imputer'].transform(df_batch[self.numeric_features])
            
            for col in self.categorical_features:
                if col in df_batch.columns and col in self.label_encoders:
                    encoder = self.label_encoders[col]
                    df_batch[col] = df_batch[col].apply(lambda x: x if str(x) in encoder.classes_ else 'unknown')
                    df_batch[col] = encoder.transform(df_batch[col].astype(str))

            if 'scaler' in self.preprocessors and self.numeric_features:
                df_batch[self.numeric_features] = self.preprocessors['scaler'].transform(df_batch[self.numeric_features])
            
            return df_batch.values
            
        except Exception as e:
            print(f"Ошибка batch препроцессинга: {e}")
            raise

    def predict_batch(self, features_list: List[Dict[str, Any]]) -> List[Tuple[int, Dict[str, float]]]:
        """Предсказывает классы для батча грибов."""
        if self.model is None:
            raise ValueError("Модель не загружена")

        try:
                validated_features_list = []
                for features in features_list:
                    validated_features = MushroomFeatures(**features).dict()
                    validated_features_list.append(validated_features)
                X_batch = self.preprocess_batch(features_list)
                predictions = self.model.predict(X_batch)
                probabilities = self.model.predict_proba(X_batch)
                results = []
                for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                    prob_dict = {
                        'edible': float(prob[0]),
                        'poisonous': float(prob[1])
                    }
                    results.append((int(pred),prob_dict))
                return results
        except ValidationError as e:
            raise ValueError(f"Невалидные данные в батче: {e}")
        except Exception as e:
            print(f"Ошибка batch предсказания: {e}")
            return self._predict_sequential(features_list)
        
    def _predict_sequential(self, features_list: List[Dict[str, Any]]) -> List[Tuple[int, Dict[str, float]]]:
        """Метод последовательного предсказания."""
        results = []
        for features in features_list:
            try:
                prediction, probabilities = self.predict(features)
                results.append((prediction, probabilities))
            except Exception as e:
                print(f"Ошибка предсказания для гриба: {e}")
                results.append((0, {'edible': 0.5, 'poisonous': 0.5}))
        return results

    def get_allowed_categories(self) -> Dict[str, List[str]]:
        """Возвращает допустимые значения категориальных признаков."""
        return {feature: list(categories) for feature, categories in self.allowed_categories.items()}

    def retrain_model(self, data: List[Dict[str, Any]], target_column: str) -> bool:
        """Переобучает модель на новых данных."""
        try:
            df = pd.DataFrame(data)
            print(f"Данные для переобучения: {df.shape}")

            if target_column not in df.columns:
                raise ValueError(f"Целевая колонка '{target_column}' не найдена в данных")

            df_columns_normalized = {col: col.replace('-', '_') for col in df.columns}
            df = df.rename(columns=df_columns_normalized)
            print(f"Колонки после нормализации: {list(df.columns)}")

            missing_features = set(self.features) - set(df.columns)
            if missing_features:
                print(f"Предупреждение: отсутствуют признаки: {missing_features}")
                for feature in missing_features:
                    if feature in self.numeric_features:
                        df[feature] = 0.0
                    else:
                        df[feature] = 'unknown'
                        print(f"Добавлен категориальный признак {feature} = 'unknown'")
            X = df[self.features].copy()
            y = df[target_column]

            print(f"Признаки: {X.shape}, Целевая переменная: {y.shape}")

            for col in self.categorical_features:
                if col in X.columns:
                    encoder = self.label_encoders[col]
                    unique_values = X[col].unique()
                    def transform_value(x, enc=encoder):
                        x_str = str(x)
                        return x_str if str(x) in enc.classes_ else 'unknown'
                    X[col] = X[col].apply(transform_value)
                    X[col] = encoder.transform(X[col].astype(str))

            if 'numeric_imputer' in self.preprocessors and self.numeric_features:
                X[self.numeric_features] = self.preprocessors['numeric_imputer'].transform(X[self.numeric_features])
            
            if 'scaler' in self.preprocessors and self.numeric_features:
                X[self.numeric_features] = self.preprocessors['scaler'].transform(X[self.numeric_features])

            X = X.astype(float)

            print("Начинаем переобучение модели...")
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(X, y)

            from datetime import datetime
            self.train_date = datetime.now()

            model_data = {
                'model': self.model,
                'label_encoders': self.label_encoders,
                'preprocessors': self.preprocessors,
                'features': self.features,
                'numeric_features': self.numeric_features,
                'categorical_features': self.categorical_features,
                'train_date': self.train_date,
                'feature_names': self.feature_names
            }

            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)

            print(f"Модель успешно переобучена. Дата: {self.train_date}")
            return True

        except Exception as e:
            print(f"Ошибка переобучения модели: {e}")
            import traceback
            print("Детали ошибки:")
            print(traceback.format_exc())
            return False


mushroom_model = MushroomModel()
