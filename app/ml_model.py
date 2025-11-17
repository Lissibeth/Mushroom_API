import os
import pickle
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class MushroomModel:
    def __init__(self):
        self.model = None
        self.label_encoders = None
        self.features = None
        self.numeric_features = None
        self.categorical_features = None
        self.train_date = None
        self.feature_names = None
        self.model_path = "mushroom_model.pkl"

    def load_model(self) -> bool:
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
                print(f"Модель загружена. Дата обучения: {self.train_date}")
                return True
            else:
                print("Файл модели не найден")
                return False
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            return False

    def preprocess_features(self, features: Dict[str, Any]) -> np.ndarray:
        try:
            feature_dict = {}
            for feature in self.features:
                feature_key = feature.replace('_', '-')
                if feature_key in features:
                    feature_dict[feature] = [features[feature_key]]
                else:
                    if feature in self.numeric_features:
                        feature_dict[feature] = [0.0]
                    else:
                        feature_dict[feature] = ['unknown']

            df = pd.DataFrame(feature_dict)

            for col in self.categorical_features:
                if col in df.columns:
                    encoder = self.label_encoders[col]
                    def transform_value(x, enc=encoder):
                        return x if str(x) in enc.classes_ else 'unknown'
                    df[col] = df[col].apply(transform_value)

            if self.numeric_features:
                scaler = StandardScaler()
                df[self.numeric_features] = scaler.fit_transform(df[self.numeric_features])

            return df.values

        except Exception as e:
            print(f"Ошибка препроцессинга: {e}")
            raise

    def predict(self, features: Dict[str, Any]) -> Tuple[int, Dict[str, float]]:
        if self.model is None:
            raise ValueError("Модель не загружена")

        X = self.preprocess_features(features)
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]

        prob_dict = {
            'edible': float(probabilities[0]),
            'poisonous': float(probabilities[1])
        }

        return int(prediction), prob_dict

    def predict_batch(self, features_list: List[Dict[str, Any]]) -> List[Tuple[int, Dict[str, float]]]:
        if self.model is None:
            raise ValueError("Модель не загружена")

        results = []
        for features in features_list:
            try:
                prediction, probabilities = self.predict(features)
                results.append((prediction, probabilities))
            except Exception as e:
                print(f"Ошибка предсказания для гриба: {e}")
                results.append((0, {'edible': 0.5, 'poisonous': 0.5}))

        return results

    def retrain_model(self, data: List[Dict[str, Any]], target_column: str) -> bool:
        """Переобучение модели на новых данных"""
        try:
            df = pd.DataFrame(data)
            print(f"Данные для переобучения: {df.shape}")

            if target_column not in df.columns:
                raise ValueError(f"Целевая колонка '{target_column}' не найдена в данных")

            X = df[self.features].copy()
            y = df[target_column]

            print(f"Признаки: {X.shape}, Целевая переменная: {y.shape}")

            for col in self.categorical_features:
                if col in X.columns:
                    encoder = self.label_encoders[col]
                    def transform_value(x, enc=encoder):
                        return x if str(x) in enc.classes_ else 'unknown'
                    X[col] = X[col].apply(transform_value)

            if self.numeric_features:
                scaler = StandardScaler()
                X[self.numeric_features] = scaler.fit_transform(X[self.numeric_features])

            X = X.fillna(0)

            print("Начинаем переобучение модели...")
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(
                n_estimators=50,
                max_depth=15,
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(X, y)

            from datetime import datetime
            self.train_date = datetime.now()

            model_data = {
                'model': self.model,
                'label_encoders': self.label_encoders,
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
            return False


mushroom_model = MushroomModel()
