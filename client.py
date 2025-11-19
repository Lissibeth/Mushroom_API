import json
from typing import Any, Dict, List

import requests

class MushroomClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8008"):
        self.base_url = base_url

    def predict(
        self, 
        features: Dict[str, Any], 
        return_probabilities: bool = True
    ) -> Dict[str, Any]:
        """Предсказание для одного гриба"""
        params = features.copy()
        params["return_probabilities"] = return_probabilities
        response = requests.get(f"{self.base_url}/predict", params=params)
        response.raise_for_status()
        return response.json()

    def predict_batch(
        self, 
        mushrooms: List[Dict[str, Any]], 
        return_probabilities: bool = True
    ) -> Dict[str, Any]:
        """Пакетное предсказание"""
        params = {"return_probabilities": return_probabilities}
        data = {"mushrooms": mushrooms}
        response = requests.post(f"{self.base_url}/predict_batch", params=params, json=data)
        response.raise_for_status()
        return response.json()

    def get_categories(self) -> Dict[str, Any]:
        """Получить допустимые значения категориальных признаков"""
        response = requests.get(f"{self.base_url}/categories")
        response.raise_for_status()
        return response.json()

    def get_status(self) -> Dict[str, Any]:
        """Статус модели"""
        response = requests.get(f"{self.base_url}/status")
        response.raise_for_status()
        return response.json()

    def fit_model(self, data: List[Dict[str, Any]], target_column: str = "target") -> Dict[str, Any]:
        """Переобучение модели"""
        fit_data = {
            "data": data,
            "target_column": target_column
        }
        response = requests.post(f"{self.base_url}/fit", json=fit_data)
        response.raise_for_status()
        return response.json()


def main():
    """Пример использования клиента"""
    client = MushroomClient()

    test_mushroom = {
        "cap_diameter": 5.0,
        "cap_shape": "b",
        "cap_surface": "f",
        "cap_color": "n",
        "does_bruise_or_bleed": "t",
        "gill_attachment": "a",
        "gill_spacing": "c",
        "gill_color": "b",
        "stem_height": 7.0,
        "stem_width": 1.0,
        "stem_root": "b",
        "stem_surface": "f",
        "stem_color": "n",
        "veil_type": "p",
        "veil_color": "n",
        "has_ring": "t",
        "ring_type": "c",
        "spore_print_color": "k",
        "habitat": "g",
        "season": "a"
    }

    try:
        print("1. Статус модели:")
        status = client.get_status()
        print(json.dumps(status, indent=2, default=str))

        print("\n2. Допустимые категории:")
        categories = client.get_categories()
        print(json.dumps(categories, indent=2))

        print("\n3. Предсказание для одного гриба (с вероятностями):")
        prediction = client.predict(test_mushroom, return_probabilities=True)
        print(json.dumps(prediction, indent=2))

        print("\n4. Предсказание для одного гриба (только класс):")
        prediction_class_only = client.predict(test_mushroom, return_probabilities=False)
        print(json.dumps(prediction_class_only, indent=2))

        print("\n5. Пакетное предсказание:")
        batch_pred = client.predict_batch([test_mushroom, test_mushroom], return_probabilities=True)
        print(json.dumps(batch_pred, indent=2))

        print("\n6. Переобучение модели (пример):")
        training_data = [
            {**test_mushroom, "target": 0},
            {**test_mushroom, "target": 1}
        ]
        try:
            fit_result = client.fit_model(training_data, "target")
            print(json.dumps(fit_result, indent=2))
        except Exception as e:
            print(f"Переобучение не удалось (нормально для теста): {e}")

    except requests.exceptions.RequestException as e:
        print(f"Ошибка подключения: {e}")
        print("Убедитесь, что сервер запущен на http://127.0.0.1:8008")


if __name__ == "__main__":
    main()