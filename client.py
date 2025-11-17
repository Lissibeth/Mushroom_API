import json
from typing import Any, Dict, List

import requests


class MushroomClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8008"):
        self.base_url = base_url

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Предсказание для одного гриба"""
        response = requests.get(f"{self.base_url}/predict", params=features)
        response.raise_for_status()
        return response.json()

    def predict_proba(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Вероятности для одного гриба"""
        response = requests.get(f"{self.base_url}/predict_proba", params=features)
        response.raise_for_status()
        return response.json()

    def predict_batch(self, mushrooms: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Пакетное предсказание"""
        data = {"mushrooms": mushrooms}
        response = requests.post(f"{self.base_url}/predict_batch", json=data)
        response.raise_for_status()
        return response.json()

    def predict_proba_batch(self, mushrooms: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Пакетные вероятности"""
        data = {"mushrooms": mushrooms}
        response = requests.post(f"{self.base_url}/predict_proba_batch", json=data)
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

        print("\n2. Предсказание для одного гриба:")
        prediction = client.predict(test_mushroom)
        print(json.dumps(prediction, indent=2))

        print("\n3. Вероятности для одного гриба:")
        proba = client.predict_proba(test_mushroom)
        print(json.dumps(proba, indent=2))

        print("\n4. Пакетное предсказание:")
        batch_pred = client.predict_batch([test_mushroom, test_mushroom])
        print(json.dumps(batch_pred, indent=2))

        print("\n5. Пакетные вероятности:")
        batch_proba = client.predict_proba_batch([test_mushroom, test_mushroom])
        print(json.dumps(batch_proba, indent=2))

    except requests.exceptions.RequestException as e:
        print(f"Ошибка подключения: {e}")
        print("Убедитесь, что сервер запущен на http://127.0.0.1:8008")

if __name__ == "__main__":
    main()
