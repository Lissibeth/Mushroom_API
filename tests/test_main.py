from app.schemas import BatchPredictionResponse, ModelStatus, PredictionResponse


class TestMainEndpoints:
    """Тесты для эндпоинтов main.py"""

    def test_root_endpoint(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "endpoints" in data

    def test_status_endpoint(self, client):
        response = client.get("/status")
        assert response.status_code == 200
        status_data = ModelStatus(**response.json())
        assert isinstance(status_data.model_loaded, bool)

    def test_predict_endpoint_success(self, client, sample_mushroom_params):
        response = client.get("/predict", params=sample_mushroom_params)
        if response.status_code == 200:
            prediction_data = PredictionResponse(**response.json())
            assert prediction_data.prediction in [0, 1]
        else:
            assert response.status_code == 500
            data = response.json()
            assert "detail" in data

    def test_predict_proba_endpoint_success(self, client, sample_mushroom_params):
        response = client.get("/predict_proba", params=sample_mushroom_params)
        if response.status_code == 200:
            prediction_data = PredictionResponse(**response.json())
            probs = prediction_data.probabilities
            total = probs["edible"] + probs["poisonous"]
            assert abs(total - 1.0) < 0.001
        else:
            assert response.status_code == 500

    def test_predict_batch_endpoint_success(self, client, sample_batch_data):
        response = client.post("/predict_batch", json=sample_batch_data)
        if response.status_code == 200:
            batch_response = BatchPredictionResponse(**response.json())
            assert len(batch_response.predictions) == 1
        else:
            assert response.status_code == 500

    def test_predict_proba_batch_endpoint_success(self, client, sample_batch_data):
        response = client.post("/predict_proba_batch", json=sample_batch_data)
        if response.status_code == 200:
            batch_response = BatchPredictionResponse(**response.json())
            assert len(batch_response.predictions) == 1
        else:
            assert response.status_code == 500

    def test_predict_missing_parameters(self, client):
        response = client.get("/predict", params={"cap_diameter": 5.0})
        assert response.status_code == 422

    def test_invalid_parameter_types(self, client):
        invalid_params = {
            "cap_diameter": "invalid",
            "cap_shape": "b", "cap_surface": "f", "cap_color": "n",
            "does_bruise_or_bleed": "t", "gill_attachment": "a",
            "gill_spacing": "c", "gill_color": "b", "stem_height": 7.0,
            "stem_width": 1.0, "stem_root": "b", "stem_surface": "f",
            "stem_color": "n", "veil_type": "p", "veil_color": "n",
            "has_ring": "t", "ring_type": "c", "spore_print_color": "k",
            "habitat": "g", "season": "a"
        }
        response = client.get("/predict", params=invalid_params)
        assert response.status_code == 422


def test_swagger_docs_available(client):
    response = client.get("/docs")
    assert response.status_code == 200

def test_redoc_available(client):
    response = client.get("/redoc")
    assert response.status_code == 200
