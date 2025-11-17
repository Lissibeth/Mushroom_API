import pytest
from fastapi.testclient import TestClient

try:
    from app.main import app
    from app.ml_model import mushroom_model
except ImportError:
    import os
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from app.main import app
    from app.ml_model import mushroom_model

@pytest.fixture(autouse=True)
def load_model_for_tests():
    mushroom_model.load_model()
    yield

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def sample_mushroom_params():
    return {
        "cap_diameter": 5.0, "cap_shape": "b", "cap_surface": "f",
        "cap_color": "n", "does_bruise_or_bleed": "t", "gill_attachment": "a",
        "gill_spacing": "c", "gill_color": "b", "stem_height": 7.0,
        "stem_width": 1.0, "stem_root": "b", "stem_surface": "f",
        "stem_color": "n", "veil_type": "p", "veil_color": "n",
        "has_ring": "t", "ring_type": "c", "spore_print_color": "k",
        "habitat": "g", "season": "a"
    }

@pytest.fixture
def sample_batch_data():
    return {
        "mushrooms": [{
            "cap_diameter": 5.0, "cap_shape": "b", "cap_surface": "f",
            "cap_color": "n", "does_bruise_or_bleed": "t", "gill_attachment": "a",
            "gill_spacing": "c", "gill_color": "b", "stem_height": 7.0,
            "stem_width": 1.0, "stem_root": "b", "stem_surface": "f",
            "stem_color": "n", "veil_type": "p", "veil_color": "n",
            "has_ring": "t", "ring_type": "c", "spore_print_color": "k",
            "habitat": "g", "season": "a"
        }]
    }
