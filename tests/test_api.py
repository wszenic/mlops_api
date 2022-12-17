import pytest
from fastapi.testclient import TestClient

from app import app, get_settings
from starter.starter.config import Settings


def get_settings_override():
    return Settings(_env_file=".test.env")


class TestAPI:
    @pytest.fixture(scope="class")
    def client(self):
        app.dependency_overrides[get_settings] = get_settings_override
        return TestClient(app)

    @pytest.fixture(scope="class")
    def case_class_under(self):
        return {
              "age": 42,
              "workclass": "State-gov",
              "education": "Bachelors",
              "marital_status": "Never-married",
              "occupation": "Adm-clerical",
              "relationship": "Not-in-family",
              "race": "White",
              "sex": "Male",
              "capital_gain": 2174,
              "capital_loss": 0,
              "hours_per_week": 40,
              "native_country": "United-States"
            }

    @pytest.fixture(scope="class")
    def case_class_higher(self):
        return {
              "age": 42,
              "workclass": "State-gov",
              "education": "Bachelors",
              "marital_status": "Never-married",
              "occupation": "Adm-clerical",
              "relationship": "Not-in-family",
              "race": "White",
              "sex": "Male",
              "capital_gain": 100000,
              "capital_loss": 0,
              "hours_per_week": 40,
              "native_country": "United-States"
            }

    def test_get_path(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert r.json() == {
            "greeting": "Welcome to the model's API",
            "environment": "testing"
    }

    def test_lower_class(self, client, case_class_under):
        r = client.post(f"/inference/", json=case_class_under)
        assert r.status_code == 200
        assert r.json()["prediction"]["salary_class_id"] == 0
        assert r.json()["prediction"]["salary_class"] == "<=50K"
        assert r.json()["environment"] == "testing"

    def test_higher_class(self, client, case_class_higher):
        r = client.post(f"/inference/", json=case_class_higher)
        assert r.status_code == 200
        assert r.json()["prediction"]["salary_class_id"] == 1
        assert r.json()["prediction"]["salary_class"] == ">50K"
        assert r.json()["environment"] == "testing"
