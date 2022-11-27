import pytest

from starter.starter.ml.model import train_model


class TestModelTraining():

    @pytest.fixture(scope="class")
    def mock_train_model(self, mock_process_train_data, mock_train_configuration):
        x, y, _, _ = mock_process_train_data

        return train_model(x, y)

    def test_model_returned(self, mock_train_model):
        assert mock_train_model is not None

    def test_model_is_trained(self, mock_train_model):
        """
        Untrained XGBoost models dont have classes attribute,
        method should be replaced if the model type is changed
        """
        assert hasattr(mock_train_model, "classes_")
