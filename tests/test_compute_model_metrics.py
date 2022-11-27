import pytest

from starter.starter.ml.model import inference


class TestInference:

    @pytest.fixture(scope="class")
    def mock_inference(self, mock_train_model, mock_process_train_data):
        x, _, _, _ = mock_process_train_data
        model = mock_train_model
        return inference(model, x)

    def test_inference_returned(self, mock_inference):
        assert mock_inference is not None

    def test_predictions_in_range(self, mock_inference):
        assert (mock_inference >= 0).all()
        assert (mock_inference <= 1).all()