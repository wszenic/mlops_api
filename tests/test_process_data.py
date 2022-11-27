import pytest
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

from starter.starter.ml.data import process_data


class TestDataProcessor:
    @pytest.fixture(scope="class")
    def test_process_data(self, mock_test_train_data, mock_train_configuration):
        train, _ = mock_test_train_data

        return process_data(train, categorical_features=mock_train_configuration["categorical_features"],
                     label=mock_train_configuration["label"], training=True)


    def test_same_number_of_rows(self, mock_test_train_data, test_process_data):
        train, _ = mock_test_train_data
        X_train, _, _, _ = test_process_data

        assert train.shape[0] == X_train.shape[0]

    def test_label_binarizer_returned(self, test_process_data):
        _, _, _, lb = test_process_data

        assert lb is not None
        assert type(lb) == type(LabelBinarizer())

    def test_encoder_returned(self, test_process_data):
        _, _, encoder, _ = test_process_data

        assert encoder is not None
        assert type(encoder) == type(OneHotEncoder())