import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from starter.starter.ml.data import process_data
from starter.starter.ml.model import train_model


@pytest.fixture(scope="session")
def mock_train_configuration():
    return {
        "categorical_features":
            [
                "workclass",
                "education",
                "marital-status",
                "occupation",
                "relationship",
                "race",
                "sex",
                "native-country",
            ],
        "label": "salary"
    }


@pytest.fixture(scope="session")
def mock_test_data():
    """ Mock data, a small subset of the original data. """
    return pd.read_csv("./test_data/test_set.csv")


@pytest.fixture(scope="session")
def mock_test_train_data(mock_test_data):
    """ split to train and test data """
    return train_test_split(mock_test_data, test_size=0.2)


@pytest.fixture(scope="session")
def mock_process_train_data(mock_test_train_data, mock_train_configuration):
    """ Mock data, a small subset of the original data. """
    train, test = mock_test_train_data
    return process_data(train, categorical_features=mock_train_configuration["categorical_features"],
                        label=mock_train_configuration["label"], training=True)


@pytest.fixture(scope="session")
def mock_train_model(mock_process_train_data):
    """Train the model."""
    x, y, _, _ = mock_process_train_data

    model = train_model(x, y)
    return model
