from collections import namedtuple

import joblib
import pandas as pd
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from starter.starter.config import Settings
from starter.starter.modelling_config import CATEGORICAL_FEATURES

model_metrics = namedtuple("model_metrics", ["precision", "recall", "fbeta"])

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train) -> float:
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    model = XGBClassifier()
    model.fit(X_train, y_train)

    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return model_metrics(precision, recall, fbeta)


def inference_on_slices(model, X, Y, encoder, column_to_split_by):
    """ Run model inferences on slices of the data and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    column_to_split_by : str
        Column name to split the data by.
    Returns
    -------
    preds : pd.DataFrame
        Predictions from the model.
    """
    preds = {}

    id_in_pipeline = CATEGORICAL_FEATURES.index(column_to_split_by)
    for column_level_name in encoder[1].categories_[id_in_pipeline]:
        column_name_with_prefix = f"x{id_in_pipeline}_{column_level_name}"
        filtering_mask = X[column_name_with_prefix] == 1
        if all(~filtering_mask):
            continue

        x_slice = X[filtering_mask]
        y_slice = Y[filtering_mask]

        slice_preds = model.predict(x_slice)

        preds[column_level_name] = compute_model_metrics(y_slice, slice_preds)

    output = pd.DataFrame(preds).T
    output.reset_index(drop=False, inplace=True)
    output.columns = [column_to_split_by, "precision", "recall", "fbeta"]
    return output


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds


def save_the_model(model, settings: Settings):
    """ Save the model to a file.

    Inputs
    ------
    model :
        Trained machine learning model.
    settings :
        instance of settings class, containing paths
    """
    joblib.dump(model, settings.model_save_path)


def load_the_model(settings):
    """ Read the model from a file
    Returns
    -------
    settings :
        instance of settings class, containing paths
    """
    return joblib.load(settings.model_save_path)


def save_the_pipeline(pipeline, settings: Settings):
    """ Save the pipeline to a file.

    Inputs
    ------
    model :
        Preprocessing pipeline
    settings :
        instance of settings class, containing paths
    """
    joblib.dump(pipeline, settings.pipeline_save_path)


def load_the_pipeline(settings: Settings):
    """ Read the model from a file
    Returns
    -------
    settings :
        instance of settings class, containing paths
    """
    return joblib.load(settings.pipeline_save_path)


def save_the_label_encoder(label_encoder, settings: Settings):
    """ Save the pipeline to a file.

    Inputs
    ------
    model :
        Preprocessing pipeline
    settings :
        instance of settings class, containing paths
    """
    joblib.dump(label_encoder, settings.label_encoder_save_path)


def load_the_label_encoder(settings: Settings):
    """ Read the model from a file
    Returns
    -------
    settings :
        instance of settings class, containing paths
    """
    return joblib.load(settings.label_encoder_save_path)
