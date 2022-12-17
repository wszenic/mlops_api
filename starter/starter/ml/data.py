import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, OneHotEncoder
from xgboost import XGBClassifier

from starter.starter.modelling_config import REMOVED_COLUMNS


def process_data(
    X, categorical_features=[], label=None, training=True, encoder=None, lb=None
):
    """ Process the data used in the machine learning pipeline.

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Note: depending on the type of model used, you may want to add in functionality that
    scales the continuous data.

    Inputs
    ------
    X : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    training : bool
        Indicator if training mode or inference/validation mode.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.

    Returns
    -------
    X : np.array
        Processed data.
    y : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns the encoder passed
        in.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns the binarizer
        passed in.
    """

    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])

    # dropping obsolete columns with unknown properties
    X.drop(REMOVED_COLUMNS, axis=1, errors='ignore', inplace=True)

    X_categorical = X[categorical_features].values
    X_continuous = X.drop(*[categorical_features], axis=1).reset_index(drop=True)

    if training is True:
        encoder = make_pipeline(
            SimpleImputer(strategy='constant', missing_values='?', fill_value='np.nan'),
            OneHotEncoder(sparse=False, handle_unknown="ignore")
        )

        lb = LabelBinarizer()
        X_categorical = encoder.fit_transform(X_categorical)
        y = lb.fit_transform(y.values).ravel()
    else:
        X_categorical = encoder.transform(X_categorical)
        try:
            y = lb.transform(y.values).ravel()
        # Catch the case where y is None because we're doing inference.
        except AttributeError:
            pass

    df_categorical = pd.DataFrame(
        data=X_categorical, columns=encoder.get_feature_names_out()
    )

    X = pd.concat([X_continuous, df_categorical], axis=1)
    X.columns = [*X_continuous.columns, *df_categorical.columns]
    return X, y, encoder, lb
