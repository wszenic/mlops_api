import pandas as pd
from pydantic import BaseModel

from starter.starter.config import CATEGORICAL_FEATURES
from starter.starter.ml.data import process_data
from starter.starter.ml.model import load_the_model, load_the_pipeline, inference, load_the_label_encoder


class ModelInput(BaseModel):
    age: int
    workclass: str
    education: str
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    def get_features_df(self) -> pd.DataFrame:
        df = pd.DataFrame(self.__dict__, index=[0])
        df.columns = [x.replace("_", "-") for x in df.columns]
        return df


class ModelOutput(BaseModel):
    salary_class_id: int
    salary_class: str


def get_predictions_from_model(model_input: ModelInput):
    pipeline = load_the_pipeline()
    model = load_the_model()
    label_encoder = load_the_label_encoder()

    raw = model_input.get_features_df()
    preprocessed, _, _, _ = process_data(raw, CATEGORICAL_FEATURES, training=False, encoder=pipeline)
    pred = inference(model, preprocessed)

    class_name = label_encoder.inverse_transform(pred[0])

    return ModelOutput(**{
        "salary_class_id": pred[0],
        "salary_class": class_name[0]
    })
