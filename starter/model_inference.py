import pandas as pd
from pydantic import BaseModel

from starter.starter.config import Settings
from starter.starter.ml.data import process_data
from starter.starter.ml.model import load_the_model, load_the_pipeline, inference, load_the_label_encoder
from starter.starter.modelling_config import CATEGORICAL_FEATURES


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

    class Config:
        schema_extra = {
            "example":
                {
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
        }


class ModelOutput(BaseModel):
    salary_class_id: int
    salary_class: str


def get_predictions_from_model(model_input: ModelInput, settings: Settings):
    pipeline = load_the_pipeline(settings)
    model = load_the_model(settings)
    label_encoder = load_the_label_encoder(settings)

    raw = model_input.get_features_df()
    preprocessed, _, _, _ = process_data(raw, CATEGORICAL_FEATURES, training=False, encoder=pipeline)
    pred = inference(model, preprocessed)

    class_name = label_encoder.inverse_transform(pred[0])

    return ModelOutput(**{
        "salary_class_id": pred[0],
        "salary_class": class_name[0]
    })
