CATEGORICAL_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

REMOVED_COLUMNS = [
    'fnlgt',  # unknown meaning, skipped
    "education-num"  # maps 1:1 to education, duplicated
]

MODEL_SAVE_PATH = "starter/model/model.pkl"
PIPELINE_SAVE_PATH = "starter/model/preprocessing_pipeline.pkl"
LABEL_ENCODER_SAVE_PATH = "starter/model/label_encoder.pkl"