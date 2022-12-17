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
