# Script to train machine learning model.
import pandas as pd
from sklearn.model_selection import train_test_split

from starter.starter.ml.data import process_data
from starter.starter.ml.model import train_model, compute_model_metrics, inference, save_the_model

# Add the necessary imports for the starter code.

# Add code to load in the data.
data = pd.read_csv("starter/data/census.csv")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Add code to train the model.
model = train_model(X_train, y_train)
train_pred = inference(model, X_train)
train_scores = compute_model_metrics(y_train, train_pred)

X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)

test_pred = inference(model, X_test)
test_scores = compute_model_metrics(y_test, test_pred)

save_the_model(model, "./starter/model.pkl")

