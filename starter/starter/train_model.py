# Script to train machine learning model.
import pandas as pd
from sklearn.model_selection import train_test_split

from starter.starter.config import CATEGORICAL_FEATURES
from starter.starter.ml.data import process_data
from starter.starter.ml.model import train_model, compute_model_metrics, inference, save_the_model, inference_on_slices, \
    save_the_pipeline, save_the_label_encoder

# Add the necessary imports for the starter code.

# Add code to load in the data.
data = pd.read_csv("starter/data/census.csv")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)


X_train, y_train, encoder, lb = process_data(
    train, categorical_features=CATEGORICAL_FEATURES, label="salary", training=True
)

# Add code to train the model.
model = train_model(X_train, y_train)
train_pred = inference(model, X_train)
train_scores = compute_model_metrics(y_train, train_pred)

print(f"Train scores: {train_scores}")

X_test, y_test, _, _ = process_data(
    test, categorical_features=CATEGORICAL_FEATURES, label="salary", training=False, encoder=encoder, lb=lb
)

print(inference_on_slices(model, X_test, y_test, encoder, "native-country"))

test_pred = inference(model, X_test)
test_scores = compute_model_metrics(y_test, test_pred)
print(f"Train scores: {test_scores}")


save_the_model(model)
save_the_pipeline(encoder)
save_the_label_encoder(lb)

