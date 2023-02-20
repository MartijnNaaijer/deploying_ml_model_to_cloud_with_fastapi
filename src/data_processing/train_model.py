# Script to train machine learning model.
import os
import pandas as pd
from sklearn.model_selection import train_test_split

import ml.preprocess_data as ppd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FOLDER = 'data'
DATA_FILE = 'census.csv'

# Add code to load in the data.
data = pd.read_csv(os.path.join(ROOT, DATA_FOLDER, DATA_FILE), sep=',')

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
X_train, y_train, encoder, lb = ppd.process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.

# Train and save a model.
