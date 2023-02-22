# Script to train machine learning model.
import os
import pandas as pd
from sklearn.model_selection import train_test_split

import ml.preprocess_data as ppd
import ml.model_data as md

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = '../../data'
DATA_FILE = 'census.csv'

# Add code to load in the data.
data = pd.read_csv(os.path.join(DATA_PATH, DATA_FILE), sep=',')

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
X_test, y_test, _, _ = ppd.process_data(
    test, categorical_features=cat_features, label="salary", training=False,
    encoder=encoder
)

# Train and save a model.
model = md.train_model(X_train, y_train)
md.save_model(model, os.path.join(ROOT, 'model'), 'trained_model.pkl')
