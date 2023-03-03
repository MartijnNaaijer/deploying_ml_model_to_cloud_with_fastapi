import os
import numpy as np
import pandas as pd
import pytest
import sklearn

from .config import ROOT, DATA_FOLDER, DATA_FILE
from ..src.data_processing.ml import model_data as md

 
def obj_plugin():
    """Helper function for tests"""
    return None


def pytest_configure():
    """Helper function for tests"""
    pytest.rf_model = obj_plugin()


@pytest.fixture(scope="module")
def input_df():
    df = pd.read_csv(os.path.join('../data', DATA_FILE), sep=',')
    return df


def test_model_type(input_df):
    train_data = input_df.loc[:1000]
    train_data = train_data[['age', 'salary']]
    y = train_data.salary
    X = train_data.drop(['salary'], axis=1)
    model = md.train_model(X.values, y.values)
    assert isinstance(model, sklearn.ensemble.RandomForestClassifier)

    pytest.rf_model = model
    

def test_inference_output_type():
    rf_model = pytest.rf_model
    prediction = md.inference(rf_model, np.array([25, 50]).reshape(-1, 1))
    assert isinstance(prediction, np.ndarray) 
    assert len(prediction) == 2
