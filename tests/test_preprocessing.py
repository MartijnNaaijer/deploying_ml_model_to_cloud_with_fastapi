import os
import numpy as np
import pandas as pd
import pytest

from ..src.data_processing.ml import preprocess_data as ppd
from .config import ROOT, DATA_FOLDER, DATA_FILE

 
@pytest.fixture(scope="module")
def input_df():
    df = pd.read_csv(os.path.join('../data', DATA_FILE), sep=',')
    return df


@pytest.fixture(scope="module")
def processed_data(input_df):
    cat_features=["workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country"]
    return ppd.process_data(input_df, categorical_features=cat_features, label='salary')


def test_shape_data(processed_data):
    X, y, _, _ = processed_data
    assert X.shape[1] > 14
    assert X.shape[0] > 10


def test_data_types(processed_data):
    X, y, _, _ = processed_data
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)


def test_y_values(processed_data):
    _, y, _, _ = processed_data
    assert set(y) == {0, 1}
