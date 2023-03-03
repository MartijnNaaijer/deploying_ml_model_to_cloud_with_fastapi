"""
This file contains some tests for the census dataset without preprocessing.
"""
import os

import numpy as np
import pandas as pd
import pytest

from .config import ROOT, DATA_FOLDER, DATA_FILE


@pytest.fixture(scope="module")
def input_df():
    df = pd.read_csv(os.path.join('../data', DATA_FILE), sep=',')
    return df


def test_columns_data(input_df):
    assert input_df.shape[1] == 15


def test_rows_data(input_df):
    assert input_df.shape[0] > 30000


def test_values_in_dependent_variable(input_df):
    assert set(input_df['salary']) == {'<=50K', '>50K'}


def test_mean_age(input_df):
    assert 20 < (np.mean(input_df['age'])) < 60


def test_education_values(input_df):
    assert np.min(input_df['education-num']) == 1
    assert np.max(input_df['education-num']) == 16
    