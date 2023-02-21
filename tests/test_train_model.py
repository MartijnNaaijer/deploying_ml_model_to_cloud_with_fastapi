import os
import numpy as np
import pandas as pd
import pytest

from ..config import ROOT, DATA_FOLDER, DATA_FILE

#from ..src.data_processing.ml import preprocess_data as ppd

#ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#DATA_FOLDER = 'data'
#DATA_FILE = 'census.csv'

 
@pytest.fixture(scope="module")
def input_df():
    df = pd.read_csv(os.path.join(ROOT, DATA_FOLDER, DATA_FILE), sep=',')
    return df