import os
import numpy as np
import pandas as pd
import pytest

from config import ROOT, DATA_FOLDER, DATA_FILE
from ..src/data_processing import model_data

 
@pytest.fixture(scope="module")
def input_df():
    df = pd.read_csv(os.path.join(ROOT, DATA_FOLDER, DATA_FILE), sep=',')
    return df