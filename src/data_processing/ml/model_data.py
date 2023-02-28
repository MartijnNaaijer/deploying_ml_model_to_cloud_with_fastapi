import os
import pickle

import numpy as np
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    rf_model = RandomForestClassifier(max_depth=4, random_state=7)
    rf_model.fit(X_train, y_train)
    return rf_model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def convert_output_to_binary_array(y_test):
    """"""
    return np.array([0 if value == '<=50K' else 1 for value in y_test])


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : sklearn.ensemble.RandomForestClassifier
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds


def save_model(model, folder, filename):
    """Save the trained model as pickled file.
    Inputs
    -------
    model : sklearn.ensemble.RandomForestClassifier
        Trained ml model.
    folder : str
        Directory where model is saved.
    filename: str
        Name of model file.
    """
    pickle.dump(model, 
        open(os.path.join(folder, filename), "wb"))


def load_model(folder, filename):
    """Load the pickled model.
    Inputs
    --------
    folder : str
        Directory where model is saved.
    filename : str
        Name of the model.
    Returns
    --------
        model
    """
    model = pickle.load(open(os.path.join(folder, 
                        filename), 'rb'))
    return model


def make_inference_from_api(input_json, folder, filename):
    """Makes a prediction with a trained and saved model
       with data input from the api.
    Inputs
    """
    loaded_model = load_model(folder, filename)
    pass


def evaluate_on_slices(test, test_predictions, y_test_array, column_name):
    """ Evaluates on slices of a categorical feature of the test set.

    Inputs:
    test: pd.DataFrame  Test set after split from train set, but without further preprocessing.
    test_predictions: np_array  Predictions on test set.
    y_test_array: np.array  Processed dependent variable.
    column_name: str  Categorical column from which slices are evaluated.
    Outputs:
    value: str  Value on which evaluation takes place.
    slice_length: int  Number of observations of specific value in test set. 
    precision: float  Evaluation metric.
    recall: float  Evaluation metric.
    fbeta: float  Evaluation metric.
    """
    categorical_values = set(test[column_name])
    for value in categorical_values:
        slice_indices = test[column_name] == value
        predictions_slice = test_predictions[slice_indices]
        y_test_slice = y_test_array[slice_indices]
        precision, recall, fbeta = compute_model_metrics(y_test_slice, predictions_slice)
        slice_length = len(y_test_slice)
        yield value, slice_length, precision, recall, fbeta
