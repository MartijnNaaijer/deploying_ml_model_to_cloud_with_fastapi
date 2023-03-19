import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier

from .preprocess_data import process_data, clean_data


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


def pickle_object(obj, folder, filename):
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
    pickle.dump(obj, open(os.path.join(folder, filename), "wb"))


def load_object(folder, filename):
    """Load the pickled model.
    Inputs
    --------
    folder : str
        Directory where model is saved.
    filename : str
        Name of the model.
    Returns
    --------
        loaded_object can be a model, encoder or label binarizer.
    """
    loaded_object = pickle.load(open(os.path.join(folder, filename), 'rb'))
    return loaded_object


def convert_to_class(prediction):
    """"""
    converted_predictions = np.array(['<=50K' if pred == 0 else '>50K' for pred in prediction])
    return converted_predictions


def make_inference_from_api(input_json, folder, model_name):
    """Makes a prediction with a trained and saved model
       with data input from the api.
    Inputs
    """
    cat_features = [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country",
    ]

    loaded_model = load_object(folder, model_name)
    encoder = load_object(folder, 'encoder.pkl')
    input_df = pd.DataFrame(dict(input_json), index=[0])
    input_df = clean_data(input_df)
    X_new, _, _, _ = process_data(input_df, categorical_features=cat_features, training=False,
                                  encoder=encoder)
    prediction = inference(loaded_model, X_new)
    converted_pred = convert_to_class(prediction)

    return converted_pred[0]


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
