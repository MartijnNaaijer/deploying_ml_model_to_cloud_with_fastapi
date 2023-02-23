import os
import pickle

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
    _______
    folder : str
        Directory where model is saved.
    filename : str
        Name of the model.
    Returns
        model
    """
    model = pickle.load(open(os.path.join(folder, 
                        filename), 'rb'))
    return model

def make_inference_from_api(input_json):
    pass
