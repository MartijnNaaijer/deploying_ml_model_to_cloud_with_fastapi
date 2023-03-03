# Script to train machine learning model.
import logging
import os
import pandas as pd
from sklearn.model_selection import train_test_split

import ml.preprocess_data as ppd
import ml.model_data as md


logging.basicConfig(
    filename='../logs/logs_model_performance.log',
    level = logging.INFO,
    filemode='w')


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = '../../data'
DATA_FILE = 'census.csv'

# Add code to load in the data.
data = pd.read_csv(os.path.join(DATA_PATH, DATA_FILE), sep=',')

# Do some basic cleaning: remove redundant colsumns, and harmonize values.
data = ppd.clean_data(data)

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
    train, categorical_features=cat_features, label='salary', training=True
)

# Proces the test data with the process_data function.
X_test, y_test, _, _ = ppd.process_data(
    test, categorical_features=cat_features, label='salary', training=False,
    encoder=encoder
)

# Train and save a model, encoder and lb.
model = md.train_model(X_train, y_train)
md.pickle_object(model, os.path.join(ROOT_DIR, 'model'), 'trained_model.pkl')
md.pickle_object(encoder, os.path.join(ROOT_DIR, 'model'), 'encoder.pkl')
md.pickle_object(lb, os.path.join(ROOT_DIR, 'model'), 'lb.pkl')


## Validate model on testset and slices
test_predictions = md.inference(model, X_test)
y_test_array = md.convert_output_to_binary_array(y_test)

precision, recall, fbeta = md.compute_model_metrics(y_test_array, test_predictions)
logging.info('Overall model performance')
logging.info(f'precision {precision}, recall {recall}, fbeta {fbeta}')
logging.info('')


slice_eval_features = ['sex', 'race']
for feature in slice_eval_features:
    logging.info(f'Performance on slices of feature {feature}')
    for value, obs_count, precision, recall, fbeta in md.evaluate_on_slices(test, 
                                                                            test_predictions, 
                                                                            y_test_array, 
                                                                            feature):
        logging.info(f'Value {value} has {obs_count} observations.')
        logging.info(f'precision {precision}, recall {recall}, fbeta {fbeta}')
    logging.info('')
