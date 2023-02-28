# Model Card
For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Random Forest Classification model, trained with sklearn. The model predicts whether ones salary is higher or lower than 50 K (output variable "salary" has values ">50K" and "<=50K"). The input variables are a mix of numerical and categorical features.

## Intended Use
The model can be used by loading the pickled model in the directory src/model, or by making predictions using the REST API Heroku.

## Training Data
The dataset is the [Census Income Data Set](https://archive.ics.uci.edu/ml/datasets/census+income).
The training set consists of 80% of the total dataset by randomly selecting rows from it.

## Evaluation Data
The evaluation dataset consists of 20% of the original dataset. It is the part of it not included in the training set.

## Metrics
_Please include the metrics used and your model's performance on those metrics._

## Ethical Considerations
There are no specific ethical considerations. The data are openly available.

## Caveats and Recommendations
The model is trained on US data. The situation in the US may differ from the European situation, so the model will probably not perform on new European data as on the test set. 
