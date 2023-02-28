# Model Card
For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Random Forest Classification model, trained with sklearn. The model predicts whether ones salary is higher or lower than 50 K (output variable "salary" has values ">50K" and "<=50K"). The input variables are a mix of numerical and categorical features.

## Intended Use
The model can be used by loading the pickled model in the directory src/model, or by making predictions using the app deployed on Heroku.

## Training Data
The dataset is the [Census Income Data Set](https://archive.ics.uci.edu/ml/datasets/census+income).
The training set consists of 80% of the total dataset by randomly selecting rows from it.

## Evaluation Data
The evaluation dataset consists of 20% of the original dataset. It is the part of it not included in the training set.

## Metrics
In the folder src/logs, the evaluation on the test set is logged. On the complete test set, the result is:
precision 0.8160583941605839, recall 0.3618122977346278, fbeta 0.5013452914798205

The file also contains evaluations on slices of the features "sex" and "race". The results are:

Performance on slices of feature sex
Value  Female has 2160 observations.
precision 1.0, recall 0.09813084112149532, fbeta 0.17872340425531913
Value  Male has 4353 observations.
precision 0.8102409638554217, recall 0.4042073628850488, fbeta 0.5393483709273182

Performance on slices of feature race
Value  Black has 596 observations.
precision 0.8333333333333334, recall 0.24691358024691357, fbeta 0.380952380952381
Value  White has 5627 observations.
precision 0.8225039619651348, recall 0.3678242381289865, fbeta 0.5083251714005876
Value  Other has 51 observations.
precision 0.3333333333333333, recall 0.3333333333333333, fbeta 0.3333333333333333
Value  Amer-Indian-Eskimo has 55 observations.
precision 0.5, recall 0.2, fbeta 0.28571428571428575
Value  Asian-Pac-Islander has 184 observations.
precision 0.72, recall 0.4, fbeta 0.5142857142857142

## Ethical Considerations
There are no specific ethical considerations. The data are openly available.

## Caveats and Recommendations
The model is trained on US data. The situation in the US may differ from the European situation, so the model will probably not perform on new European data as on the test set.
Note that the features that are evaluated in the log file are somewhat imbalanced. The results for the values with many observations may be more reliable than those with fewer observations..
