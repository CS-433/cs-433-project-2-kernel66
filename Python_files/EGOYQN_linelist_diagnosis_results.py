#### This file is to produce our best result for Ebola prediction in linelist
# Please refer to the Notebook for more plots and explanations

# We were unable to "ignore" the warnings that will show, please omit them
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

import pandas as pd
import numpy as np

from sklearn.metrics import f1_score,accuracy_score,roc_curve, auc
from sklearn.model_selection import  GridSearchCV, RepeatedStratifiedKFold
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif


# Please respect the confidientiality of the data.
train_clin = pd.read_csv('../Data/Linelist_train.csv', encoding = 'ISO-8859-1')
test_clin = pd.read_csv('../Data/Linelist_test.csv', encoding = 'ISO-8859-1')

X_ebo_train = train_clin.drop(columns = ['epistat'])
X_ebo_test =test_clin.drop(columns = ['epistat'])
y_ebo_train = train_clin['epistat']
y_ebo_test = test_clin['epistat']


## Model for best prediction on diagnosis

# Found in Notebook
best_k = 87

# We will run only rbf as we expect it to be the best and otherwise takes too long to run
parameters = {'anova__k': [best_k] ,'model__kernel':['rbf'], 'model__C': np.logspace(-1, 1, 20), 'model__gamma' : np.logspace(-2, 0, 20)}
svc = svm.SVC(random_state=123)


fs = SelectKBest(score_func=f_classif, k= best_k)
pipeline = Pipeline(steps=[('anova',fs), ('model', svc)])

## Find scores and best parameters 
fit_best = fs.fit(X_ebo_train,y_ebo_train)
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=123)

clf = GridSearchCV(pipeline, parameters, scoring = 'roc_auc', n_jobs=-1, cv = cv)

clf.fit(X_ebo_train, y_ebo_train)


svc = svm.SVC(
    random_state=123,
    C= clf.best_params_['model__C'], 
    gamma = clf.best_params_['model__gamma'], 
    kernel = clf.best_params_['model__kernel']
)


### Add the fs.transform and fs.transform_fit
X_train_sel_ftest = fs.fit_transform(X_ebo_train, y_ebo_train)
X_test_sel_ftest = fs.transform(X_ebo_test)

### Fit
svc.fit(X_train_sel_ftest , y_ebo_train)

print("Predicting on training set \n")
## predict on training set
prediction_train = svc.predict(X_train_sel_ftest)

# Compute metrics for the train set
accuracy_train = accuracy_score(y_ebo_train, prediction_train)

# False Positive Rate, True Positive Rate, Threshold
fpr_train, tpr_train, thresholds_train = roc_curve(y_ebo_train, prediction_train)
auc_train = auc(fpr_train, tpr_train)

f1_score_train = f1_score(y_ebo_train, prediction_train)
### predict on test set
print("Predicting on test set \n")
prediction_test = svc.predict(X_test_sel_ftest)

accuracy_test = accuracy_score(y_ebo_test, prediction_test)

fpr_test, tpr_test, thresholds_test = roc_curve(y_ebo_test, prediction_test)
auc_test = auc(fpr_test, tpr_test)

f1_score_test = f1_score(y_ebo_test, prediction_test)

print("On training we get an Accuracy {}, an AUC {} and F1 score {} ".format(accuracy_train, auc_train, f1_score_train))

print("For test we get an Accuracy {}, an AUC {} and F1 score {}".format(accuracy_test, auc_test, f1_score_test))

