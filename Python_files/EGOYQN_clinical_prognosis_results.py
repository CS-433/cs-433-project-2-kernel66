###### This file enables you to get the best results from the clinical dataframe for prognosis #######

# Please  note this is to get the output of the model with the best accuracy but for many other plots and thorough explanations, refer to Notebook
# It still takes a little while to run


# We were unable to "ignore" the warnings that will show, please omit them

import pandas as pd
import numpy as np

import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

from sklearn.metrics import f1_score,accuracy_score,roc_curve, auc
from sklearn.model_selection import  GridSearchCV, RepeatedStratifiedKFold

from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif

# Please respect the confidientiality of the data.
train_clin = pd.read_csv('../Data/Clinic_train.csv', encoding = 'ISO-8859-1')
test_clin = pd.read_csv('../Data/Clinic_test.csv', encoding = 'ISO-8859-1')

X_out_train = train_clin.drop(columns = ['outcome'])
X_out_test =test_clin.drop(columns = ['outcome'])
y_out_train = train_clin['outcome']
y_out_test = test_clin['outcome']

print("Imported and splitted data: \n")


####### Model for "best prediction" on outcome

# The best k is hardcoded but can be found in notebook due to high running time
best_k = 53


# We will run only rbf as we know it to be the best and otherwise takes too long to run

parameters = {'anova__k': [best_k] ,'model__kernel':['rbf'], 'model__C': np.logspace(-1, 1, 30), 'model__gamma' : np.logspace(-4, 0, 30)}
svc = svm.SVC(random_state=123)

fs = SelectKBest(score_func=f_classif, k= best_k)

pipeline = Pipeline(steps=[('anova',fs), ('model', svc)])

## Find scores and best parameters
fit_best = fs.fit(X_out_train,y_out_train)
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=123)

print("Entering GridSearchCV \n")
clf = GridSearchCV(pipeline, parameters, scoring = 'roc_auc', n_jobs=-1, cv = cv)


print("Fit \n")
clf.fit(X_out_train, y_out_train)

print("svm")
svc = svm.SVC(
    random_state=123,
    C= clf.best_params_['model__C'],
    gamma = clf.best_params_['model__gamma'],
    kernel = clf.best_params_['model__kernel'])


print("Fitting \n")
X_train_sel_ftest = fs.fit_transform(X_out_train, y_out_train)
X_test_sel_ftest = fs.transform(X_out_test)

### Fit
svc.fit(X_train_sel_ftest , y_out_train)

print("Predicting on training set \n")
## predict on training set
prediction_train = svc.predict(X_train_sel_ftest)

# Compute metrics for the train set
accuracy_train = accuracy_score(y_out_train, prediction_train)

# False Positive Rate, True Positive Rate, Threshold
fpr_train, tpr_train, thresholds_train = roc_curve(y_out_train, prediction_train)
auc_train = auc(fpr_train, tpr_train)

f1_score_train = f1_score(y_out_train, prediction_train)
### predict on test set
print("Predicting on test set \n")
prediction_test = svc.predict(X_test_sel_ftest)

accuracy_test = accuracy_score(y_out_test, prediction_test)

fpr_test, tpr_test, thresholds_test = roc_curve(y_out_test, prediction_test)
auc_test = auc(fpr_test, tpr_test)

f1_score_test = f1_score(y_out_test, prediction_test)

print("On training we get an Accuracy {}, an AUC {} and F1 score {} ".format(accuracy_train, auc_train, f1_score_train))

print("For test we get an Accuracy {}, an AUC {} and F1 score {}".format(accuracy_test, auc_test, f1_score_test))


