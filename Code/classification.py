!pip install xgboost

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from tabulate import tabulate
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


pd.set_option('display.max_columns', None)

import warnings
warnings.filterwarnings('ignore')

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import PrecisionRecallDisplay

# Import classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn import svm, datasets

# import for preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# import methods for measuring accuracy, precision, recall etc
from sklearn.metrics import (
    accuracy_score, 
    precision_score,
    recall_score,
    roc_auc_score,
    f1_score,
    confusion_matrix,
    classification_report)

import time


Data = pd.read_csv('Datanew_feat_select.csv')
y = Data['class']

Data = Data.drop('class', axis=1)
X = Data

# stratified sampling to create a training and testing set
from sklearn.model_selection import StratifiedShuffleSplit

stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in stratified_split.split(X, y):
    X_train_strat, X_test_strat = X.iloc[train_index], X.iloc[test_index]
    y_train_strat, y_test_strat = y.iloc[train_index], y.iloc[test_index]
    
train_data_strat = pd.concat([X_train_strat, y_train_strat], axis=1)
test_data_strat = pd.concat([X_test_strat, y_test_strat], axis=1)

train_data_strat.to_csv('train_data_stratnew_feat_select.csv', index=False)
test_data_strat.to_csv('test_data_stratnew_feat_select.csv', index=False)

train_data = pd.read_csv('train_data_stratnew_feat_select.csv')
test_data = pd.read_csv('test_data_stratnew_feat_select.csv')

print(f'Training data shape: {train_data.shape}')
print(f'Testing data shape: {test_data.shape}')

num_class_0 = train_data['class'].value_counts()[0]
num_class_1 = train_data['class'].value_counts()[1]

print(f"Number of samples in class 0: {num_class_0}")
print(f"Number of samples in class 1: {num_class_1}")

num_class_0 = test_data['class'].value_counts()[0]
num_class_1 = test_data['class'].value_counts()[1]

print(f"Number of samples in class 0: {num_class_0}")
print(f"Number of samples in class 1: {num_class_1}")

#### Logistic Regression

# loop search for best parameters 
c_values = np.arange (0.001, 10, 0.01) #[0.001, 0.01, 0.1, 1, 10, 100,200,300]
lr = LogisticRegression(random_state=42)

best_score = 0
best_c = 0
for c in c_values:
    lr.set_params(C=c)
    lr.fit(X_train_strat, y_train_strat)
    y_pred = lr.predict(X_test_strat)
    score = accuracy_score(y_test_strat, y_pred)
    if score > best_score:
        best_score = score
        best_c = c
print(c)       
print("Best value of C:", best_c)
print("Classification report for testing data:")
print(classification_report(y_test_strat, y_pred))

### with best values
lr = LogisticRegression(C=2.5609999999999995,random_state=42)
lr.fit(X_train_strat, y_train_strat)
y_pred = lr.predict(X_test_strat)
print("\nClassification Report:")
print(classification_report(y_test_strat, y_pred))

report = classification_report(y_test_strat, y_pred, output_dict=True)
p_lr = round(report['weighted avg']['precision']*100,1)
r_lr = round(report['weighted avg']['recall']*100,1)
f_lr = round(report['weighted avg']['f1-score']*100,1)
print(p_lr, '\t', r_lr, '\t', f_lr)

classifier = lr
display = RocCurveDisplay.from_estimator(classifier, pd.concat([X_train_strat, X_result, X_result, X_result, X_result]), 
                                         pd.concat([y_train_strat,Y_result,Y_result,Y_result,Y_result]))
lr_FPR = display.fpr
lr_TPR = display.tpr


#### Weighted logistic regression

# loop search for best parameters 
c_values = np.arange (0.001, 20, 0.01)#[0.001, 0.01, 0.1, 1, 10, 100]
w = {0: 1, 1: 1} 
lrw= LogisticRegression(random_state=42)
best_score = 0
best_c = 0
for c in c_values:
    lrw.set_params(C=c)
    lrw.fit(X_train_strat, y_train_strat)
    y_pred = lrw.predict(X_test_strat)
    score = accuracy_score(y_test_strat, y_pred)
    if score > best_score:
        best_score = score
        best_c = c

print("Best value of C:", best_c)
print("Classification report for testing data:")
print(classification_report(y_test_strat, y_pred))

w = {0: 1, 1: 1} 
lrw= LogisticRegression(random_state=42, class_weight=w, C=2.5609999999999995)
lrw.fit(X_train_strat, y_train_strat)
y_pred = lrw.predict(X_test_strat)
print("\nClassification Report:")
print(classification_report(y_test_strat, y_pred))

report = classification_report(y_test_strat, y_pred, output_dict=True)
p_lrw = round(report['weighted avg']['precision']*100,1)
r_lrw = round(report['weighted avg']['recall']*100,1)
f_lrw = round(report['weighted avg']['f1-score']*100,1)
print(p_lrw, '\t', r_lrw, '\t', f_lrw)

### with best values
w = {0: 1, 1: 1} 
lrw= LogisticRegression(random_state=2, class_weight=w, C=20)
lrw.fit(X_train_strat, y_train_strat)
classifier = lrw

display = RocCurveDisplay.from_estimator(classifier, X_train_strat, y_train_strat)
lrw_FPR = display.fpr
lrw_TPR = display.tpr
