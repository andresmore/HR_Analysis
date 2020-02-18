#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 11:22:59 2019

@author: paula
"""

import numpy as np
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

X = np.loadtxt('x')
y = np.loadtxt('y.csv')


X_scale = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scale, y, test_size=0.33, random_state=42, stratify=y) #shuffle default is True
pca_all = PCA(n_components=538).fit(X_train)
X_train = pca_all.transform(X_train)

clf = LogReg(C=0.001, random_state=0, solver='liblinear', class_weight='balanced',verbose=1,max_iter=7000).fit(X_train, y_train)

X_test = pca_all.transform(X_test)
y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

report = classification_report(y_test, y_pred)

print(report)
print(cm)

from sklearn.metrics import precision_recall_curve
from inspect import signature

y_score = clf.decision_function(X_test)
precision, recall, _ = precision_recall_curve(y_test, y_score)

step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
from sklearn.metrics import average_precision_score
average_precision = average_precision_score(y_test, y_score)

plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format( average_precision))
plt.savefig('900-fft.pdf')

from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score, pos_label=1)
roc_auc = metrics.auc(fpr, tpr)

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score, pos_label=1)
roc_auc = metrics.auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.savefig('roc_optimal_model.pdf', dpi=300)
plt.show()