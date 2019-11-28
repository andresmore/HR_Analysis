#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 16:58:52 2019

@author: paula
"""

import pandas as pd
import numpy as np
import os 

from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import feature_selection
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_curve, auc, precision_recall_curve,confusion_matrix,classification_report
import matplotlib.pyplot as plt
from inspect import signature
from sklearn.metrics import average_precision_score
import skopt
from skopt.space.space import Real 
from skopt.callbacks import DeltaYStopper
from sklearn.metrics import make_scorer,fbeta_score

def get_indexes_frequencies(freq_delta):
    indexes = []
    for i in range(1,int(X.shape[1]/freq_delta)):
        indexes.append(freq_delta*i)
    return indexes

def plot_precision_recall_curve(recall, precision, avg_precision, file):
    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    plt.figure()
    lw = 1
    plt.plot(recall, precision, color='k',marker = '.', lw=lw, label='Precision-recall curve (AP = %0.2f)' % avg_precision)
    #plt.fill_between(recall, precision, alpha=0.2, color='lightgray')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.legend(loc="upper right")
    plt.title('Precision-Recall Curve')
    plt.savefig(file, dpi=300, bbox_inches='tight')
    plt.show()
    
def plot_roc_curve(fpr, tpr, roc_auc_score, file):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='k',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_score)
    plt.plot([0, 1], [0, 1], color='grey', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic curve')
    plt.legend(loc="lower right")
    plt.savefig(file, dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_trained_model(model, X_test, y_test, name):
    y_score = model.decision_function(X_test)
    y_pred = model.predict(X_test)

    average_precision = average_precision_score(y_test, y_score)
    
    precision, recall, other = precision_recall_curve(y_test, y_score)
    fpr, tpr, other = roc_curve(y_test, y_score)
    roc_auc_score = auc(fpr, tpr)
    
    filenname = name + "prcurve.pdf"
    plot_precision_recall_curve(precision, recall, average_precision, filenname)
    filenname = name + "roccurve.pdf"
    plot_roc_curve(fpr, tpr, roc_auc_score, filenname)
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(conf_matrix)
    return y_score
#Read fft data
X = np.loadtxt('../x')
y = np.loadtxt('../y.csv')

#separate, use the same settings for all tests
indices = range(len(X))
X_train, X_test, y_train, y_test,indices_train,indices_test = train_test_split(X, y,indices, test_size=0.33, random_state=42, stratify=y)

#First: analyze directly without PCA using different frequencies only
title = 'Frequencies'
print(title)
freqs = [900, 450] #corresponds to frequencies 0.5, 1, 1.5 , etc

for freq_delta in freqs:
    print(freq_delta)
    indexes = get_indexes_frequencies(freq_delta)
    X_new   = X[:, indexes]
    X_train = X_new[indices_train,:]
    X_test  = X_new[indices_test,:]
    
    pipeline = Pipeline(steps=[('scale', StandardScaler()),
                    ('logreg', LogReg(random_state=0, class_weight='balanced',solver='lbfgs',max_iter=10000))])
    
    pipeline.fit(X_train, y_train)
    evaluate_trained_model(pipeline, X_test, y_test, "pictures/freq_"+str(freq_delta)+"_")

#Analyze using PCA
title = 'PCA-90%'
print(title)
X_train = X[indices_train,:]
X_test  = X[indices_test,:]

#select components pca 
X_scale = StandardScaler().fit_transform(X)
pca_all = PCA().fit(X_scale)
explained_var_cumsum = np.cumsum(pca_all.explained_variance_ratio_)
x_axis = range(len(explained_var_cumsum))
n_comp = np.where(explained_var_cumsum >0.9)[0][0]
print(n_comp)
pipeline = Pipeline(steps=[('scale', StandardScaler()),
                ('pca', PCA(n_components=n_comp)), #based on previous result on 0.85 explained variance)
                ('logreg', LogReg(random_state=0,class_weight='balanced',solver='lbfgs',max_iter=10000))])

pipeline.fit(X_train, y_train)
y_score = evaluate_trained_model(pipeline, X_test, y_test, "pictures/pca90")

title = 'PCA - 85%'
print(title)
n_comp = np.where(explained_var_cumsum >0.85)[0][0]
pipeline = Pipeline(steps=[('scale', StandardScaler()),
                ('pca', PCA(n_components=n_comp)), #based on previous result on 0.85 explained variance)
                ('logreg', LogReg(random_state=0, class_weight='balanced', solver='lbfgs',max_iter=10000))])

pipeline.fit(X_train, y_train)
evaluate_trained_model(pipeline, X_test, y_test, "pictures/pca85")


#Optimize parameters
title = 'Optimize'
print(title)

pipe = Pipeline(steps=[('scale', StandardScaler()),
        ('pca', PCA(n_components=256)),
        ('logreg',LogReg(C=1, random_state=0, solver='liblinear', class_weight='balanced',verbose=0,max_iter=70))])

parameters = {'logreg__C': Real(0.0001, 1000, 'log-uniform')}
scorer = make_scorer(fbeta_score, beta=2, pos_label=1)
optimizer = skopt.BayesSearchCV(
        pipe,
        parameters,
        cv=5,
        n_iter=20,
        scoring=scorer,
        n_jobs=1,
        return_train_score=True
    )
deltaY = DeltaYStopper(delta=0.0005, n_best=8)
optimizer.fit(X_train, y_train, callback=deltaY)
print(f'Best params for model : {optimizer.best_params_} , re fitting with complete train set')
best_model = pipe.set_params(**optimizer.best_params_)
best_model.fit(X_train, y_train)

#optimizer.best_params_
#Out[9]: {'logreg__C': 0.0001}
print("The model is trained on the train  set.")
print("The scores are computed on the test set.")
print()
evaluate_trained_model(best_model, X_test, y_test, "pictures/pcaoptimize")
