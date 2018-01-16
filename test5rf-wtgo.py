#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Regression


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## chap 4
from io import StringIO
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from itertools import combinations
import matplotlib.pyplot as plt

## chap 5
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from scipy import interp

# for sklearn 0.18's alternative syntax
from distutils.version import LooseVersion as Version
from sklearn import __version__ as sklearn_version
if Version(sklearn_version) < '0.18':
    from sklearn.grid_search import train_test_split
    from sklearn.cross_validation import StratifiedKFold
    from sklearn.cross_validation import cross_val_score
    from sklearn.learning_curve import learning_curve
    from sklearn.learning_curve import validation_curve
    from sklearn.grid_search import GridSearchCV
else:
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import learning_curve
    from sklearn.model_selection import validation_curve
    from sklearn.model_selection import GridSearchCV

import sys
#for f in range(max_feat): print("%2d) %-*s %f" % (f + 1, 30, indices[f]+1, importances[indices[f]]))
#sys.exit()
#############################################################################
import os
odir="out"
#os.mkdir(odir)
#os.makedirs(odir, exist_ok=True)
nvar = 88
# get 88 variables
if os.path.exists(odir):
      print ('dir found.')
else:
    os.makedirs(odir, exist_ok=True)

#from here.

import csv # CSVファイルを扱うためのモジュールのインポート
f1='innerjoin-Inpute.csv'

mgd=pd.read_csv(f1,  delimiter=',' )
print ('read dataset has %d cases %d variables' % mgd.shape)

y = mgd.ix[:,0]
#X = mgd.ix[:,1:]
idcol=mgd.columns.str.contains("mmGO")
useid= ~ (idcol)
useid[0:1] = [False]
Xt= mgd.ix[:, np.array(useid)]
#print ('Xt col is:', Xt.columns)
X=pd.DataFrame( Xt )

print ('X, find null:' , X.isnull().sum())
print ("now X has %d cases, %d variables" % X.shape)
#sys.exit()

# cut the first 3 columns.
print ("consider %d cases, %d variables" % X.shape)


#X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
# 7割を学習データに、のこりをテストデータ（正しい精度の計算）に
X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.3, random_state=0)

# 規格化をする
from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)


# Random Forest
forest = RandomForestClassifier(n_estimators=100,
                                random_state=0,
                                n_jobs=-1)
forest.fit(X_train, y_train)
print('Training accuracy:', forest.score(X_train_std, y_train))
print('Test accuracy:', forest.score(X_test_std, y_test))


#####
print(50 * '=')
print('Section: Reading a confusion matrix')
print(50 * '-')

from sklearn.metrics import confusion_matrix
y_pred = forest.predict(X_test)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print('Confusion matrix', confmat)

print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred))
print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred))
print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred))



feat_labels = X.columns
print(50 * '-')
print('feat_labels')
for i in range(len(feat_labels)):
    list_item = feat_labels[i]
    print('{0}:{1}'.format(i+1, list_item))


#print (importances)
print(50 * '=')
print('get importance')
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
print (indices)
print (feat_labels[indices])

max_feat = len(feat_labels)
print(50 * '=')
print('Top %d Important features' % max_feat)
for f in range(max_feat):
    print("%2d) %-*s %f" % (f + 1, 30,
                            feat_labels[indices[f]],
                            importances[indices[f]]))


#plt.title('Feature Importances')
max_feat = 20
fig = plt.figure(2)
plt.bar(range(max_feat),
        importances[indices[0:max_feat]],
        color='blue',
        align='center')

plt.xticks(range(max_feat),
            feat_labels[indices[0:max_feat]],
           #indices[0:max_feat]+1,
           #feat_labels[indices[0:max_feat]],
	   rotation=90)
plt.xlim([-1, max_feat])
plt.tight_layout()
fname = (odir + "/random_forest_importance-%s-%s.png" % (max_feat, os.path.basename(sys.argv[0])) )
plt.savefig(fname, dpi=300)
#plt.show()


#import sys
#sys.exit()
