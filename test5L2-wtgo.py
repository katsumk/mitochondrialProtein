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

#chap6
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

#############################################################################
import sys
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


#X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
# 7割を学習データに、のころをテストデータ（正しい精度の計算）に
X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.3, random_state=0)

# 規格化をする
from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
X_train = stdsc.fit_transform(X_train)
X_test = stdsc.transform(X_test)

# ロジスティック回帰　L1 という方法で変数の数は少なくさせる
from sklearn.linear_model import LogisticRegression


lr = LogisticRegression(penalty='l2', C=10)
lr.fit(X_train, y_train)
print('Training accuracy:', lr.score(X_train, y_train))
print('Test accuracy:', lr.score(X_test, y_test))


y_pred = lr.predict(X_test)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print('Confusion matrix', confmat)

# Page 185, part1.
print(50 * '=')
print('Section: Optimizing the precision and recall of a classification model')
print(50 * '-')

print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred))
print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred))
print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred))


#####
print(50 * '=')
print('Coeff')
print(lr.coef_[0],
    lr.intercept_[0],
    lr.n_iter_[0])


print(50 * '=')
feat_labels = X.columns
print(50 * '-')
print('feat_labels')
for i in range(len(feat_labels)):
    list_item = feat_labels[i]
    print('{0}:{1}'.format(i+1, list_item))


# from  (0,1,2) 3, ... 1245. (=1249 - 3-1)
fig = plt.figure(1)
Xc = np.arange(1,lr.coef_.shape[1]+1)

plt.plot(Xc, lr.coef_[0])
#plt.plot(Xc, lr.coef_[0])

fname = (odir + "/lr2.coef-%s.png" % os.path.basename(sys.argv[0] ))
fig.savefig(fname, dpi=300)


importances = abs(lr.coef_[0])
#print (importances)
print(50 * '-')
indices = np.argsort(importances)[::-1]
print (indices)
print (feat_labels[indices])


max_feat = 20

#plt.title('Feature Importances')

fig = plt.figure(2)
plt.bar(range(max_feat),
        importances[indices[0:max_feat]],
        color='green',
        align='center')

plt.xticks(range(max_feat),
            feat_labels[indices[0:max_feat]],
           #indices[0:max_feat]+1,
           #feat_labels[indices[0:max_feat]],
	   rotation=90)
plt.xlim([-1, max_feat])
plt.tight_layout()
fname = (odir + "/lr2.coef-%s-%s.png" % (max_feat, os.path.basename(sys.argv[0]) ))

plt.savefig(fname, dpi=300)
#plt.show()


#import sys
#sys.exit()



#f = open('lr1-coef.txt', 'w') # 書き込みモードで開く
#f.write('intercept: %s' % lr.intercept_[0])
#f.write('coef: %s' % lr.coef_[0])
#f.close() # ファイルを閉じる

df=pd.DataFrame({ 'intercept':lr.intercept_[0], 'coef':lr.coef_[0] })
df.to_csv(odir+"/lr2-coef-%s.csv" % os.path.basename(sys.argv[0]) , sep=",")
