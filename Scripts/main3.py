import os
import re
import pickle
import datetime
import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedKFold
from scipy import sparse
from sklearn.manifold import TSNE
from statsmodels.distributions.empirical_distribution import ECDF

import sys
my_dir = os.getcwd()
sys.path.append(my_dir+'/Santander/Scripts')

import xgb_clf

my_dir = os.getcwd()
df_train = pd.read_csv(my_dir+'/Santander/Data/train.csv')
df_test = pd.read_csv(my_dir+'/Santander/Data/test.csv')

n_train = df_train.shape[0]
y = df_train['TARGET'].values
df_train.drop('TARGET', axis=1, inplace=True)
ids = df_test['ID']

# combine train and test
df_all = pd.concat([df_train, df_test], axis=0, ignore_index=True)

# drop ids
df_all.drop('ID', axis=1, inplace=True)

one_way = pd.read_csv(my_dir+'/Santander/Results/one_way.csv')
two_way = pd.read_csv(my_dir+'/Santander/Results/two_way_comparison.csv')

df_all = df_all[one_way['feature']]

X_all = df_all.values
X = X_all[:n_train, :]
X_test = X_all[n_train:, :]

param = {}
param['booster'] = "gbtree"
param['objective'] = 'binary:logistic'
param['eval_metric'] = 'auc'
param['nthread'] = 10
param['silent'] = 1

param['colsample_bytree'] = 0.8
param['subsample'] = 0.8
param['eta'] = 0.01
param['max_depth'] = 5

num_round = 1000

R = 1000
y_pred_sum = np.zeros(X_test.shape[0])
for r in range(R):
	cols = np.random.choice(range(X.shape[1]), 200)
	xg_train = xgb.DMatrix(X[:, cols], y)
	evallist  = [(xg_train,'train')]
	bst = xgb.train(param, xg_train, num_round, evallist)
	xg_test = xgb.DMatrix(X_test[:, cols])
	y_pred = (bst.predict(xg_test)+bst.predict(xg_test, ntree_limit=900)+
	bst.predict(xg_test, ntree_limit=800))/3
	y_pred_sum += y_pred

y_pred = y_pred_sum/R
sub = pd.DataFrame(data={'ID':ids, 'TARGET':y_pred}, 
	columns=['ID', 'TARGET'])
my_dir = os.getcwd()+'/Santander/Subs/'
sub.to_csv(my_dir+'sub.csv', index=False)