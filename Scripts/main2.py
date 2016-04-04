import os
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

remove = pickle.load(open(my_dir+'/Santander/Outputs/'+'remove.p', 'rb'))

df_all.drop(remove, axis=1, inplace=True)

# OHE
cols_dum = []
for col in df_all.columns:
	if df_all[col].dtype==np.int64 and len(np.unique(df_all[col]))>2 and len(np.unique(df_all[col]))<=100 and np.any(np.unique(df_all[col])%3!=0):
		print(col, len(np.unique(df_all[col])))
		cols_dum.append(col)

cols_bin = []
for col in df_all.columns:
	if df_all[col].dtype==np.int64 and len(np.unique(df_all[col]))==2:
		print(col, np.unique(df_all[col]))
		cols_bin.append(col)
		df_all[col] -= min(df_all[col])
		df_all[col] /= max(df_all[col])

cols_float = []
for col in df_all.columns:
	if df_all[col].dtype==np.float64:
		cols_float.append(col)

for col in cols_dum:
	dum = pd.get_dummies(df_all[col], prefix=col, drop_first=True)
	df_all.drop(col, axis=1, inplace=True)
	df_all = pd.concat([df_all, dum], axis=1)

X_all = df_all.values
X = X_all[:n_train, :]
X_test = X_all[n_train:, :]

scores = []
y_pred_sum = np.zeros(X_test.shape[0])
R = 1
for r in range(R):
	my_xgb = xgb_clf.my_xgb(obj='binary:logistic', eval_metric='auc', num_class=2, 
	nthread=10, silent=1, verbose_eval=50, eta=0.02, colsample_bytree=0.8, subsample=0.8, 
	max_depth=5, max_delta_step=0, gamma=0, alpha=0, param_lambda=1, n_fold=5, seed=r)
	y_pred, score = my_xgb.predict(X, y, X_test, 'meta')
	scores.append(score)
	y_pred_sum += y_pred

y_pred = y_pred_sum/R

sub = pd.DataFrame(data={'ID':ids, 'TARGET':y_pred}, 
	columns=['ID', 'TARGET'])
my_dir = os.getcwd()+'/Santander/Subs/'
sub.to_csv(my_dir+'sub.csv', index=False)
