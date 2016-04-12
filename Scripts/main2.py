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

def find_cols(all_col, key):
	cols = []
	for col in all_col:
		if key in col:
			cols.append(col)
	return(cols)

cols_ind = find_cols(df_all.columns, 'ind')

cols_red = []
for col in cols_ind:
	col_new = col.replace('ind', 'num')
	if col_new in df_all.columns and np.all((df_all[col_new]>0).astype(int)==df_all[col]):
		cols_red.append(col)

df_all.drop(cols_red, axis=1, inplace=True)

# remove columns starting with delta
cols_delta = find_cols(df_all.columns, 'delta')
df_all.drop(cols_delta, axis=1, inplace=True)

# remove columns with zero std
remove1 = []
for col in df_all.columns:
	if df_all[col].std()==0:
		remove1.append(col)

print('Columns with zero std')
print(remove1)
df_all.drop(remove1, axis=1, inplace=True)

# remove identical columns
remove2 = []
n_col = len(df_all.columns)
for i in range(n_col-1):
	col1 = df_all.columns[i]
	if col1 in remove2:
		continue
	for j in range(i+1, n_col):
		col2 = df_all.columns[j]
		if np.all(df_all[col1]==df_all[col2]):
			remove2.append(col2)

print('Identical columns')
print(remove2)
df_all.drop(remove2, axis=1, inplace=True)

# remove highly correlated columns
remove3 = []
n_col = len(df_all.columns)
for i in range(n_col-1):
	col1 = df_all.columns[i]
	if col1 in remove3:
		continue
	for j in range(i+1, n_col):
		col2 = df_all.columns[j]
		if abs(np.corrcoef(df_all[col1], df_all[col2])[0, 1])>0.999999:
			remove3.append(col2)

print('Highly correlated columns')
print(remove3)
df_all.drop(remove3, axis=1, inplace=True)

X = df_all.values
my_xgb = xgb_clf.my_xgb(obj='binary:logistic', eval_metric='auc', num_class=2, 
	nthread=10, silent=1, verbose_eval=50, eta=0.1, colsample_bytree=1, subsample=0.8, 
	max_depth=5, max_delta_step=0, gamma=0, alpha=0, param_lambda=1, n_fold=5, seed=0)

scores_col = {}
for i in range(len(df_all.columns)):
	col = df_all.columns[i]
	df_all_col = df_all[col]
	X = np.reshape(df_all_col.values, (-1, 1))
	X_col = X[:n_train, :]
	X_test_col = X[n_train:, :]
	try:
		y_pred, score = my_xgb.predict(X_col, y, X_test_col, 'meta')
	except:
		score = 0.5
	print(col, score)
	scores_col[col] = score

keys = np.array(list(scores_col.keys()))
sorted_ind = np.argsort(list(scores_col.values()))[::-1]
sorted_col = keys[sorted_ind]

df_all_sort = df_all[sorted_col]
X_all = df_all_sort.values
X = X_all[:n_train, :]
X_test = X_all[n_train:, :]

scores = []
y_pred_sum = np.zeros(X_test.shape[0])
R = 1000
for r in range(R):
	my_xgb = xgb_clf.my_xgb(obj='binary:logistic', eval_metric='auc', num_class=2, 
	nthread=10, silent=1, verbose_eval=50, eta=0.1, colsample_bytree=0.8, subsample=0.8, 
	max_depth=5, max_delta_step=0, gamma=0, alpha=0, param_lambda=1, n_fold=5, seed=r)
	cols = list(range(10))+list(np.random.choice(range(10, df_all.shape[1]), 100, False))
	y_pred, score = my_xgb.predict(X[:, cols], y, X_test[:, cols], 'meta')
	scores.append(score)
	y_pred_sum += score*y_pred

y_pred = y_pred_sum/np.sum(scores)

sub = pd.DataFrame(data={'ID':ids, 'TARGET':y_pred}, 
	columns=['ID', 'TARGET'])
my_dir = os.getcwd()+'/Santander/Subs/'
sub.to_csv(my_dir+'sub.csv', index=False)
