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

remove = remove1+remove2

# baseline
X_all = df_all.values
X = X_all[:n_train, :]
X_test = X_all[n_train:, :]

my_xgb = xgb_clf.my_xgb(obj='binary:logistic', eval_metric='auc', num_class=2, 
    nthread=15, silent=1, verbose_eval=50, eta=0.1, colsample_bytree=0.8, subsample=0.8, 
    max_depth=5, max_delta_step=0, gamma=0, alpha=0, param_lambda=1, n_fold=5, seed=0)

y_pred, score_baseline = my_xgb.predict(X, y, X_test, 'meta')

# two-way interaction
add = []
scores = []
n_col = len(df_all.columns)-1
for i in range(n_col-1):
	col1 = df_all.columns[i]
	for j in range(i+1, n_col):
		col2 = df_all.columns[j]
		if np.corrcoef(df_all[col1], df_all[col2])[0, 1]>0.95:
			print('---------------------------------')
			print('Checking '+str(i)+'-'+str(j)+'...')
			df_all['diff'] = df_all[col1]-df_all[col2]
			X_all = df_all.values
			df_all.drop('diff', axis=1, inplace=True)
			X = X_all[:n_train, :]
			X_test = X_all[n_train:, :]
			my_xgb = xgb_clf.my_xgb(obj='binary:logistic', eval_metric='auc', num_class=2, 
    			nthread=15, silent=1, verbose_eval=False, eta=0.1, colsample_bytree=0.8, subsample=0.8, 
    			max_depth=5, max_delta_step=0, gamma=0, alpha=0, param_lambda=1, n_fold=5, seed=0)
			y_pred, score_add = my_xgb.predict(X, y, X_test, 'meta')
			if score_add>score_baseline:
				print('Adding '+col1+'-'+col2+'...')
				add.append((col1, col2))
				scores.append(score_add)
			print('---------------------------------')

# save
pickle.dump(remove, open(my_dir+'/Santander/Outputs/'+'remove.p', 'wb'))
pickle.dump(add, open(my_dir+'/Santander/Outputs/'+'add_two_way.p', 'wb'))
pickle.dump(scores, open(my_dir+'/Santander/Outputs/'+'scores_two_way.p', 'wb'))
