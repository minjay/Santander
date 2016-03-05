import os
import datetime
import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import KFold
from scipy import sparse

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
remove = []
for col in df_all.columns:
	if df_all[col].std()==0:
		remove.append(col)

print('Columns with zero std')
print(remove)
df_all.drop(remove, axis=1, inplace=True)

remove = []
# remove identical columns
n_col = len(df_all.columns)
for i in range(n_col-1):
	col1 = df_all.columns[i]
	if col1 in remove:
		continue
	for j in range(i+1, n_col):
		col2 = df_all.columns[j]
		if np.all(df_all[col1]==df_all[col2]):
			remove.append(col2)

print('Identical columns')
print(remove)
df_all.drop(remove, axis=1, inplace=True)

df_all['saldo_var_diff'] = df_all['saldo_var30']-df_all['saldo_var42']

X_all = df_all.values
X = X_all[:n_train, :]

# specify parameters for xgb
# no num_class!
param = {}
param['booster'] = "gbtree"
param['objective'] = 'binary:logistic'
param['eval_metric'] = 'auc'
param['nthread'] = 10
param['silent'] = 1

param['colsample_bytree'] = 0.8
param['subsample'] = 0.8
param['eta'] = 0.1

num_round = 10000

X_test = X_all[n_train:, :]
xg_test = xgb.DMatrix(X_test)

np.random.seed(0)
n_fold = 5
kf = KFold(n_train, n_folds=n_fold, shuffle=True)
i = 0
best_score = []
y_pred_sum = np.zeros(X_test.shape[0])

for train, val in kf:
	i += 1
	print(i)
	X_train, X_val, y_train, y_val = X[train], X[val], y[train], y[val]
	xg_train = xgb.DMatrix(X_train, y_train)
	xg_val = xgb.DMatrix(X_val, y_val)
	evallist  = [(xg_train,'train'), (xg_val,'eval')]
	bst = xgb.train(param, xg_train, num_round, evallist, early_stopping_rounds=30)
	best_score += [bst.best_score]
	y_pred = bst.predict(xg_test, ntree_limit=bst.best_iteration)
	y_pred_sum = y_pred_sum+y_pred

print(np.mean(best_score), np.std(best_score))

y_pred = y_pred_sum/n_fold

sub = pd.DataFrame(data={'ID':ids, 'TARGET':y_pred}, 
	columns=['ID', 'TARGET'])
my_dir = os.getcwd()+'/Santander/Subs/'
sub.to_csv(my_dir+'sub.csv', index=False)