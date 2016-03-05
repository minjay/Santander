import os
import datetime
import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder

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

df_all.replace(-999999.0, -1, inplace=True)

df_all['saldo_var_diff'] = df_all['saldo_var30']-df_all['saldo_var42']

X_all = df_all.values
X = X_all[:n_train, :]

# train test split
# specify random seed
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=0)

## fit by xgb
xg_train = xgb.DMatrix(X_train, y_train)
xg_val = xgb.DMatrix(X_val, y_val)

# specify parameters for xgb
# no num_class!
param = {}
param['booster'] = "gbtree"
param['objective'] = 'binary:logistic'
param['eval_metric'] = 'auc'
param['nthread'] = 2
param['silent'] = 1

param['colsample_bytree'] = 0.8
param['subsample'] = 0.8
param['eta'] = 0.1

num_round = 10000

evallist  = [(xg_train,'train'), (xg_val,'eval')]
bst = xgb.train(param, xg_train, num_round, evallist, early_stopping_rounds=100)

X_test = X_all[n_train:, :]
xg_test = xgb.DMatrix(X_test)

# the pred is prob
y_pred = bst.predict(xg_test, ntree_limit=bst.best_iteration)
sub = pd.DataFrame(data={'ID':ids, 'TARGET':y_pred}, 
	columns=['ID', 'TARGET'])
my_dir = os.getcwd()+'/Santander/Subs/'
sub.to_csv(my_dir+'sub.csv', index=False)