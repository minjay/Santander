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
import pyexcel as pe
import pyexcel.ext.xlsx
from sklearn.feature_selection import *
from sklearn.preprocessing import *

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
df_train = df_all.iloc[:n_train, :]

def feature_selection(train_df_feature, train_df_label, test_df_feature, p):
    X_bin = Binarizer().fit_transform(scale(train_df_feature))
    selectChi2 = SelectPercentile(chi2, percentile=p).fit(X_bin, train_df_label)
    selectF_classif = SelectPercentile(f_classif, percentile=p).fit(train_df_feature, train_df_label)
    chi2_selected = selectChi2.get_support()
    chi2_selected_features = [ f for i,f in enumerate(train_df_feature.columns) if chi2_selected[i]]
    print('Chi2 selected {} features {}.'.format(chi2_selected.sum(),
       chi2_selected_features))
    f_classif_selected = selectF_classif.get_support()
    f_classif_selected_features = [ f for i,f in enumerate(train_df_feature.columns) if f_classif_selected[i]]
    print('F_classif selected {} features {}.'.format(f_classif_selected.sum(),
       f_classif_selected_features))
    selected = chi2_selected & f_classif_selected
    print('Chi2 & F_classif selected {} features'.format(selected.sum()))
    features = [ f for f,s in zip(train_df_feature.columns, selected) if s]
    return features


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
param['max_depth'] = 5

num_round = 10000



np.random.seed(0)
n_fold = 5
# Stratified CV seems to be better
kf = StratifiedKFold(y, n_folds=n_fold, shuffle=True)
i = 0
best_score = []
y_pred_sum = np.zeros(df_all.shape[0]-n_train)

for train, val in kf:
	i += 1
	print(i)
	feat_sel = feature_selection(df_train.iloc[train,:], y[train], df_train.iloc[val, :], 95) 
	X = df_train[feat_sel].values
	X_train, X_val, y_train, y_val = X[train], X[val], y[train], y[val]
	xg_train = xgb.DMatrix(X_train, y_train)
	xg_val = xgb.DMatrix(X_val, y_val)
	evallist  = [(xg_train,'train'), (xg_val,'eval')]
	bst = xgb.train(param, xg_train, num_round, evallist, early_stopping_rounds=30)
	best_score += [bst.best_score]
	df_test = df_all.iloc[n_train:, :]
	X_test = df_test[feat_sel].values
	xg_test = xgb.DMatrix(X_test)
	y_pred = bst.predict(xg_test, ntree_limit=bst.best_iteration)
	y_pred_sum = y_pred_sum+y_pred

print(np.mean(best_score), np.std(best_score))




# load interactions
book = pe.get_book(file_name=my_dir+'/Santander/Interactions/XgbFeatureInteractions_zero_dup_removed.xlsx')
sheets = book.to_dict()
feat_one_way = np.array(sheets['Interaction Depth 0'])[1:, 0]
remove = np.setdiff1d(df_all.columns, feat_one_way)

X_all = df_all.values
X = X_all[:n_train, :]
X_test = X_all[n_train:, :]
my_xgb = xgb_clf.my_xgb(obj='binary:logistic', eval_metric='auc', num_class=2, 
	nthread=10, silent=1, verbose_eval=50, eta=0.1, colsample_bytree=0.8, subsample=0.8, 
	max_depth=5, max_delta_step=0, gamma=0, alpha=0, param_lambda=1, n_fold=5, seed=0)
y_pred, score_base = my_xgb.predict(X, y, X_test, 'meta')

feat_two_way = np.array(sheets['Interaction Depth 1'])[1:, 0]
for i in range(len(feat_two_way)):
	pair = feat_two_way[i]
	feat1, feat2 = pair.split('|')
	if feat1==feat2:
		continue
	df_all[feat1+'_'+feat2+'_minus'] = df_all[feat1]-df_all[feat2]
	df_all_drop = df_all.drop(remove, axis=1, inplace=False)
	X_all = df_all_drop.values
	X = X_all[:n_train, :]
	X_test = X_all[n_train:, :]
	y_pred, score = my_xgb.predict(X, y, X_test, 'meta')
	if score>score_base:
		score_base = score
		print('Adding...')
	else:
		df_all.drop(feat1+'_'+feat2+'_minus', axis=1, inplace=True)

sub = pd.DataFrame(data={'ID':ids, 'TARGET':y_pred}, 
	columns=['ID', 'TARGET'])
my_dir = os.getcwd()+'/Santander/Subs/'
sub.to_csv(my_dir+'sub.csv', index=False)
