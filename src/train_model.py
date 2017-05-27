# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


#%% import libs and set global parameters
import pandas as pd
from pandas import DataFrame as df
import matplotlib.pyplot as plt
import lightgbm as lgb
import numpy as np
from sklearn.cross_validation import train_test_split
from feature_engineering import feture_engineer

seed = 3


#%% load data
train_data = pd.read_csv('../data/train_joined_data.csv')
test_data = pd.read_csv('../data/test_joined_data.csv')

train_data, test_data, used_features, categorical_feature = feture_engineer(
        train_data, test_data)

#%% train
data_x = train_data[used_features]
data_y = train_data['label']
x_train, x_val, y_train, y_val = train_test_split(data_x.values,
                                                    data_y.values,
                                                    test_size=0.2,
                                                    random_state=seed)

dtrain = lgb.Dataset(x_train, label=y_train, max_bin=500,
                     feature_name=used_features,
                     categorical_feature=categorical_feature)
dvalidation = lgb.Dataset(x_val, label=y_val, max_bin=500,
                     feature_name=used_features,
                     categorical_feature=categorical_feature,
                     reference=dtrain)

num_boost_round = 233
params = {}
params['application'] = 'binary'
params['boosting'] = 'gbdt'
params['num_leaves'] = 128
params['num_threads'] = 4
params['max_depth'] = -1
params['min_data_in_leaf'] = 10
params['min_sum_hessian_in_leaf'] = 1e-3
params['feature_fraction'] = 1
params['feature_fraction_seed'] = seed
params['bagging_fraction'] = 0.8
params['bagging_freq'] = 10
params['bagging_seed'] = seed
params['lambda_l1'] = 0
params['lambda_l2'] = 0
params['min_gain_to_split'] = 0
params['data_random_seed'] = seed
params['metric'] = 'binary_logloss'

clf = lgb.train(params, dtrain, num_boost_round=num_boost_round,
                valid_sets=[dtrain, dvalidation],
                valid_names=['train','val'],
                init_model=None, feature_name=used_features,
                categorical_feature=categorical_feature,
                early_stopping_rounds=0,
                learning_rates=lambda iter: 0.1 * (0.99 ** iter))


#%% test
x_test = test_data[used_features]
ypred = clf.predict(x_test.values, num_iteration=clf.best_iteration)
submission = df({'instanceID':np.arange(1,ypred.shape[0]+1),
'prob': ypred})
submission.to_csv('../submission/val_{0:.5f}.csv'.format(clf.eval_valid()[0][2]), index=False)
submission.to_csv('../submission/submission.csv', index=False)