# -*- coding: utf-8 -*-
"""
Created on Fri May 26 09:12:51 2017

@author: RunNing
"""


import pandas as pd
from pandas import DataFrame as df


def join_data(dataset='train'):
    """
    Join data for training/testing
    
    Join data from different csv files for training/testing
    and write joined data to local disk.
    
    Parameters
    ----------
    dataset: {'train', 'test'}
        
    Raises
    ------
    ValueError
        when `dataset` is not in ['train', 'test'].
        
    """
    if (dataset != 'train' and dataset != 'test'):
        raise ValueError("`dataset` must be 'train' or 'test'")
    if dataset == 'train':
        ori_data = pd.read_csv('../data/train.csv')
    else:
        ori_data = pd.read_csv('../data/test.csv')
    user = pd.read_csv('../data/user.csv')
    position = pd.read_csv('../data/position.csv')
    ad = pd.read_csv('../data/ad.csv')
    
    data = df()
    data['label'] = ori_data['label']
    data['clickTime'] = ori_data['clickTime']
    if dataset == 'train':
        data['conversionTime'] = ori_data['conversionTime']
    
    data['creativeID'] = ori_data['creativeID']
    ad_val = ad.sort_values(by='creativeID').values.copy()
    ad_indexed_creativeID = df(ad_val[:,1:], index=ad_val[:,0], columns=ad.columns[1:])
    creativeID_in_ori_data = ori_data['creativeID']
    tmp = ad_indexed_creativeID.ix[creativeID_in_ori_data]
    tmp = df(tmp.values.copy(), index=data.index, columns=tmp.columns)
    data = pd.concat([data, tmp], axis=1)
    
    data['userID'] = ori_data['userID']
    user_val = user.values.copy()
    user_indexed_userID = df(user_val[:,1:], index=user_val[:,0], columns=user.columns[1:])
    userID_in_ori_data = ori_data['userID']
    tmp = user_indexed_userID.ix[userID_in_ori_data]
    tmp = df(tmp.values.copy(), index=data.index, columns=tmp.columns)
    data = pd.concat([data, tmp], axis=1)
    
    data['positionID'] = ori_data['positionID']
    position_val = position.sort_values(by='positionID').values.copy()
    position_indexed_positionID = df(position_val[:,1:], index=position_val[:,0], columns=position.columns[1:])
    positionID_in_ori_data = ori_data['positionID']
    tmp = position_indexed_positionID.ix[positionID_in_ori_data]
    tmp = df(tmp.values.copy(), index=data.index, columns=tmp.columns)
    data = pd.concat([data, tmp], axis=1)
    
    data['connectionType'] = ori_data['connectionType']
    data['telecomsOperator'] = ori_data['telecomsOperator']
    if dataset == 'train':
        data.to_csv('../data/train_joined_data.csv', index=False)
    else:
        data.to_csv('../data/test_joined_data.csv', index=False)
        

if __name__ == '__main__':
    join_data(dataset='train')
    join_data(dataset='test')
