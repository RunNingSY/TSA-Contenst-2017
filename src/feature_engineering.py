# -*- coding: utf-8 -*-
"""
Created on Sat May 27 11:19:41 2017

@author: RunNing
"""

import pandas as pd
from pandas import DataFrame as df
import matplotlib.pyplot as plt
import numpy as np


def feture_engineer(train_data, test_data):
    train_data['residenceProvince'] = train_data['residence'] // 100
    train_data['hometownProvince'] = train_data['hometown'] // 100
    test_data['residenceProvince'] = test_data['residence'] // 100
    test_data['hometownProvince'] = test_data['hometown'] // 100
             
             
             
    used_features = \
    ['creativeID',
     'adID',
     'camgaignID',
     'advertiserID',
     'appID',
     'appPlatform',
    #'userID',
     'age',
     'gender',
     'education',
     'marriageStatus',
     'haveBaby',
    #'hometownProvince',
     'residenceProvince',
     'positionID',
    #'sitesetID',
     'positionType',
     'connectionType',
     'telecomsOperator']
    
    categorical_feature = \
    [
     'creativeID',
     'adID',
     'camgaignID',
     'advertiserID',
     'appID',
     'appPlatform',
    #'userID',
    #'age',
     'gender',
     'education',
     'marriageStatus',
     'haveBaby',
    #'hometownProvince',
     'residenceProvince',
    #'positionID',
    #'sitesetID',
     'positionType',
     'connectionType',
     'telecomsOperator']
    return train_data, test_data, used_features, categorical_feature