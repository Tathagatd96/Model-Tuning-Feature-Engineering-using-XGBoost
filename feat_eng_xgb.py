# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 17:28:42 2019

@author: tatha
"""

import xgboost as xgb
from sklearn.model_selection import train_test_split
import pandas as pd

data = pd.read_csv("pima-indians-diabetes.csv")
print(data.describe())
print(data.keys())

X_data=data.drop(["Class","Group"],axis=1)
y_data=data["Class"]

dtrain = xgb.DMatrix(X_data,y_data)

params = {
        'objective':'binary:logistic',
        'max-depth':2,
        'silent':1,
        'eta':0.5
        }

num_rounds = 5

bst = xgb.train(params,dtrain,num_rounds)

tdump = bst.get_dump(fmap = "C:\\Users\\tatha\\.spyder-py3\\featmap.txt", with_stats = True)

for trees in tdump:
    print(trees)
  
    
xgb.plot_importance(bst, importance_type = 'gain', xlabel = 'Gain')